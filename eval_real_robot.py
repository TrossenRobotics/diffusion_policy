"""
Usage:
uv run python eval_real_robot.py -i <ckpt_path> -o <save_dir> \
    --follower_ip <follower_ip> --leader_ip <leader_ip>

================ Human in control ==============
Robot movement:
Move the LEADER arm to teleoperate the follower (the follower mirrors the
leader's absolute 6-DOF pose, exactly like demo_real_robot.py).

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly!

Recording control:
Press "S" to stop evaluation and gain control back.
Press "I" to intervene (DAgger). Two modes, set via --dagger_mode:
  absolute (default): pauses recording, leader arm auto-syncs to the follower's current
    pose (do not touch either arm while this happens), then hands teleop control to you.
  relative: no pause, no leader movement -- teleop resumes immediately using a fixed
    offset from the leader's live pose (clutch-style).
Either way, press "S" when done to save the whole episode (policy portion + your
recovery) into the DAgger dataset directory.
"""

# %%
import time
import json
import shutil
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.real_world.real_env import RealEnv
# Trossen teleop (leader) + follower controller replace the UR5 SpaceMouse setup.
from diffusion_policy.real_world.leader_arm_shared_memory import LeaderArm
from diffusion_policy.real_world.trossen_arm_controller import TrossenArmController
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform


OmegaConf.register_new_resolver("eval", eval, replace=True)

# matches TrossenArmController's default; used to clip the relative-mode gripper target.
GRIPPER_MAX_WIDTH = 0.044


def pose_to_mat(pose):
    "[x,y,z,rx,ry,rz] (angle-axis) -> 4x4 SE(3) matrix"
    mat = np.eye(4)
    mat[:3, 3] = pose[:3]
    mat[:3, :3] = st.Rotation.from_rotvec(pose[3:6]).as_matrix()
    return mat


def mat_to_pose(mat):
    "4x4 SE(3) matrix -> [x,y,z,rx,ry,rz] (angle-axis)"
    pose = np.zeros(6)
    pose[:3] = mat[:3, 3]
    pose[3:6] = st.Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return pose


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--follower_ip', '-fi', required=True, help="Follower arm IP, e.g. 192.168.1.4")
@click.option('--leader_ip', '-li', required=True, help="Leader arm IP, e.g. 192.168.1.2")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--dagger_output', '-do', default=None, help="Directory to save DAgger intervention episodes. Defaults to '<output>_dagger'.")
@click.option('--dagger_sync_duration', '-dsd', default=5.0, type=float, help="Seconds for the leader arm to auto-sync to the follower's pose during an intervention. Only used in --dagger_mode=absolute.")
@click.option('--dagger_mode', '-dm', default='absolute', type=click.Choice(['absolute', 'relative']),
    help="absolute: leader physically syncs to the follower's pose (pauses recording, ~dagger_sync_duration). "
         "relative: no leader movement or pause -- a fixed offset is computed once at intervention time and "
         "composed onto the leader's live pose (clutch-style), so teleop resumes immediately.")
def main(input, output, follower_ip, leader_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints,
    steps_per_inference, max_duration,
    frequency, command_latency,
    dagger_output, dagger_sync_duration, dagger_mode):
    if dagger_output is None:
        dagger_output = f'{output.rstrip("/")}_dagger'
    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    elif 'robomimic' in cfg.name:
        # BCRNN model
        policy: BaseImagePolicy
        policy = workspace.model

        device = torch.device('cuda')
        policy.eval().to(device)

        # BCRNN always has action horizon of 1
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)

    elif 'ibc' in cfg.name:
        policy: BaseImagePolicy
        policy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096

        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = 1
        delta_action = cfg.task.dataset.get('delta_action', False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    # action dimensionality the policy was trained with (6 for Trossen, 2 for sim pusht)
    action_dim = cfg.task.shape_meta['action']['shape'][0]
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)
    print("action_dim:", action_dim)

    # home position for both arms (7 joints), used only when --init_joints is set
    home_joints = np.array([0.0, np.pi/3, np.pi/6, np.pi/5, 0.0, 0.0, 0.0])
    init_joints_pos = home_joints if init_joints else None

    # DAgger dataset: episodes with a mid-episode intervention are routed here instead
    # of the regular eval output dir. Just a second Zarr store + video dir, no hardware.
    dagger_output_dir = pathlib.Path(dagger_output)
    dagger_video_dir = dagger_output_dir.joinpath('videos')
    dagger_video_dir.mkdir(parents=True, exist_ok=True)
    dagger_zarr_path = str(dagger_output_dir.joinpath('replay_buffer.zarr').absolute())
    dagger_replay_buffer = ReplayBuffer.create_from_path(zarr_path=dagger_zarr_path, mode='a')
    print(f"DAgger episodes will be saved to {dagger_output_dir}")

    with SharedMemoryManager() as shm_manager:
        # Build the leader (teleop input) and the follower (TrossenArmController),
        # then hand the follower to RealEnv via robot=... so RealEnv does not try to
        # create its own UR5 RTDE controller.
        with LeaderArm(
                shm_manager=shm_manager,
                leader_ip=leader_ip,
                frequency=100,
                init_joints_pos=init_joints_pos,
            ) as leader, \
            TrossenArmController(
                shm_manager=shm_manager,
                follower_ip=follower_ip,
                frequency=125,
                init_joints_pos=init_joints_pos,
            ) as robot, \
            RealEnv(
                output_dir=output, 
                robot_ip=follower_ip,  # unused when robot= is provided
                frequency=frequency,
                n_obs_steps=n_obs_steps,
                obs_image_resolution=obs_res,
                obs_float32=True,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                robot=robot) as env:
            cv2.setNumThreads(1)

            # Match data collection: demo_real_robot.py used auto exposure / white balance.
            env.realsense.set_exposure()
            env.realsense.set_white_balance()

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == action_dim
                del result

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['TargetTCPPose']
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = np.minimum(vis_img, match_img)

                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    key_stroke = cv2.pollKey()
                    if key_stroke == ord('q'):
                        # Exit program
                        env.end_episode()
                        exit(0)
                    elif key_stroke == ord('c'):
                        # Exit human control loop
                        # hand control over to the policy
                        break

                    precise_wait(t_sample)
                    # get teleop command from leader arm
                    # action = 6D EEF pose + 1D gripper width = 7D
                    leader_state = leader.get_state()
                    target_pose = np.array(leader_state['LeaderTCPPose'])
                    gripper_width = float(leader_state['LeaderGripperPos'])
                    target_action = np.append(target_pose, gripper_width)

                    # execute teleop command
                    env.exec_actions(
                        actions=[target_action],
                        timestamps=[t_command_target-time.monotonic()+time.time()])
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                intervened = False
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # index start_episode() used for this episode's video dir; captured now
                    # because pop_episode() (used if we later route this to DAgger) removes
                    # it from replay_buffer.n_episodes.
                    episode_id_at_start = env.replay_buffer.n_episodes
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions.
                        # TODO(abhi): delta actions
                        this_target_poses = action.astype(np.float64)

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])


                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break
                        elif key_stroke == ord('i'):
                            # ===== DAgger intervention =====
                            intervened = True
                            sync_row_start = None
                            sync_row_end = None
                            offset_mat = None
                            gripper_offset = None

                            if dagger_mode == 'absolute':
                                # exact row index the episode's low-dim data is at right
                                # now. Everything from this row up to sync_row_end (set
                                # below) is the leader-sync pause window, trimmed
                                # afterward by trim_dagger_sync.py.
                                sync_row_start = len(env.obs_accumulator)
                                print('Intervening! Pausing recording, leader syncing to follower pose...')
                                env.pause_recording()

                                # freeze target: exactly where the follower currently is
                                follower_state = env.get_robot_state()
                                sync_pose = np.array(follower_state['ActualTCPPose'])
                                sync_gripper = float(np.array(follower_state['gripper_position']).reshape(-1)[0])

                                sync_vis = env.get_obs()[f'camera_{vis_camera_idx}'][-1].copy()
                                cv2.putText(sync_vis, 'SYNCING - do not touch either arm',
                                    (10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1, color=(0,0,255))
                                cv2.imshow('default', sync_vis[...,::-1])
                                cv2.waitKey(1)

                                leader.move_to_pose(sync_pose, sync_gripper,
                                    duration=dagger_sync_duration, wait=True)
                                env.resume_recording()
                            else:
                                # relative (clutch): no leader movement, no pause -- a
                                # fixed offset is computed once from the leader/follower's
                                # CURRENT poses, then applied to the leader's live pose
                                # every step from here on (see CLAUDE.md DAgger section).
                                print('Intervening! Clutching in, teleop resumes immediately (no sync pause).')
                                leader_state = leader.get_state()
                                follower_state = env.get_robot_state()
                                leader_pose0 = np.array(leader_state['LeaderTCPPose'])
                                leader_gripper0 = float(leader_state['LeaderGripperPos'])
                                follower_pose0 = np.array(follower_state['ActualTCPPose'])
                                follower_gripper0 = float(np.array(follower_state['gripper_position']).reshape(-1)[0])
                                offset_mat = np.linalg.inv(pose_to_mat(leader_pose0)) @ pose_to_mat(follower_pose0)
                                gripper_offset = follower_gripper0 - leader_gripper0

                            print('Teleop control! Move the leader arm. Press S to end episode.')

                            # ===== DAgger recovery teleop (same episode, still recording) =====
                            t_start_dagger = time.monotonic()
                            iter_idx_dagger = 0
                            while True:
                                t_cycle_end_dagger = t_start_dagger + (iter_idx_dagger + 1) * dt
                                t_sample_dagger = t_cycle_end_dagger - command_latency
                                t_command_target_dagger = t_cycle_end_dagger + dt

                                dagger_obs = env.get_obs()
                                if dagger_mode == 'absolute' and sync_row_end is None:
                                    # this first post-resume get_obs() is exactly where the
                                    # accumulator catches its internal clock up to real time
                                    # in one shot -- read the row index right after it happens.
                                    sync_row_end = len(env.obs_accumulator)
                                dagger_vis = dagger_obs[f'camera_{vis_camera_idx}'][-1].copy()
                                cv2.putText(dagger_vis, 'DAgger recovery - Press S to end episode',
                                    (10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1, color=(255,255,255))
                                cv2.imshow('default', dagger_vis[...,::-1])

                                key_stroke_dagger = cv2.pollKey()
                                if key_stroke_dagger == ord('s'):
                                    env.end_episode()
                                    print('Stopped.')
                                    break

                                precise_wait(t_sample_dagger)
                                leader_state = leader.get_state()
                                if dagger_mode == 'absolute':
                                    dagger_target_pose = np.array(leader_state['LeaderTCPPose'])
                                    dagger_gripper_width = float(leader_state['LeaderGripperPos'])
                                else:
                                    leader_pose_now = np.array(leader_state['LeaderTCPPose'])
                                    leader_gripper_now = float(leader_state['LeaderGripperPos'])
                                    dagger_target_pose = mat_to_pose(pose_to_mat(leader_pose_now) @ offset_mat)
                                    dagger_gripper_width = float(np.clip(
                                        leader_gripper_now + gripper_offset, 0.0, GRIPPER_MAX_WIDTH))
                                # only append gripper for tasks trained with a 7D
                                # (pose+gripper) action space -- matches how
                                # RealEnv.exec_actions() itself gates gripper scheduling.
                                if action_dim > 6:
                                    dagger_target_action = np.append(dagger_target_pose, dagger_gripper_width)
                                else:
                                    dagger_target_action = dagger_target_pose

                                env.exec_actions(
                                    actions=[dagger_target_action],
                                    timestamps=[t_command_target_dagger - time.monotonic() + time.time()])
                                precise_wait(t_cycle_end_dagger)
                                iter_idx_dagger += 1
                            break

                        # auto termination (timeout only).
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()

                print("Stopped.")

                if intervened:
                    # route the whole episode (policy portion + human recovery) into the
                    # DAgger dataset instead of the regular eval output.
                    episode = env.replay_buffer.pop_episode()
                    dagger_replay_buffer.add_episode(episode, compressors='disk')
                    dagger_episode_id = dagger_replay_buffer.n_episodes - 1
                    src_video_dir = env.video_dir.joinpath(str(episode_id_at_start))

                    # record the exact sync-pause row range measured live, so
                    # trim_dagger_sync.py can cut precisely instead of guessing from data.
                    if sync_row_end is not None:
                        sync_info = {
                            'start_idx': sync_row_start,
                            'end_idx': sync_row_end,
                            'dt': dt,
                        }
                        if src_video_dir.exists():
                            (src_video_dir / 'sync_window.json').write_text(json.dumps(sync_info))

                    dst_video_dir = dagger_video_dir.joinpath(str(dagger_episode_id))
                    if src_video_dir.exists():
                        shutil.move(str(src_video_dir), str(dst_video_dir))
                    print(f'Routed episode to DAgger dataset as episode {dagger_episode_id}: {dagger_output_dir}')



# %%
if __name__ == '__main__':
    main()
