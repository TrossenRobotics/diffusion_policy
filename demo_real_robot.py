"""
Usage:
python demo_real_robot.py -o <demo_save_dir> --follower_ip <ip> --leader_ip <ip>

Robot movement:
Move the leader arm - teleoperation.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.leader_arm_shared_memory import LeaderArm
from diffusion_policy.real_world.trossen_arm_controller import TrossenArmController
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--follower_ip', '-fi', required=True, help="Follower arm IP address e.g. 192.168.1.3")
@click.option('--leader_ip', '-li', required=True, help="Leader arm IP address e.g. 192.168.1.2")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robots joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between reading leader pose and executing on follower in Sec.")
def main(output, follower_ip, leader_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1/frequency
    # home position — both arms start here before teleoperation
    home_joints = np.array([0.0, np.pi/3, np.pi/6, np.pi/5, 0.0, 0.0, 0.0])
    init_joints_pos = home_joints if init_joints else None

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            LeaderArm(
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
                # recording resolution
                obs_image_resolution=(1280,720),
                frequency=frequency,
                init_joints=init_joints, # unused when robot= is provided
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                robot=robot,
            ) as env:
            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_exposure()
            # realsense white balance
            env.realsense.set_white_balance()

            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            target_pose = np.array(state['ActualTCPPose'])
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                stage = key_counter[Key.space]

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                precise_wait(t_sample)
                # get teleop command from leader arm
                # action = 6D EEF pose + 1D gripper width (meters) = 7D
                leader_state = leader.get_state()
                target_pose = np.array(leader_state['LeaderTCPPose'])
                gripper_width = float(leader_state['LeaderGripperPos'])
                target_action = np.append(target_pose, gripper_width)

                # execute teleop command
                env.exec_actions(
                    actions=[target_action],
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
