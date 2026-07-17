"""
Post-process a DAgger dataset dir recorded by eval_real_robot.py.

Each DAgger episode contains a leader-sync pause: when 'I' is pressed, recording is
"paused" while the leader arm auto-syncs to the follower's pose, then teleop resumes.
Pausing just stops feeding the low-dim accumulator -- it doesn't erase the elapsed wall
time, so when accumulation resumes it fills the gap by repeating the last known sample.
The net effect is a run of exact-duplicate rows (follower frozen, action = hold) baked
into the middle of the episode, roughly `dagger_sync_duration` seconds long -- which is
"do nothing" data that would actively teach a policy to stall mid-task if trained on.

eval_real_robot.py records the exact row range live in a `sync_window.json` sidecar next
to each intervened episode's videos: the accumulator's row index the instant 'I' was
pressed, and the row index right after the first post-resume get_obs(), which is exactly
when the accumulator's internal clock catches itself up to real time. This script trims
exactly that range from the low-dim replay buffer and re-encodes each camera's video
dropping the matching frame-time window, so video and low-dim stay frame-aligned. Writes
a cleaned copy; the input dataset is left untouched. Errors out on any episode missing
the sidecar rather than guessing.

Usage:
uv run python diffusion_policy/scripts/trim_dagger_sync.py \
    --input data/eval_out_dagger --output data/eval_out_dagger_clean
"""

if __name__ == "__main__":
    import sys
    import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import json
import shutil
import pathlib
import click
import av
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.real_world.video_recorder import VideoRecorder


def trim_video(video_path, cut_start_sec, cut_end_sec, fps, crf, thread_count):
    "Re-encode video_path, dropping frames whose presentation time falls in [cut_start_sec, cut_end_sec)."
    tmp_path = str(video_path) + '.trim.mp4'
    recorder = VideoRecorder.create_h264(
        fps=fps, crf=crf, thread_type='FRAME', thread_count=thread_count,
        input_pix_fmt='bgr24')
    recorder.start(tmp_path)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            t = float(frame.time)
            if cut_start_sec <= t < cut_end_sec:
                continue
            recorder.write_frame(frame.to_ndarray(format='bgr24'))
    recorder.stop()
    os.replace(tmp_path, video_path)


@click.command()
@click.option('--input', '-i', required=True, help='DAgger raw dataset dir (replay_buffer.zarr + videos/)')
@click.option('--output', '-o', required=True, help='Directory to write the cleaned dataset')
@click.option('--video_fps', default=30, type=int, help='Raw video capture fps (must match what eval_real_robot.py recorded with)')
@click.option('--video_crf', default=21, type=int, help='H.264 CRF for re-encoding (must match original)')
@click.option('--video_threads', default=3, type=int, help='Encoder thread count per video')
def main(input, output, video_fps, video_crf, video_threads):
    input = pathlib.Path(input)
    output = pathlib.Path(output)
    assert (input / 'replay_buffer.zarr').is_dir()

    if output.exists():
        click.confirm(f'{output} already exists! Overwrite?', abort=True)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    src_rb = ReplayBuffer.create_from_path(str(input / 'replay_buffer.zarr'), mode='r')
    out_rb = ReplayBuffer.create_from_path(str(output / 'replay_buffer.zarr'), mode='a')
    out_video_dir = output / 'videos'
    out_video_dir.mkdir(parents=True)

    n_trimmed_rows = 0
    for ep in range(src_rb.n_episodes):
        src_video_dir = input / 'videos' / str(ep)
        dst_video_dir = out_video_dir / str(ep)

        sidecar = src_video_dir / 'sync_window.json'
        if not sidecar.exists():
            raise RuntimeError(
                f'episode {ep}: no sync_window.json in {src_video_dir}. '
                'This episode was not recorded with the sync-window logging eval script '
                '(or was never intervened). Re-record it, or remove it from the input '
                'dataset before running this script.')
        info = json.loads(sidecar.read_text())
        run_start, run_end = info['start_idx'], info['end_idx']
        dt = info['dt']
        run_len = run_end - run_start
        cut_start_sec = run_start * dt
        cut_end_sec = run_end * dt

        episode = src_rb.get_episode(ep, copy=True)
        pose = episode['robot_eef_pose']
        print(f'episode {ep}: trimming {run_len} frozen rows '
              f'(idx {run_start}-{run_end - 1}, t={cut_start_sec:.2f}s-{cut_end_sec:.2f}s)')

        # continuity check: pose should barely change across the cut (leader was synced
        # to this exact follower pose before teleop resumed) -- only the fake elapsed
        # time is being removed, not a real motion. Scrub the cleaned video to
        # cut_start_sec to see the same thing visually: policy-driven approach cuts
        # straight into human-driven recovery, no frozen segment in between.
        before = pose[run_start - 1] if run_start > 0 else pose[run_start]
        after = pose[min(run_end, len(pose) - 1)]
        print(f'  pose just before cut: {np.round(before, 4)}')
        print(f'  pose just after cut:  {np.round(after, 4)}  (delta norm={np.linalg.norm(after - before):.4f}m)')

        keep = np.ones(len(pose), dtype=bool)
        keep[run_start:run_end] = False
        episode = {k: v[keep] for k, v in episode.items()}
        out_rb.add_episode(episode, compressors='disk')

        dst_video_dir.mkdir(parents=True)
        for video_path in sorted(src_video_dir.glob('*.mp4')):
            shutil.copy(video_path, dst_video_dir / video_path.name)
        for video_path in sorted(dst_video_dir.glob('*.mp4')):
            trim_video(video_path, cut_start_sec, cut_end_sec,
                fps=video_fps, crf=video_crf, thread_count=video_threads)

        n_trimmed_rows += run_len

    print(f'\n{src_rb.n_episodes} episodes trimmed, '
          f'{n_trimmed_rows} frozen rows removed total. Cleaned dataset written to {output}')


if __name__ == '__main__':
    main()
