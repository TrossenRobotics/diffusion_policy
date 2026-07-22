"""
Merge a base raw dataset dir with a DAgger raw dataset dir into a fresh combined raw
dataset dir.

Usage:
uv run python diffusion_policy/scripts/merge_dagger_dataset.py \
    --base data/cleaning_real --dagger data/cleaning_real_dagger \
    --output data/cleaning_real_combined
"""

if __name__ == "__main__":
    import sys
    import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import shutil
import pathlib
import click
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer


@click.command()
@click.option('--base', '-b', required=True, help='Base raw dataset dir (replay_buffer.zarr + videos/)')
@click.option('--dagger', '-d', required=True, help='DAgger raw dataset dir (replay_buffer.zarr + videos/)')
@click.option('--output', '-o', required=True, help='Directory to write the combined raw dataset')
def main(base, dagger, output):
    base = pathlib.Path(base)
    dagger = pathlib.Path(dagger)
    output = pathlib.Path(output)
    assert (base / 'replay_buffer.zarr').is_dir()
    assert (dagger / 'replay_buffer.zarr').is_dir()

    if output.exists():
        click.confirm(f'{output} already exists! Overwrite?', abort=True)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    print(f'Copying base dataset from {base}')
    out_zarr_path = output / 'replay_buffer.zarr'
    combined = ReplayBuffer.copy_from_path(
        str(base / 'replay_buffer.zarr'), store=zarr.DirectoryStore(str(out_zarr_path)))
    shutil.copytree(base / 'videos', output / 'videos')
    n_base_episodes = combined.n_episodes

    print(f'Appending DAgger episodes from {dagger}')
    dagger_rb = ReplayBuffer.create_from_path(str(dagger / 'replay_buffer.zarr'), mode='r')
    for i in range(dagger_rb.n_episodes):
        combined.add_episode(dagger_rb.get_episode(i, copy=True), compressors='disk')
        new_id = combined.n_episodes - 1
        shutil.copytree(dagger / 'videos' / str(i), output / 'videos' / str(new_id))

    print(f'Combined dataset written to {output}: '
          f'{n_base_episodes} base + {dagger_rb.n_episodes} DAgger = {combined.n_episodes} episodes')


if __name__ == '__main__':
    main()
