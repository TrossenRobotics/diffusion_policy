# Diffusion Policy on Trossen WXAI-v0

Quick start for collecting demos, training, and evaluating a policy on the Trossen
leader/follower arms. Package manager is `uv`, Python 3.13.

## Install

```bash
git clone https://github.com/TrossenRobotics/diffusion_policy.git
cd diffusion_policy
git checkout create_pyproject
uv sync
```

> If `uv sync` fails with a `cmake_minimum_required`/"Compatibility with CMake < 3.5"
> error, run `CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync` instead.

## 1. Collect demonstrations

Teleoperate the follower with the leader arm and record episodes.

```bash
uv run python demo_real_robot.py -o data/<task>_real \
  --follower_ip 192.168.1.4 --leader_ip 192.168.1.2 --frequency 10
```

- Click the camera window so it has focus.
- `C` start recording an episode, `S` stop it, `Backspace` delete the last one, `Q` quit.
- Move the leader arm to teleoperate the follower.

## 2. Train

```bash
uv run python train.py --config-name=train_diffusion_unet_real_hybrid_workspace \
  task=<task> task.dataset_path=data/<task>_real logging.mode=offline
```

- `task=` picks the config under `diffusion_policy/config/task/` (e.g. `trossen_cleaning`,
  `trossen_stacking`). Copy an existing one and edit `shape_meta`/`dataset_path` for a
  new task.
- Checkpoints land in `data/outputs/<date>/<time>_.../checkpoints/`.

## 3. Evaluate

```bash
uv run python eval_real_robot.py -i <path-to-checkpoint>.ckpt -o data/eval_<task> \
  --follower_ip 192.168.1.4 --leader_ip 192.168.1.2
```

- `C` hands control to the policy, `S` stops the episode and hands control back to you.
- Keep a hand near the e-stop while the policy is driving.
