# Experiment Protocol: Slide-Cube Diffusion Policy Study

**Purpose of this document:** a single source of truth that (a) forces this experiment
to follow a rigorous, falsifiable, reproducible protocol instead of an informal "try it
and see" pass, and (b) is complete enough that someone else — a person collecting data,
collecting DAgger corrections, or reading your results later — can understand
exactly what was done and why, without asking you questions.

---


## 1. Experiment Questions

This experiment is designed to answer the following four questions:

- **RQ1 (camera ablation):** Using 100 teleop demonstrations of the
  slide-cube task recorded with all four cameras (`cam_right_wrist`, `cam_high`, `cam_low`,
  `cam_left_wrist`), what is the in-distribution success rate of a diffusion policy on the
  Trossen WXAI solo embodiment as a function of camera set trained and evaluated
  separately on (a) `cam_right_wrist` alone, (b) `cam_right_wrist` + `cam_high`, and (c)
  `cam_right_wrist` + `cam_left_wrist`? Recording keeps all four cameras so `cam_low` and
  three-/four-camera combinations remain available for a later experiment; only
  these three conditions are trained/evaluated in this round. DAgger (RQ2-RQ4) is
  performed only after this camera-ablation sweep, using the checkpoint selected from
  it.
- **RQ2 (in-distribution DAgger):** Does adding 50 DAgger-corrected episodes (collected from
  observed in-distribution failures) and fine-tuning from the base checkpoint increase
  the in-distribution success rate, compared to the base checkpoint?
- **RQ3 (out-of-distribution DAgger):** Does adding ~50 DAgger episodes
  collected out-of-distribution (cube placed outside the marked area, a different
  physical WXAI solo robot, and different lighting) improve performance in that OOD
  setting, compared to the in-distribution-only checkpoint?
- **RQ4 (naive-user robustness):** When an evaluator with no knowledge of
  the experiment or the robot's training distribution is simply told "evaluate my
  slide-bot," what failure modes appear, are they attributable to out-of-distribution
  cube placement or something else, and what could be done to address them?

---

## 2. Hypotheses

Each hypothesis below states a prediction plus the reasoning behind it, written
*before* running the relevant phase, not adjusted afterward to fit the results.
Predictions are stated as comparisons (which condition beats which, and roughly by how
much). Readers are welcome to add their own hypotheses here.

- **H1 (RQ1):** Success rate rises with camera count — one camera is weakest, two
  cameras do better (rough guess: ~50-60% with `cam_right_wrist` alone, ~70-80% with
  two cameras). Of the two-camera pairs, `cam_right_wrist`+`cam_left_wrist` is
  predicted to edge out `cam_right_wrist`+`cam_high` by a small margin (~5%).
  Reasoning: more camera views reduce occlusion and visual ambiguity, giving the
  policy more task-relevant state to condition on; and `cam_left_wrist` sits closer to
  the manipulation area than `cam_high`, so it should contribute more task-relevant
  detail (hence the small predicted edge for that pair).
- **H2 (RQ2):** In-distribution success improves after DAgger fine-tuning, clearly beating
  the base checkpoint (rough guess: ~50-60% → >90%). Reasoning: DAgger data is specifically the
  cases where the base policy was failing, so it targets the model's weak points
  rather than adding generic data, which should help training escape whatever local
  minimum the base policy was stuck in.
- **H3 (RQ3):** OOD success is low before OOD DAgger and improves substantially after
  it (rough guess: ~50-60% → ~80%+). Same reasoning as H2: DAgger supplies exactly the
  cases the policy is failing on, so it should be a targeted, efficient improvement.
- **H4 (RQ4):** Naive third-party evaluation surfaces more failure modes and a lower
  success rate than the controlled in-distribution eval.
  Reasoning: the real world has effectively unbounded variability, so covering all its
  edge cases is hard and will likely need many rounds of DAgger, not just the single
  out-of-distribution round in Phase 4.

---

## 3. Definitions

| Term | Definition |
|---|---|
| **Task** | Slide the red cube to a target/end position, gripper will be closed all the time and oriented straight down (~90°) for the entire episode:<br>1. Correct the cube's orientation first.<br>2. Then slide it using only axis-aligned moves — either up/down then left/right, or left/right then up/down — never a direct diagonal move.<br>3. Robot start pose: randomized, but always with the cube in the wrist camera's frame.<br>`[TODO: more detail to be added after watching a data collector run a trial pass, or any other reader's suggestion]` |
| **Marked area** | The physical region on the table within which the cube's start and end positions are randomized for in-distribution collection/eval. Marked with a lightly-drawn, reproducible outline so a data collector can recreate it without asking. The marked area must be fully visible in all four cameras. `[TODO: exact dimensions]` |
| **Cube final position marker** | The printed target at `assets/robot_alignment_marker_final.png`: a goal square marking where the cube should end up, plus slightly larger surrounding lines used to judge whether a trial counts as complete. |
| **In-distribution (ID) trial** | Cube start pose sampled from within the marked area (position + orientation), as during training data collection. |
| **Out-of-distribution (OOD) trial** | Cube start pose outside the marked area, and/or different robot unit, and/or different lighting, and/or different cube color. |
| **Success** | See §4 (Success Criteria) below. |

---

## 4. Success Criteria

The rubric below is complete enough that two different people scoring the same
trial video reach the same verdict, with no judgment calls needed:

- **Final position:** the trial is a success when the cube's final resting
  position lies within the marked target region — the cube's target marker on the
  table defines the tolerance.
- **Timing:** the task normally takes ~30 s; a trial must complete within
  **1 minute** of the policy taking control. Exceeding that is a failure.
- **Disqualifying events (not counted toward the success rate):**
  - The arm's start pose does not place the cube in the wrist camera's view. Set
    up each start pose so the cube is visible in the wrist camera; if it is not,
    disqualify the trial.
  - Any unforeseen external event during the trial that is not a policy failure
    (e.g. hardware fault, table bumped). These are excluded, not scored as
    failures.
- **Trial count & pose selection:** each camera-set condition is evaluated over at
  least 20 episodes, with the cube's start position/orientation for every trial
  randomized and fixed in advance — e.g. a physical template sheet with the cube's
  position/orientation cut out — before evaluating any of them.

This is the exact rubric every evaluator applies.

---

## 5. Experiment phases

### Phase 0 — Baseline data collection
- Collect **100 demonstrations** of the task (see §3 for the full task definition).
- Record with all 4 cameras, even for conditions that will later train on
  fewer: adding cameras after the fact is easy, but DAgger data is tied to a
  specific trained camera set, so capturing all views up front keeps every
  camera-subset option open.

### Phase 1 — Training (camera-ablation sweep)
- Train three checkpoints from the same 100-episode dataset (recorded with all four
  cameras), varying only the camera set used for training (per RQ1):
  1. `cam_right_wrist` only
  2. `cam_right_wrist` + `cam_high`
  3. `cam_right_wrist` + `cam_left_wrist`
- `cam_low`, and three-/four-camera combinations, are left for a later experiment —
  not trained/evaluated in this round.
- Hyperparameters are identical across all three runs; only the camera set
  changes. This keeps the sweep a clean camera-count comparison. Because all
  three runs use the same 100-episode dataset, a fixed *epoch* budget is a fair
  comparison (each run sees the same number of gradient updates).

**Training configuration (identical for all three runs)** — from
`train_diffusion_unet_real_hybrid_workspace.yaml`:

| Setting | Value |
|---|---|
| Policy | Diffusion Policy, UNet hybrid image, DDIM (`num_inference_steps=8`) |
| Horizon / obs steps / action steps | 16 / 2 / 8 |
| Optimizer | AdamW, `lr=1e-4`, cosine schedule, 500 warmup steps |
| Batch size | 64 |
| EMA | on (the deployed policy uses the EMA weights, not the raw weights) |
| Seed | 42 (the answer to life, the universe, and everything) |
| Training budget | fixed 1000 epochs — no loss-based early stopping |
| Checkpointing | every 50 epochs → 20 checkpoints retained per run |

**Stopping and checkpoint selection.** We do not stop or select on training
loss. A diffusion policy's training loss is a denoising (noise-prediction) MSE that
keeps dropping as the model overfits and correlates poorly with task success, and
this setup has no held-out validation split (`val_ratio=0`). Instead each run
trains the full fixed budget, and the checkpoint deployed for each camera set is the
one with the highest real-robot success rate (scored with the §4 rubric on the
§7 trial list), evaluated on the EMA weights. Training loss is kept only as a
divergence/sanity check.

Launch (once per camera-set condition):
```bash
uv run python train.py \
  --config-name=train_diffusion_unet_real_hybrid_workspace \
  task=<task config for this camera set> \
  logging.mode=offline
```
`[TODO: the three per-camera-set task config names.]`

### Phase 2 — In-distribution evaluation
- Evaluate all three checkpoints from Phase 1.
- Only test within the marked/defined area — do not place the cube out-of-distribution at
  this stage.
- Goal: map out *where* and *how* the policy breaks in-distribution, not just a single
  success-rate number.
- Deliverable: a short write-up of the failure modes you observe (and any other
  interesting observations), not just "N/M succeeded".

### Phase 3 — In-distribution DAgger
- Collect **50 DAgger episodes**, correcting the failures observed in Phase 2 (i.e.
  intervening when the policy gets stuck/fails within the marked area).
- Fine-tune from the Phase 1 checkpoint on base + DAgger data combined.
  `[TODO: add the fine-tuning config + exact commands for this run.]`
- Re-run the same Phase 2 evaluation protocol (same trial list, same rubric, same
  evaluator if possible) on the fine-tuned checkpoint.
- This is the direct test of RQ2 — did in-distribution success rate change, and by how
  much? Produce the
  same kind of write-up as Phase 2 (failure modes + any other observations).

### Phase 4 — Out-of-distribution DAgger
- Collect **50 DAgger episodes** with:
  - cube placed outside the marked area,
  - a different physical robot unit,
  - a different lighting condition,
  - a different-colored cube.
- Fine-tune from the Phase 3 checkpoint on base + Phase 3 DAgger + Phase 4 DAgger
  combined.
- Re-evaluate in the OOD setting only (not in-distribution), using the same rubric
  and the same kind of write-up as before (failure modes + any other observations).

### Phase 5 — Naive third-party evaluation
- Recruit someone with **no knowledge** of the experiment, the robot's training
  distribution, or the task's internal details.
- Instruction given to them: only "evaluate my slide-bot" (or similar), nothing more.
- Do not tell them where the cube "should" go or what counts as in-distribution vs.
  out-of-distribution.
- Observe and jot down: where they place the cube, what they expect to happen, and
  where/why it fails from their perspective — this is qualitative data about
  real-world robustness, not a success-rate number.
- Deliverable: the same kind of short write-up/notes as the earlier phases
  (observations, not a scored number).

---

## 6. Data collection protocol (for whoever is collecting demos or DAgger episodes)

- Task, orientation-then-slide behavior, closed gripper, marked-area randomization,
  minimal idle time.
- Data collection command (see `TROSSEN_README.md` for the general
  pipeline):
  ```bash
  uv run python demo_real_robot.py -o data/slide_cube_real \
    --follower_ip <follower_ip> --leader_ip <leader_ip> --frequency 10
  ```
- DAgger collection command:
  ```bash
  uv run python eval_real_robot.py -i <checkpoint> -o data/eval_slide_cube \
    --follower_ip <follower_ip> --leader_ip <leader_ip> --dagger_mode relative
  ```
- All DAgger collection (Phase 3 and Phase 4) uses `--dagger_mode relative`; see
  `TROSSEN_README.md` for details.


---

## 7. Evaluation protocol & sample size

- **Fixed trial list:** fix the set of eval start poses (position + orientation within
  the marked area) *before* Phase 2 begins, and reuse the identical set for every
  checkpoint evaluated in Phases 2-4 (in-distribution portion). 
  
  Method: a paper template with the cube's start positions/orientations cut out;
  for each trial, lay the template on the marked area, place the cube in the cutout,
  remove the template.
- **Trial count:** at least **20 episodes per condition**. 
 `[TODO: may raise above 20 if more trials turn out to be needed.]`
- **Video:** record every eval trial (the repo already saves eval videos by default —
  keep them, don't overwrite/delete after scoring, in case a scoring decision needs
  review later).

---

## 8. Results (fill in as each phase completes)

### Phase 2 — `cam_right_wrist` only, in-distribution
- Trials: `(pending)` — Success rate: `(pending)`
- Failure modes observed: `(pending)`

### Phase 2 — `cam_right_wrist` + `cam_high`, in-distribution
- Trials: `(pending)` — Success rate: `(pending)`
- Failure modes observed: `(pending)`

### Phase 2 — `cam_right_wrist` + `cam_left_wrist`, in-distribution
- Trials: `(pending)` — Success rate: `(pending)`
- Failure modes observed: `(pending)`

### Phase 3 — post in-distribution DAgger, in-distribution
- Trials: `(pending)` — Success rate: `(pending)`
- Comparison to Phase 2: `(pending)`
- Failure modes observed: `(pending)`

### Phase 4 — post OOD DAgger
- In-distribution trials: `(pending)` — Success rate: `(pending)`
- OOD trials: `(pending)` — Success rate: `(pending)`
- Failure modes observed: `(pending)`

### Phase 5 — naive evaluator
- Observations: `(pending)`

---
