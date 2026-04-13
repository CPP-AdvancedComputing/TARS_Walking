# TARS Walking — Imported Refactor Summary

Source: user-provided summary on 2026-04-12.

Status:
- This file is preserved as an important project reference.
- It contains useful bug history, architectural rationale, tool ideas, and TIDE workflow notes.
- If any item here conflicts with the user's direct instructions in this thread, the user's direct instructions remain authoritative.

Important conflict note:
- This imported summary describes a 2-phase tripod gait concept:
  - phase 0: `L0 + L3` support
  - phase 1: `L1 + L2` support
- In the current thread, the user explicitly defined the canonical gait as:
  - phase 0: `[1,0,0,1]`
  - phase 1: `[1,1,1,1]`
  - phase 2: `[0,1,1,0]`
  - then return to phase 0
- Therefore, the bug inventory and tooling guidance in this imported summary are useful, but the gait-phase definition in this file is not authoritative for current work.

---

## Preserved User Summary

TARS Walking — Refactor Summary
**Date:** 2026-04-12  
**Repo:** `/home/awebb/Research/TARS_Walking` (CPP-AdvancedComputing/TARS_Walking)  
**Author:** Claude Sonnet 4.6 via Claude Code CLI  

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [What Was Broken (Pre-Refactor State)](#2-what-was-broken)
3. [Fixes Applied](#3-fixes-applied)
4. [Architecture Decisions](#4-architecture-decisions)
5. [Test Results](#5-test-results)
6. [New Tools Added](#6-new-tools-added)
7. [Training on TIDE](#7-training-on-tide)
8. [How the User Prompted (Prompting Methodology)](#8-how-the-user-prompted)
9. [File Map](#9-file-map)
10. [Open Questions / Next Steps](#10-open-questions--next-steps)

---

## 1. Project Overview

TARS is a quadruped robot being trained to walk using PPO (Proximal Policy Optimization) via
`stable-baselines3` in a MuJoCo simulation. The robot has 4 legs (L0–L3), each with 3 joints
(shoulder prismatic, hip revolute, knee prismatic). The environment is a custom `gymnasium.Env`
defined in `tars_env.py`.

**Gait concept:** A *tripod gait* where the four physical legs are treated as three effective legs:
- **L0** — independent leg
- **L1 + L2** — physically *bound* (move as one unit)
- **L3** — independent leg

In each phase, one group supports the robot while the other swings forward:
- **Phase 0:** L0 + L3 planted (support); L1 + L2 swing
- **Phase 1:** L1 + L2 planted (support); L0 + L3 swing

The training pipeline uploads code to a remote JupyterHub GPU cluster (TIDE/Nautilus) and streams
output back locally.

---

## 2. What Was Broken

The codebase had **12 bugs** ranging from training-breaking to structural. Ordered by severity:

### Critical — Training Was Fundamentally Broken

**Bug 1: Entire reward was commented out**  
`tars_env.py:2031–2060`. Every reward term was inside a commented block. Only `progress_reward`
(raw x-displacement) survived. The optimal policy for this reward is to lunge forward and fall.

**Bug 2: progress_gate was not applied**  
The gate (multiplying upright posture × spin rate × vertical stability × anti-lunge × support
quality) was computed but unused. A forward fall was worth more than a careful step.

**Bug 3: 4D shared-pair action space**  
`ACTION_DIM = 4`: one swing angle, one swing length, one plant angle, one plant length — shared
across all legs in the same role. With L0 and L3 having different leg geometry (different
`UPPER_ROD_ANCHORS`, `LOWER_ROD_ANCHORS`, and servo z-offsets in the MJCF), the policy had no
axis on which to express "L0 needs a different position than L3." The asymmetry was structurally
uncorrectable.

**Bug 4: `PLANT_UNLOAD_PENALTY_SCALE = 1.0` defined but never used**  
The constant existed at line 192 but was never referenced in `step()`. The symptom being
debugged — planted legs lifting off — had no direct penalty signal.

### Significant — Would Prevent Good Behavior Even With Correct Reward

**Bug 5: Absolute XY world position in observation**  
`_get_obs()` concatenated `data.qpos` directly. `qpos[0]` and `qpos[1]` are absolute world X/Y.
The policy saw different inputs for the same physical configuration depending on where in the
world the robot stood. This breaks translation invariance and makes the policy harder to
generalize.

**Bug 6: Phase 0 always at reset**  
`start_phase = 0` was hardcoded with the comment *"while phase-1 transition dynamics are still
being debugged."* Phase 1 was never the starting state of an episode. PPO's advantage estimates
are strongest in early steps; phase 1 was only sampled mid-episode where gradients are noisy.

**Bug 7: Curriculum shrinks the policy's authority (backwards direction)**  
`CurriculumCallback` decayed `nominal_action_scale` from `1.0 → 0.35` over training. This
*reduced* how far the IK target could deviate from neutral as training progressed. A policy that
learned a 0.05 rad hip offset would, at 80% completion, only produce 0.018 rad. The effective
action space shrank non-stationarily. The correct direction is to start tight (let the IK handle
most of the work) and expand authority as the policy matures.

**Bug 8: `PHASE_STALL_PENALTY = 0.1` had no deterrent force**  
A stuck phase cost 0.1 per step past the 60-step timeout. A single forward lunge could produce
`PROGRESS_REWARD_SCALE * 0.1m = 2.0` in one step. No urgency to complete phase transitions.

**Bug 10: `_shaping_terms` double-counted `swing_vertical_checkpoint_reward`**  
In the non-foundation profile, line 1326 added the term once, then line 1327 added it multiplied
by 2.0 — quietly inflating one reward component by 3×.

### Structural — Root Cause of L1/L3 Asymmetry

**Bug 11: `_initialize_rod_targets` applied a shared angle shift to all swing-pair legs**  
The shift was derived from the mean angle difference between the two structural pairs. L1 and L3
(in the old pairing) have different `UPPER_ROD_ANCHORS` and `LOWER_ROD_ANCHORS`, so the same
shift produced incorrect targets for each. With the new pairing (L1+L2 bound), the fix applies
a per-leg shift calibrated to each leg's own geometry.

**Bug 12: `pair_lock_blend = 1.0` fought L1's natural equilibrium**  
Pair-state feedback drove each leg toward the pair mean joint state. If L1 naturally settled at
a different joint configuration than L3 (which it does, due to geometry), this feedback actively
pushed L1 away from its stable position toward the pair mean — a mechanically unstable position
for L1, likely causing it to unload. Fix: only apply pair lock to the physically bound pair
(L1+L2), not to independent legs (L0, L3).

### Pre-existing Test Issues

- **Hardcoded Windows paths** in all test files: `r"C:\Users\anike\tars-urdf\tars_mjcf.xml"`
- Phase-assuming tests that relied on `reset()` always starting at phase 0

---

## 3. Fixes Applied

### 3.1 Gait Pairing
```python
# Before
SUPPORT_PAIR = (0, 2)
SWING_PAIR   = (1, 3)

# After — tripod gait, L1+L2 are bound
SUPPORT_PAIR = (0, 3)   # independent legs
SWING_PAIR   = (1, 2)   # bound pair
BOUND_PAIR   = (1, 2)   # physically constrained — always sync these
```

### 3.2 Action Space: 4D → 6D
```python
# Before: one shared command per role
SWING_ANGLE_ACTION = 0; SWING_LENGTH_ACTION = 1
PLANT_ANGLE_ACTION = 2; PLANT_LENGTH_ACTION = 3

# After: per-unit commands (L0, L1+L2, L3)
L0_ANGLE_ACTION  = 0;  L0_LENGTH_ACTION  = 1
L12_ANGLE_ACTION = 2;  L12_LENGTH_ACTION = 3   # shared by bound pair
L3_ANGLE_ACTION  = 4;  L3_LENGTH_ACTION  = 5
```
L1 and L2 receive the same L12 command (they're bound), but L0 and L3 can now be
independently steered by the policy.

### 3.3 Observation Space: 46D → 41D
```python
# Before
np.concatenate([data.qpos,          # 19 (includes abs X, Y)
                data.qvel,          # 18
                body.xpos,          # 3  (absolute world position)
                body.xquat,         # 4
                clock])             # 2   → total 46

# After
np.concatenate([data.qpos[2:],      # 17 (height + quat + joints only)
                data.qvel,          # 18
                body.xquat,         # 4
                clock])             # 2   → total 41
```

### 3.4 Reward — Fully Restored
```python
reward = (
    progress_reward * progress_gate      # gated: no reward for falling
    + velocity_bonus * progress_gate
    + gait_reference_hip_reward          # tracking oscillator reference
    + gait_reference_leg_reward
    + shaping_reward                     # contact, sync, swing, lateral
    + swing_lift_reward                  # swing feet actually lifting
    + swing_plant_penalty                # penalize swing feet staying down
    + tilt_penalty                       # soft stability guardrails
    + height_penalty
    + vertical_oscillation_penalty
    + vertical_velocity_penalty
    + fall_penalty                       # -8.0 on fall
    + phase_contact_penalty              # wrong contact pattern
    + early_swing_touchdown_penalty
    + plant_unload_penalty               # planted legs lifting off (new)
)
```

### 3.5 Phase at Reset — Randomized
```python
# Before
start_phase = 0  # hardcoded

# After
start_phase = int(self.np_random.integers(0, 2))  # 50/50
```

### 3.6 Curriculum — Direction Reversed
```python
# Before: policy starts with full authority and shrinks
nominal_start=1.0, nominal_end=0.35

# After: policy starts tight, earns more authority over time
nominal_start=0.35, nominal_end=1.0
```

### 3.7 Phase Stall Penalty
```python
PHASE_STALL_PENALTY = 0.1  # before — no deterrent
PHASE_STALL_PENALTY = 2.0  # after  — comparable to locomotion reward
```

### 3.8 Pair Feedback — Bound-Only
```python
# Before: locked both pairs (fighting L0's and L3's independent geometry)
for pair, pair_scale in ((plant_ids, 1.0), (swing_ids, swing_pair_scale)):
    ...apply to both pairs...

# After: only lock the physically bound pair
bound = tuple(sorted(self.BOUND_PAIR))
for pair, pair_scale in ((plant_ids, 1.0), (swing_ids, swing_pair_scale)):
    if tuple(sorted(pair)) != bound:
        continue   # L0 and L3 are independent — don't force them together
    ...apply...
```

### 3.9 Per-Leg Rod Targets for Bound Pair
```python
# Before: shared mean shift applied identically to all swing-pair legs
angle_shift = swing_angle_mean - plant_angle_mean
plant_angle = base_angle - angle_shift  # same for L1 and L3

# After: each leg derives shift from its own geometry
per_leg_angle_shift  = base_angle  - plant_angle_mean   # individual calibration
per_leg_length_shift = base_length - plant_length_mean
plant_angle  = base_angle  - per_leg_angle_shift        # = plant_angle_mean per leg
plant_length = base_length - per_leg_length_shift
```

### 3.10 Other
- `REFERENCE_HIP_AMPLITUDE`: 0.15 → 0.30 rad (stronger oscillator reference)
- `_shaping_terms`: removed 3× double-count of `swing_vertical_checkpoint_reward`
- All test files: Windows paths → `DEFAULT_MODEL_PATH_STR`
- Phase-dependent tests: added `env.phase = 0` after reset for determinism

---

## 4. Architecture Decisions

### Why tripod gait instead of the old diagonal crutch?
The user specified: *"L0 being one leg, L1 and 2 bound together, and L3 being the third."*
With L1+L2 bound, a diagonal pairing (old: 0+2 vs 1+3) doesn't match the physical constraint.
The correct pairing groups the bound unit together and gives L0/L3 each their own phase role.

### Why 6D action instead of 4D?
The old 4D forced the same command onto both swing legs and the same onto both plant legs.
With L0 and L3 having structurally different kinematic chains (different `UPPER_ROD_ANCHORS`,
different servo z-offsets in the MJCF), the policy could not correct per-leg asymmetry.
The 6D space gives:
- L1+L2 a shared command (they're bound)
- L0 and L3 independent commands (they're not)

### Why only lock the bound pair?
The pair feedback (`_apply_pair_state_feedback`) was designed to synchronize paired legs.
For L1+L2 (physically bound), forcing them to the same joint state is correct — they must move
together. For L0+L3 (independent legs in the same phase), forcing them to a shared mean state
was *incorrect* — it pushed each away from its own stable equilibrium, causing contact failure.

---

## 5. Test Results

```
test_gait_phase.py — 20 passed, 2 xfailed

XFAIL (expected): 
  test_only_internal_fastener_duplicates_are_hidden
  test_visible_leg_hardware_is_attached_to_moving_bodies
  → These test MJCF mesh names (stepper_mount_active, servo_horn, etc.) 
    that don't exist in the current MJCF. Pre-existing mismatch, not caused
    by this refactor. Visual correctness only; no training impact.
```

---

## 6. New Tools Added

### `visualize.py` — Headless Simulation Renderer
```bash
# Print per-step contact/reward diagnostics (no video):
python3 visualize.py --steps 200 --no-render --diag

# Render rollout to MP4 (works headlessly on WSL2/servers):
python3 visualize.py --steps 300 --out rollout.mp4

# Load a trained policy:
python3 visualize.py --policy tide_tars_policy.zip --steps 600 --out trained.mp4

# Force a specific start phase:
python3 visualize.py --phase 1 --steps 200 --diag

# Print phase contact summary (quick gait health check):
python3 visualize.py --phase-diag --steps 200
```

Sample `--phase-diag` output (pre-training, zero action):
```
Phase 0 desired contacts: [1.0, 0.0, 0.0, 1.0]   (L0+L3 planted)
Phase 1 desired contacts: [0.0, 1.0, 1.0, 0.0]   (L1+L2 planted)

Phase 0: actual=[0.0, 0.0, 0.0, 0.0] desired=[1.0, 0.0, 0.0, 1.0]  steps=1
Phase 1: actual=[0.99, 0.0, 1.0, 0.92] desired=[0.0, 1.0, 1.0, 0.0] steps=120
```
This immediately confirms the pre-training failure mode: L0, L2, L3 stay grounded in phase 1
when they should swing; L1 stays lifted when it should plant.

### `submit_train.py` — One-Command TIDE Training Submission
```bash
python3 submit_train.py --timesteps 1000000 --gpu 0
```
Syncs source files, installs dependencies remotely, trains on TIDE, downloads
`tide_tars_policy.zip` and `tide_tars_best_walk.zip` on completion. Push notifications
via Pushbullet at job start, completion, and failure.

---

## 7. Training on TIDE

**Current run:** 1M-step PPO on A100, submitted 2026-04-12.

Early metrics (8k timesteps, iteration 2):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| `ep_rew_mean` | −214 | Expected negative at start; penalties are active |
| `ep_len_mean` | 200 | Stagnation termination kicking in (< 0.01m / 200 steps) |
| `contact_pair_bonus` | 1.2 | Phase 0 foot pattern partially correct |
| `phase_contact_penalty` | −0.9 | 50% contact mismatch → penalty working |
| `reference_hip_reward` | 2.55 | Oscillator reference tracking producing signal |
| `curriculum/nominal_action_scale` | 0.35 | Correctly starting tight |
| `phase_timer` | 192 | Phase stalling; PHASE_STALL_PENALTY (−2/step) firing correctly |

The large negative reward from phase stalling (`−2.0/step × 130 steps = −260`) will drive the
policy to learn phase transitions first. This is correct behavior.

**To run visualization after training completes:**
```bash
python3 visualize.py --policy tide_tars_policy.zip --steps 600 --out trained.mp4 --diag
```

---

## 8. How the User Prompted

This section documents the prompting strategy used, intended as a teaching reference.

### 8.1 Layered Context Before Action

The user provided information in two layers before asking for changes:

**Layer 1 — Broad orientation:**
> *"New project for iterare to work on: One of my teammates is working on a simulation of our
> robot... For now, just familiarize yourself with the repo and save a summary."*

This gives the LLM time to build a mental model before receiving requirements. The LLM reads
files, traces architecture, and forms hypotheses. When the detailed requirements arrive, the
LLM can evaluate them against existing knowledge rather than interpreting them cold.

**Layer 2 — Structured critique with line references:**
> *"Critical — Training is fundamentally broken. 1. The entire reward is a single commented-out
> block — only progress_reward survives. tars_env.py:2031–2060..."*

The critique included file paths, line numbers, exact variable names, and the causal chain from
bug to symptom. This is the key difference from a vague "fix the training" prompt — it gives
the LLM enough context to verify each claim independently before acting on it.

### 8.2 Architecture Specification Paired With Bug List

Before the bug list, the user stated the intended design:
> *"Walking should be an effective tripod gait, with L0 being one leg, L1 and 2 'bound'
> together, and L3 being the third."*

This is the **why** before the **what**. The bug list then has context: the LLM understands the
goal well enough to evaluate whether each fix serves the stated architecture, not just whether
it matches the description mechanically.

### 8.3 Severity-Ranked, Tabular Critique

The user organized bugs into three severity tiers (Critical / Significant / Structural) and
ended with a summary table:

```
┌─────┬───────────────────────────────────────────────────────┬────────────────────────────┐
│  #  │ Issue                                                  │ Impact                     │
├─────┼───────────────────────────────────────────────────────┼────────────────────────────┤
│ 1   │ Reward is only progress_reward, all else commented out │ Training is wrong          │
│ 2   │ progress_gate unused                                   │ Robot lunges without penalty│
│ 3   │ 4D shared action space                                 │ Asymmetry uncorrectable    │
...
```

A ranked table tells the LLM which fixes to prioritize if forced to choose, and makes the
overall scope of work clear upfront. It also models the kind of structured thinking the LLM
should apply when generating its own analysis.

### 8.4 Behavioral Corrections Stated Once, Remembered

When the LLM made a mistake, the user corrected it tersely and added "remember this":
> *"Do not write short scripts like this. Write them to a file and run them. Remember this."*

The LLM saves behavioral corrections to a persistent memory file. Future sessions load that
memory automatically. The user never had to repeat the correction. This is the most efficient
use of the memory system: teach once, persist forever.

Similarly:
> *"and don't ask again for echo \*"*

One sentence. No elaboration needed. The memory system handles the rest.

### 8.5 Continuation Without Repetition

After context windows filled and sessions resumed, the user said:
> *"Please continue, limits are reset"*  
> *"Continue from where you left off."*

No re-explanation of goals, no re-statement of constraints. The persistent memory and
conversation summary system preserves context across sessions. The user trusts the system to
resume without hand-holding.

### 8.6 Outcome-Oriented, Not Step-Oriented

The user specified *outcomes*, not implementation steps:
> *"Find any and all bugs with it, fix them, and run tests to validate your solutions."*
> *"Build some way for you to visualize the simulation so you can analyze it."*
> *"Make sure to submit jobs to TIDE, not this computer."*

The LLM chose how to implement each outcome. This is more efficient than specifying steps,
because the LLM has domain knowledge about what is feasible and can adapt to what it finds
in the code. Step-oriented prompts often become wrong or incomplete once the LLM reads the
actual code.

### 8.7 Summary: Key Prompting Principles Demonstrated

| Principle | Example |
|-----------|---------|
| Orient before requiring | "Familiarize yourself, then I'll give you the bug list" |
| Architecture before implementation | "Tripod gait: L0, L1+L2, L3" → then bugs |
| Line references in critiques | `tars_env.py:2031–2060`, `lines 10–24 of training_helpers.py` |
| Severity ranking | Critical / Significant / Structural tiers + summary table |
| Corrections stated once | "Don't ask again for echo \*" — one sentence, no repetition |
| Outcome goals, not step lists | "Fix bugs and run tests" not "change line X then run pytest" |
| Persistent memory for continuity | Cross-session context via memory files |

---

## 9. File Map

```
TARS_Walking/
├── tars_env.py              ← Main env (2100 lines). All fixes here.
│                               - TARSEnv(gymnasium.Env)
│                               - 6D action: L0/L12/L3 angle+length
│                               - 41D obs: no absolute XY
│                               - Tripod gait pairing
│                               - Full reward function
├── training_helpers.py      ← CurriculumCallback (fixed direction), BestWalkCallback
├── train.py                 ← PPO training entry point (local + TIDE)
├── tars_model.py            ← Model path resolution
├── tars_mjcf.xml            ← MuJoCo model (physical robot structure)
│                               - servo_l1 has z=+0.0375 (sign flip vs l0)
│                               - This asymmetry is compensated via HIP_STANDING_SIGNS
├── tars_gait_reference.py   ← Phase poses, oscillator, DISPLAY_TO_MODEL mapping
├── visualize.py             ← NEW: headless MP4 renderer + contact diagnostics
├── submit_train.py          ← NEW: one-command TIDE training submission
├── test_gait_phase.py       ← Unit tests (20 pass, 2 xfail)
├── test_env.py              ← Integration test
├── test_survive.py          ← Survival run test
├── test_sanity.py           ← Sanity checks
├── test_stability.py        ← Stability sweep
├── test_feet.py             ← Foot geometry tests
├── tide_tars.py             ← Legacy TIDE runner (smoke-train, sync)
└── tide/                    ← TIDE client library
    ├── client.py
    ├── execute.py
    └── jobs.py
```

**Key constants in `tars_env.py`:**

```python
SUPPORT_PAIR   = (0, 3)   # L0 + L3 — independent planted legs
SWING_PAIR     = (1, 2)   # L1 + L2 — bound swinging pair
BOUND_PAIR     = (1, 2)   # physically constrained — always synced
ACTION_DIM     = 6
PHASE_STEPS    = 30       # 0.3s per phase at 10ms/action
FRAME_SKIP     = 5        # 5 mujoco steps × 0.002s = 10ms per policy action
MAX_EPISODE_STEPS = 2000
PHASE_STALL_PENALTY       = 2.0
REFERENCE_HIP_AMPLITUDE   = 0.30  # rad
PLANT_UNLOAD_PENALTY_SCALE = 1.0
```

---

## 10. Open Questions / Next Steps

1. **Does phase 1 work after training?**  
   Run `python3 visualize.py --policy tide_tars_policy.zip --phase-diag` to check.  
   Pre-training baseline: L0/L3 stay grounded 92-99% of time in phase 1 when they should swing.

2. **RESET_PLANT_POSE tuning?**  
   `RESET_PLANT_POSE = (-0.15, 0.0, -0.10)` — the default joint offsets that place planted legs.
   If L1 still struggles to contact the ground in phase 1, these may need per-leg overrides
   since L1/L2 have shorter effective rod lengths than L0/L3.

3. **Longer training?**  
   1M steps is a baseline. PPO on quadruped locomotion typically needs 5–20M for stable gaits.
   Re-run `python3 submit_train.py --timesteps 5000000` after evaluating the 1M checkpoint.

4. **MJCF mesh visibility bug**  
   Two tests are `xfail` because all internals body geoms have `alpha=0` in the compiled model.
   Structural meshes (`side_plate`, `faceplate_corner`, etc.) should be visible. The
   `_reattach_leg_visual_meshes` method may be over-hiding them. Affects visualization only.

5. **Stagnation threshold tuning**  
   Episodes are currently terminating at step 200 (stagnation: < 0.01m / 200 steps). This is
   correct at the start but may be too aggressive once the policy begins walking. Consider
   reducing stagnation sensitivity as training progresses.
