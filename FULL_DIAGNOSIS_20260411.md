# Full Diagnosis

Date: 2026-04-11

Scope:
- Current code in `tars_env.py`, `training_helpers.py`, `train.py`, and `tide_tars.py`
- Completed 500k-step TIDE run in `run_logs/tide_train_500k_20260411_rewardfix.log`
- Remote text-only post-training evaluation of `tars_policy.zip`

This report separates:
- fixed issues that were present earlier and are now addressed
- current blockers that still prevent the robot from learning a usable gait
- infrastructure issues discovered during training/visualization attempts

## Executive Summary

Training is no longer failing because the reward was fully degenerate. That part was materially improved.

The robot still does not learn to walk.

The current failure mode is:
- PPO learns to collect shaping/reference/contact reward
- forward progress remains near zero or negative at evaluation
- phase progression stalls in one phase until stagnation termination
- the learned contact pattern remains structurally wrong, especially in phase 1

The strongest current blocker is not raw optimization instability. It is that the environment still allows a high-reward non-walking local optimum:
- the policy can earn positive reward from shaping terms
- progress is mostly gated out
- phase timeout does not force a transition or a hard episode failure
- episodes often end after 200 stagnating steps with little or negative displacement

## Run Outcome Summary

Source: `run_logs/tide_train_500k_20260411_rewardfix.log`

Observed at the end of the 500k run:
- `ep_rew_mean` reached about `+513` near step `499712`
- final evaluation still had `mean_x = -0.0738`
- final evaluation survival was only `0.20`
- final eval contact rates were:
  - overall: `[0.91, 0.28, 0.932, 0.543]`
  - phase 0: `[1.0, 0.23, 1.0, 0.34]`
  - phase 1: `[0.73, 0.38, 0.795, 0.95]`

Key evidence:
- `run_logs/tide_train_500k_20260411_rewardfix.log:11527`
- `run_logs/tide_train_500k_20260411_rewardfix.log:11533`

Interpretation:
- the robot is not producing sustained forward locomotion
- leg 2 is still effectively over-planted
- leg 1 is still under-supporting
- phase 1 still does not resemble the intended `[0,1,0,1]` support pattern

## Remote Post-Training Behavior

Text-only post-training rollout evaluation showed:
- many episodes remain in a single phase for the full 200-step stagnation window
- the robot may briefly move forward, then drift backward
- reward can remain positive while final `x` is negative

Representative behavior:
- phase-1 starts: `x` moves from about `-0.0046` to `+0.0221`, then falls back to `-0.0032` by step 200
- phase-0 starts: reward stays strongly positive while final `x` reaches about `-0.0414`

This means the policy is not learning “walk forward.”
It is learning “hold a rewarded pose/contact regime while not advancing.”

## Issues Fixed Since The Earlier Broken Version

These earlier defects were real, and are now materially addressed in the current code.

### 1. Reward is no longer just raw progress

Current reward composition includes:
- gated progress
- gated velocity bonus
- shaping reward
- gait reference rewards
- swing lift / swing plant penalties
- phase contact penalty
- early touchdown penalty
- multiple body-stability penalties
- plant unload penalty
- fall penalty

Source:
- `tars_env.py:2069`

### 2. `progress_gate` is now active

`progress_reward` is multiplied by `progress_gate`.

Source:
- `tars_env.py:2070`

### 3. `PLANT_UNLOAD_PENALTY_SCALE` is now used

`plant_unload_penalty` is computed from `plant_loss_fraction` and included in reward.

Sources:
- `tars_env.py:2019`
- `tars_env.py:2093`

### 4. Action space is no longer 4D pair-shared

The action space is now 8D, with per-leg angle/length residuals.

Sources:
- `tars_env.py:94`
- `tars_env.py:997`

This removed the earlier hard incapacity to command `l1` and `l3` differently.

### 5. Observation no longer exposes absolute world XY

Observation uses `qpos[2:]` rather than all `qpos`, removing absolute world `x/y`.

Source:
- `tars_env.py:1742`

### 6. Reset no longer always starts from phase 0

Reset randomizes start phase.

Source:
- `tars_env.py:1619`

### 7. Curriculum no longer shrinks policy authority

`nominal_action_scale` now grows from `0.35` to `1.0`, while pair locking decays toward zero.

Source:
- `training_helpers.py:10`

### 8. The duplicate shaping term was removed

The old duplicated `swing_vertical_checkpoint_reward` in the non-foundation shaping profile is gone.

Source:
- `tars_env.py:1325`

### 9. Per-leg rod targets are no longer initialized from a single pair-mean offset hack

Role targets are now derived from actual phase poses for each leg independently.

Source:
- `tars_env.py:1364`

### 10. Pair-lock is no longer fixed at 1.0

The default hard-coded blend was reduced and the curriculum now decays it.

Sources:
- `tars_env.py:179`
- `training_helpers.py:15`

## Current Critical Blockers

These are the main reasons gait is still not learning.

### 1. Phase timeout does not actually resolve stalled phases

Current logic:
- phase increments timer every step
- `_phase_switch_ready(...)` must become true for a switch
- if timeout is reached and the switch is not ready, only a penalty is applied
- the phase does not switch
- the episode does not end immediately

Source:
- `tars_env.py:2103`

Specifically:
- `phase_stalled = self.phase_timer >= self.PHASE_SWITCH_TIMEOUT_STEPS and not ready_to_switch`
- if `phase_stalled`: `reward -= self.PHASE_STALL_PENALTY`
- there is no forced switch or timeout termination

Why this is a blocker:
- the agent can remain in one phase indefinitely
- the reward system then optimizes static contact/shaping configurations within a single phase
- the rollout traces show exactly this: `phase` stays fixed while `phase_timer` grows to 200

Observed consequence:
- episodes frequently end from stagnation, not from successful gait cycles

### 2. Positive reward is still available when locomotion is absent

The current reward keeps these positive terms outside the progress gate:
- `shaping_reward`
- `gait_reference_hip_reward`
- `gait_reference_leg_reward`
- `swing_lift_reward`

Source:
- `tars_env.py:2069`

Why this matters:
- if `progress_gate` collapses to near zero, forward reward disappears
- but positive shaping/reference reward can still remain
- the policy therefore has an incentive to hold a quasi-valid non-walking configuration

Direct evidence from the 500k log:
- `progress_gate = 1.83e-29`
- `ep_rew_mean = 513`
- `x = -0.00179`

Source:
- `run_logs/tide_train_500k_20260411_rewardfix.log:11475`
- `run_logs/tide_train_500k_20260411_rewardfix.log:11505`
- `run_logs/tide_train_500k_20260411_rewardfix.log:11508`

Diagnosis:
- reward loophole remains
- the loophole is now shaping-based instead of lunge-based

### 3. Phase-contact target is binary and phase-global, but the policy still does not have enough incentive to complete a real phase transition

Desired contacts are still just “current plant pair down, swing pair up.”

Source:
- `tars_env.py:934`

What is missing:
- a stronger notion of phase completion
- a reward or success signal that explicitly values transition completion
- a hard consequence for remaining in the same phase too long

Current result:
- the agent optimizes good-looking within-phase configurations instead of alternating support cycles

### 4. Stagnation termination masks the real problem instead of solving it

Episodes terminate if progress over 200 steps is below `0.01`.

Source:
- `tars_env.py:2114`

Why this is not enough:

## 2026-04-11: Full-File 4-Phase Audit

This section supersedes the older two-phase assumptions. The current code has been partially migrated to the canonical 4-phase gait, but several deep contradictions remain in `tars_env.py`.

Authoritative gait:
- `phase 0 = [1,0,0,1]`
- `phase 1 = [1,1,1,1]`
- `phase 2 = [0,1,1,0]`
- `phase 3` returns to phase 0
- `1 = on the ground`, `0 = off the ground`

### Findings

#### Critical 1. The controller still only understands two leg roles, not four gait phases

The env derives everything from a binary `plant` vs `swing` split:
- [_phase_pairs](./tars_env.py) lines 736-744
- [_phase_reset_ctrl](./tars_env.py) lines 765-783
- [_desired_leg_role_targets](./tars_env.py) lines 1141-1210

That means the controller can only express:
- legs in a planted-role pose
- legs in a swing-role pose

But your gait needs three distinct contact states:
- outer support
- all-feet grounded
- middle support

Right now phase 1 (`[1,1,1,1]`) is not represented as its own learned pose family. It is just “everyone is in the plant role,” which is fundamentally different from a true phase-specific full-ground pose.

Impact:
- phases 0 and 2 are competing for the same binary plant/swing parameterization
- phase 1 cannot encode its own distinct all-ground geometry except through ad hoc `PHASE_CTRL_BIAS`
- PPO is not learning a true 4-phase controller; it is learning a 2-role controller wrapped in 4 contact masks

#### Critical 2. Phase-specific geometry is overwritten and collapsed into role averages

During target initialization, each leg stores snapshots only by role name:
- [_initialize_rod_targets](./tars_env.py) lines 1648-1709

The code loops over all phases, but writes:
- `role_snapshots[leg_id]["plant"] = ...`
- `role_snapshots[leg_id]["swing"] = ...`

So later phases overwrite earlier ones when they share the same role.

Concrete consequence:
- phase 0 and phase 1 both write `plant` snapshots for legs 0 and 3
- phase 1 and phase 2 both write `plant` snapshots for legs 1 and 2
- the env then averages only one `plant` and one `swing` target per leg into `leg_neutral_targets`

Impact:
- distinct support poses are being erased during initialization
- phase 0 and phase 2 cannot keep their own target geometry
- phase 1 all-ground geometry pollutes the other phases, and vice versa

This is currently one of the largest blockers in the whole file.

#### Critical 3. `phase 3` is implemented as a full phase, but its contact mask is identical to phase 0

The contact masks are:
- phase 0: `[1,0,0,1]`
- phase 1: `[1,1,1,1]`
- phase 2: `[0,1,1,0]`
- phase 3: `[1,0,0,1]`
in [tars_env.py](./tars_env.py) lines 120-125.

But your specification says phase 3 returns to phase 0. The current code treats phase 3 as another full timed phase with its own timeout and switch logic.

Impact:
- the robot spends an extra full `PHASE_STEPS` interval in a duplicate support state
- every cycle gets an unnecessary extra switch opportunity and timeout risk
- reset sampling overweights the outer-support family because both phase 0 and phase 3 are selected uniformly at reset
  - [reset](./tars_env.py) lines 1889-1890

If phase 3 is meant to be “return to phase 0,” it should not be a separate full-duration contact phase in the current form.

#### Critical 4. Phase switching criteria do not match the 4-phase semantics

Switch readiness uses generic thresholds:
- `PHASE_SWITCH_MIN_NEXT_PLANT_CONTACT_FRACTION = 0.5`
- `PHASE_SWITCH_MIN_NEXT_PLANT_LOW_FRACTION = 1.0`
in [tars_env.py](./tars_env.py) lines 186-188

And applies them to the next phase’s `plant_ids` in [_phase_switch_ready](./tars_env.py) lines 1484-1547.

This is mismatched for the new gait:
- from phase 0 to phase 1, `next_plant_ids` are all four legs
- but only `50%` next-plant contact is required to consider support ready

So the env can approve a transition toward `[1,1,1,1]` while only two of four feet are actually grounded.

Impact:
- switch readiness is too weak for the all-ground phase
- the threshold logic was inherited from a two-pair gait and not re-authored for the 4-phase cycle

#### Critical 5. `_oscillator_opposition_error()` is invalid for phase 1 and can produce `nan`

This function computes means over `plant_ids` and `swing_ids`:
- [_oscillator_opposition_error](./tars_env.py) lines 1563-1571

In phase 1:
- `plant_ids = (0,1,2,3)`
- `swing_ids = ()`

So `np.mean([... for leg_id in swing_ids])` is a mean over an empty list. That matches the runtime warnings already seen on TIDE:
- `RuntimeWarning: Mean of empty slice`

Impact:
- `oscillator_opposition_error` can become `nan`
- `oscillator_opposition_reward` can become `nan`
- `progress_gate` multiplies by that reward ratio
  - [step](./tars_env.py) lines 2204-2219

This can silently corrupt reward and diagnostics during the all-ground phase.

#### Critical 6. Reset grounding likely bakes in biased support when more than one foot should be down

The reset grounding logic uses the maximum planted-foot height:
- [_ground_support_feet](./tars_env.py) lines 1730-1735

It computes:
- `support_z = max(foot_z for planted feet)`
- then shifts the whole body so that this highest planted foot sits at the target floor height

For multi-foot support phases, especially `[1,1,1,1]`, this can leave the lower feet pushed into the floor after grounding.

Impact:
- can create the “spawn low / bounce / settle” behavior you described
- can preload some legs deeper into ground contact than others
- can poison the phase snapshots captured later for support offsets and role targets

This is a strong candidate for the early load-in distortion you observed visually.

#### Significant 7. Phase reset/reference generation is still role-based, not phase-based

Both reset and reference controllers are built from the same binary plant/swing logic:
- [_phase_reset_ctrl](./tars_env.py) lines 765-783
- [_reference_ctrl_targets](./tars_env.py) lines 791-812

The reference controller still uses a single sinusoid split into `pair_a_signal` and `pair_b_signal`, which is an inherited alternating-gait concept rather than a bespoke 4-phase reference.

Impact:
- reference tracking rewards are still pulling toward a two-role oscillator, not your actual phase sequence
- phase 1 all-ground and phase 0/2 support poses are not distinctly encoded except by the static bias table

#### Significant 8. `PHASE_CTRL_BIAS` is mostly zero, so phase identity currently depends on the broken role abstraction

`PHASE_CTRL_BIAS` is only populated for phase 1:
- [tars_env.py](./tars_env.py) lines 144-153

Phase 0, 2, and 3 are all zeros.

Impact:
- the code has no meaningful phase-specific correction for most phases
- it therefore falls back to the shared plant/swing abstraction described above
- this is consistent with the TIDE searches showing phase 0 and 2 need explicit direct-pose calibration

#### Significant 9. Reset sampling currently oversamples the outer-support family

Reset samples uniformly from `range(self.PHASE_COUNT)`:
- [tars_env.py](./tars_env.py) lines 1889-1890

Because phase 3 duplicates phase 0’s mask, the current reset distribution is effectively:
- outer-support family: 50%
- all-ground: 25%
- middle-support: 25%

Impact:
- training sees twice as many outer-support starts as middle-support starts
- this introduces a structural curriculum bias that is not part of the desired gait

#### Significant 10. The reward still measures many “pair” metrics that are only loosely meaningful when role cardinality changes

A number of sync/target rewards still use shared helpers with names like:
- `_pair_theta_difference`
- `_pair_leg_difference`
- `_pair_rod_difference`
- `_pair_foot_forward_difference`

Those helpers were made numerically safe for variable-length groups, but they still conceptually assume “within-role similarity” is desirable:
- [tars_env.py](./tars_env.py) lines 846-887
- [tars_env.py](./tars_env.py) lines 1371-1447
- [tars_env.py](./tars_env.py) lines 1553-1561

Impact:
- in phase 1, all four grounded legs are still being softly encouraged to be mutually similar
- that may fight the actual geometry needed for a stable all-ground stance if front and rear legs should not match exactly

This is not the first blocker, but it will likely matter after the phase-specific controller is fixed.

#### Significant 11. The file still contains legacy timing/comments from the old gait

Example:
- `PHASE_STEPS = 30  # ... 0.6s full cycle`
in [tars_env.py](./tars_env.py) line 117

With 4 phases, that comment is now false; a full cycle is `1.2s` at the stated timing.

This is minor on its own, but it is a signal that timing assumptions were not fully re-authored for the new gait.

## Revised Current Root Cause

The main blocker is now:

1. The environment still encodes gait control as a binary plant/swing problem.
2. Your gait is not binary; it has distinct outer-support, all-ground, and middle-support states.
3. The initialization path then collapses multiple phases into shared role targets, erasing phase-specific geometry.
4. Reset grounding likely adds biased multi-foot contact preload on top of that.
5. The switch logic and reference rewards are still inherited from the older alternating-gait model.

So the current problem is not “just tune the reward more” and not “just train longer.”

It is that the environment logic is still fundamentally shaped around the wrong gait abstraction.

## Recommended Fix Order

1. Replace binary `plant` / `swing` target generation with true phase-specific targets.
   - Each leg should have a target per phase, not per role.

2. Remove the duplicate full-duration phase 3 or make it an instantaneous return to phase 0.

3. Rewrite `_initialize_rod_targets()` to preserve all phase snapshots instead of overwriting by role.

4. Rewrite `_reference_ctrl_targets()` so it encodes the actual 4-phase gait, not a two-group oscillator.

5. Fix `_oscillator_opposition_error()` or remove it from gating/reward for phases with no swing set.

6. Redo reset grounding for multi-foot phases so it aligns support feet without over-embedding the lower ones.

7. Re-author phase-switch thresholds per transition, especially `0 -> 1`, where all four legs should be down.
- it ends bad behavior eventually
- but it still gives the policy 200 steps to mine shaping reward in a stalled regime
- it does not teach the agent how to leave the stall

This interacts badly with the non-gated shaping terms.

### 5. Final evaluation proves gait is still structurally wrong in phase 1

Phase-1 contact rates:
- foot 0: `0.73`
- foot 1: `0.38`
- foot 2: `0.795`
- foot 3: `0.95`

Source:
- `run_logs/tide_train_500k_20260411_rewardfix.log:11545`

This means:
- foot 3 is nearly always down in phase 1
- foot 2 is still mostly down though it should be up
- foot 1 is not reliably carrying/supporting
- foot 0 is still too often down

That is still fundamentally inconsistent with a clean alternating gait.

## Significant Structural Risks Still Present

These may not each be the single dominant blocker, but they still weaken learnability.

### 6. Plant-foot targets are latched and can preserve bad support geometry

Plant feet are snapped to a grounded track target once and then reused while the leg remains in plant role.

Source:
- `tars_env.py:1010`

Risk:
- if a leg enters plant role from a mechanically poor state, the target may preserve an already-bad support footprint
- this can make “support recovery” harder than it should be

### 7. Plant target x-direction is still forcibly signed negative

For plant legs:
- `vector_body[0] = -abs(vector_body[0])`

Source:
- `tars_env.py:1003`

Risk:
- this is still a hand-authored geometric bias
- it may overconstrain reachable plant configurations and hide remaining asymmetries instead of letting the controller discover them

This may still be pushing some legs toward a mechanically awkward planted target.

### 8. Pair-state feedback still exists in the control path

Even though pair locking now decays, all control targets still pass through `_apply_pair_state_feedback(...)`.

Source:
- `tars_env.py:1089`

Risk:
- residual pair regularization may still resist independently stable solutions across asymmetric legs
- the problem is smaller than before, but not necessarily gone

### 9. Evaluation metric selection can hide the reward loophole during training

`BestWalkCallback` uses `mean_x` as the saved-model score, which is good.
But training dashboards and rollout reward still emphasize episode reward heavily.

Source:
- `training_helpers.py:144`

What happened in practice:
- training reward looked improved
- actual evaluation forward motion stayed negative

This is not a bug in `BestWalkCallback`, but it shows the environment reward still does not align with true objective well enough.

## Infrastructure And Visualization Issues

### 10. TIDE rollout previews are not reliable on the current cluster image

The TIDE node is headless and lacks a working offscreen MuJoCo GL stack.

Observed failures:
- no X11 display
- no valid EGL initialization

Affected code:
- `training_helpers.py:251`
- `tide_tars.py:100`

What was done:
- preview callback was made fail-open instead of killing training
- `MUJOCO_GL=egl` was requested on remote runs

Current status:
- training continues safely
- remote visual rollout capture still does not work on the cluster image

### 11. Remote import of `mujoco` itself can fail if EGL is requested in an environment with incomplete EGL bindings

This was observed during remote post-run visual attempts.

Diagnosis:
- TIDE image has enough MuJoCo to train physics
- but not a clean enough OpenGL/EGL stack for renderer-based visualization

Practical implication:
- visual rollout inspection should be done locally from downloaded checkpoints

## What The 500k Policy Actually Learned

Based on the log and remote text rollouts:
- it learned a more orderly body posture than the broken progress-only policy
- it learned some swing-lift behavior in phase 0
- it did not learn stable alternating stepping
- it often stays in a single phase until stagnation end
- it can collect positive reward while ending with negative forward displacement

In short:
- it learned a rewarded stance-and-shift behavior
- it did not learn a gait

## Recommended Fix Order

### Priority 1. Make stalled phases impossible to exploit

Change phase-timeout behavior so timeout causes one of:
- immediate forced phase switch
- immediate episode termination
- or both, depending on debug mode

Recommended default:
- terminate the episode on phase timeout
- optionally log a `phase_timeout_terminated` diagnostic

Reason:
- this directly removes the “single-phase reward farming” loophole

### Priority 2. Gate more positive shaping under locomotion validity

At minimum, gate these terms by a stricter locomotion-validity factor:
- `shaping_reward`
- `gait_reference_hip_reward`
- `gait_reference_leg_reward`
- possibly `swing_lift_reward`

Current ungated positives are the main reason reward can remain high while progress stays negative.

### Priority 3. Make phase completion a directly rewarded event

Add an explicit positive reward for:
- successful transition into the next phase
- or sustained correct contact pattern after transition

Right now the code mostly rewards “being in a plausible phase pose,” not “completing a gait cycle.”

### Priority 4. Revisit plant target latching and plant x-sign forcing

Re-examine:
- `tars_env.py:1003`
- `tars_env.py:1010`

These are still strong hand-authored geometric constraints.

### Priority 5. Keep visualization local

Use the downloaded checkpoint:
- `tide_tars_policy_smoke.zip`

Visual inspection should be done locally with:
- `python view_mujoco_tars.py --mode policy --policy tide_tars_policy_smoke.zip`

## Bottom Line

The current code is much less broken than the earlier version.
The reward is no longer fundamentally degenerate in the old “lunge forward and fall” sense.

But the environment still contains a major loophole:
- stalled phases are not resolved
- positive shaping remains available even when locomotion fails

That loophole is now the main blocker.

The 500k run confirms this conclusively:
- training reward improved
- final walking performance did not
- final evaluation forward progress remained negative
- phase-1 support pattern remained incorrect

That is the current root diagnosis.

## 2026-04-11: Latest Fix Status

The items below were implemented after the audit above and should be treated as resolved in code pending runtime validation:

- the duplicate full-duration return phase was removed from the runtime cycle
- the env now runs `0 -> 1 -> 2 -> 0`
- target initialization is now phase-specific instead of being overwritten into shared role buckets
- transition roles are now explicit and drive target generation
- reference control now interpolates between consecutive phase poses instead of using the older alternating oscillator split
- `_oscillator_opposition_error()` now returns safely when there is no valid moving/support comparison set
- reset grounding now aligns the lowest planted foot instead of embedding lower support feet under the highest one
- phase switch readiness now evaluates the next phase’s desired grounded and airborne sets directly

What still remains unresolved until TIDE reruns:
- whether the refactored controller actually produces clean unload of `l1` in phase 0
- whether it produces clean unload of `l3` in phase 2
- whether the remaining pair-similarity rewards are too restrictive once the new transition model is exercised at runtime
