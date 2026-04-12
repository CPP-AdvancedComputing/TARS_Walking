# Progress

Primary running record:
- `OverallProgress.md`
- Treat that file as the canonical current-status document. Keep updating it on every meaningful change.

## Infrastructure And Remote Training

We wired this workspace to TIDE/JupyterHub and verified the remote server before doing any serious training work. The key local runner is `tide_tars.py`, supported by the local `tide/` package and vendored Python dependencies so the client can run from this repo without needing system-wide changes. We verified the remote server profile and used CPU for PPO because this project is a small-policy Stable-Baselines3 workload, not a GPU-heavy deep learning job.

We ran remote smoke and longer jobs through TIDE and captured logs locally. The important long-run logs are:
- `run_logs/tide_train_100k_20260411.log`
- `run_logs/tide_train_500k_20260411.log`

We also added `SYNC_MARKER.md` locally and uploaded a matching remote marker so sync state can be confirmed from JupyterHub directly.

## Core Environment Work

Most of the engineering effort went into `tars_env.py`, plus a few related training and diagnostic files.

Changes made and kept:
- `train.py` now defaults PPO to CPU for this workload.
- default episode length was increased to `2000` steps.
- default reward profile was switched to a simpler `foundation` profile.
- per-leg target generation was restored instead of flattening all legs toward pair-wide mean targets.
- swing timing was made symmetric across legs.
- stale curriculum handling and debug helper usage were corrected.
- reward-logic branching was simplified by moving the reward-profile split into a helper.
- dead reward constants and unused penalty calculations were removed to reduce bloat.

Files most affected:
- `tars_env.py`
- `train.py`
- `training_helpers.py`
- `debug_gait_inspector.py`
- `test_gait_phase.py`

## Diagnostics Added

We added persistent diagnostics so the next debugging passes do not depend on chat history:
- `diagnose_phase_support.py`
- `audit_leg1_geometry.py`
- `PROJECT_AND_EXPERIMENT_NOTES.md`

These were added to make phase-support checks and leg-1 geometry audits repeatable and easier to compare across iterations.

## Main Findings

The motion problem is real simulation/control behavior, not just a viewer illusion.

What we established:
- phase 0 is the healthier half of the gait.
- phase 1 is still the weak half.
- the remaining blocker is centered around support integrity in phase 1, especially leg `l1`.
- a repeated pattern in remote diagnostics is:
  - desired phase-1 support pair: `[0,1,0,1]`
  - actual support pattern tends to become `[1,0,0,1]`
- that means `l1` unloads while `l0` re-contacts the floor.

We also audited the raw URDF/MJCF joint definitions and did not find an obvious trivial axis-sign mismatch between `l1` and `l3`. That means the likely remaining cause is deeper than a simple axis typo and is probably tied to geometry, support under load, or mirrored target construction for leg 1.

## Things We Tried And Reverted

Several stronger interventions were tested and then reverted after remote validation showed they made behavior worse:
- forced timeout phase switching
- aggressive body-centering over the support centroid
- much stronger actuator gains

Those changes either stalled the gait, pushed the robot into all-feet contact behavior, or degraded training reward. The current workspace keeps the best-performing baseline from this session, not every experiment.

## Training Results So Far

### 100k Run

The 100k TIDE run completed successfully and downloaded:
- `tide_tars_policy_100k.zip`

The result did not solve the gait. Final evaluation still showed:
- negative forward progress overall
- phase-1 support collapse
- phase-0-dominant behavior

### 500k Run

The 500k TIDE run was started and is being logged locally:
- `run_logs/tide_train_500k_20260411.log`

At the latest checked point during the session:
- the run was still active
- performance still showed the same structural issue
- phase 1 remained effectively unlearned compared to phase 0

So the working conclusion is that longer training alone is not the fix. The remaining structural env/geometry issue still needs to be solved.

## Repo And Publishing Work

We also prepared a cleaned version of the project for the GitHub repo:
- `https://github.com/CPP-AdvancedComputing/TARS_Walking`

Work completed:
- cloned the empty repo
- copied over a curated project snapshot
- reduced a large amount of bloat
- created a cleaned commit locally in `/tmp/TARS_Walking_repo`

Push did not complete because GitHub auth was not configured in that shell. The cleaned repo state is prepared locally and can be pushed once credentials are available.

## Current Best Interpretation

The main blocker is no longer “the robot just needs more time to train.” The consistent evidence points to a phase-1 support problem that PPO cannot simply learn around. Phase 0 is trainable enough to produce stable-looking metrics, but phase 1 still breaks support integrity and keeps the robot from learning a true alternating gait.

## Recommended Next Steps

1. Continue targeted phase-1 debugging, especially leg `l1`.
2. Compare `l1` and `l3` under identical planted-target conditions.
3. Audit whether `l1` needs mirrored target construction relative to `l3`.
4. Keep using the simpler `foundation` reward profile until the structural phase-1 issue is improved.
5. Only trust longer training runs after phase-1 static support is materially better.

## 2026-04-11: Switch Deadlock Diagnosis

The 100k `switchdebug` TIDE run showed that phase transitions were still not happening at all even after timeout-farm fixes.

Key evidence from `run_logs/tide_train_100k_20260411_switchdebug.log`:
- `ep_len_mean` stayed pinned at `60`
- `phase_transition_reward` stayed at `0`
- `phase_switch_block_code` was often `3`
- when block code `3` appeared, `next_plant_contact_fraction = 1` and `next_plant_low_fraction = 1`, but `switch_swing_clear_fraction = 0`

Interpretation:
- the next support pair was often already grounded and low enough to accept weight
- the environment was still refusing to transition because one or both outgoing swing feet were dragging
- that makes swing clearance a hard deadlock condition instead of a soft quality preference

Kept fix after this diagnosis:
- phase switching now uses short history for next-plant contact, next-plant low height, and swing clearance
- transition can be force-armed after stable next-support readiness persists beyond `PHASE_STEPS + 8`
- new telemetry logs history-averaged readiness plus `phase_switch_force_armed`

This change is intended to answer the remaining question directly:
- are transitions blocked because support is not established,
- or because the switch contract is too strict to let an imperfect but viable step complete?

## 2026-04-11: TIDE Root-Cause Loop

Additional TIDE diagnostics and reruns narrowed the remaining blocker substantially.

What was confirmed:
- `zero_action()` was not reproducing `phase_reference_ctrl` before calibration.
- fixing that mismatch was real and measurable:
  - early 100k training transient improved
  - `ep_len_mean` briefly rose above the old `60`-step timeout floor
- but the run still converged back to the old failure mode:
  - `phase_transition_reward` stayed pinned at `0`
  - `ep_len_mean` returned to `60`
  - final eval still had `foot_1_contact_rate = 0`

Most important new diagnosis:
- `l1` is effectively non-load-bearing under dynamic hold.
- This was reproduced on TIDE across:
  - the original two-phase alternating gait
  - alternative support-pair templates
  - single-swing / three-feet-down support patterns
  - global phase reset pose sweeps
  - `l1`-specific phase-1 joint bias sweeps
  - `l1` foot/track z-offset sweeps
  - `l1` actuator-strength sweeps

Representative TIDE artifacts:
- training rerun:
  - `run_logs/tide_train_100k_20260411_switchdebug.log`
- current support template static hold:
  - `run_logs/tide_diag_supportpair_02_20260411.log`
- alternative support templates:
  - `run_logs/tide_diag_supportpair_03_20260411.log`
  - `run_logs/tide_diag_supportpair_12_20260411.log`
- zero-action / phase-reference consistency:
  - `run_logs/tide_diag_phase_control_consistency_20260411.log`
- three-feet-down support viability:
  - `run_logs/tide_diag_single_swing_support_20260411.log`
- l1-specific sweeps:
  - `run_logs/tide_sweep_phase1_l1_joint_bias_20260411.log`
  - `run_logs/tide_sweep_l1_contact_geometry_z_20260411.log`
  - `run_logs/tide_sweep_l1_actuator_strength_20260411.log`

Current conclusion:
- reward and switch logic were necessary to fix
- the current dominant blocker is no longer reward shaping
- the remaining failure is model/support integrity centered on `l1`
- until `l1` can carry load, PPO keeps rediscovering the same degenerate contact pattern

## 2026-04-11: Reset/Support Integrity Deep Dive

Additional diagnostics after the `l1` support conclusion tightened the failure mode further.

New code changes:
- reset now starts directly from the selected phase reference pose and delays plant-target contact capture
- phase support offsets and rod-role targets are now derived from grounded, settled phase snapshots instead of raw ungrouded phase poses
- phase plant pairs are now configurable per phase with:
  - `TARS_PHASE0_PAIR`
  - `TARS_PHASE1_PAIR`
- controller initialization now tolerates overlapping plant pairs across phases

New diagnostics:
- reset / phase-start transient trace:
  - `run_logs/tide_diag_reset_transient_phasepose_20260411.log`
- 20k reset-quarantine training check:
  - `run_logs/tide_train_20k_20260411_resetquarantine.log`
- phase-1 hold action search:
  - `run_logs/tide_search_phase1_hold_action_20260411.log`
- alternate phase-pair support test:
  - `run_logs/tide_diag_phasepairs_02_03_20260411.log`
- MJCF local-transform candidate test for `servo_l1` / `fixed_carriage_l1`:
  - `run_logs/tide_test_l1_mjcf_pose_candidates_20260411.log`

What these runs showed:
- reset/contact quarantine did not improve the early training failure mode
  - first rollout block in `tide_train_20k_20260411_resetquarantine.log` had `ep_len_mean = 52.1`
  - previous `switchdebug` run started higher at `75.6`
- the phase-start transient is now directly measured:
  - phase 1 starts correct at step `0` as `[0,1,0,1]`
  - `l1` loses contact by step `9`
  - `l0` takes over contact by step `13`
- by step `40`, phase 1 has collapsed to `[1,0,0,1]`

## 2026-04-11: Canonical 4-Phase Gait Migration

The canonical gait definition for this project was updated and should override the earlier two-phase diagonal assumptions.

Authoritative contact semantics:
- `1 = on the ground`
- `0 = off the ground`

Authoritative gait cycle in `[l0, l1, l2, l3]` order:
- `phase 0 = [1,0,0,1]`
  - legs `0` and `3` planted on the ground
  - legs `1` and `2` off the ground / leaned behind
- `phase 1 = [1,1,1,1]`
  - all legs on the ground
- `phase 2 = [0,1,1,0]`
  - legs `1` and `2` planted on the ground
  - legs `0` and `3` off the ground
- `phase 3`
  - returns to `phase 0`

Code work completed:
- `tars_env.py` now uses `PHASE_COUNT = 4`
- `PHASE_CONTACT_MASKS` now encode the canonical gait
- phase-dependent reference control bias is keyed by the 4-phase index
- phase pairing, reset sampling, oscillator timing, support calibration loops, and observation phase clock were migrated off the old hardcoded two-phase cycle

First TIDE validation artifact:
- `run_logs/tide_diag_4phase_support_20260411.log`

Current result:
- phase `1` already holds correctly as `[1,1,1,1]`
- phase `0` still collapses to `[1,0,0,0]`
- phase `2` still collapses to `[1,0,0,0]`

Interpretation:
- the 4-phase infrastructure is wired in and TIDE is exercising the correct gait masks
- the remaining blocker is now pose/support calibration for phase `0` and phase `2`
- targeted direct-pose searches are the next step, not blind PPO retraining

## 2026-04-11: Focused 4-Phase Pose Searches

Targeted TIDE searches were run to determine whether phase `0` and phase `2` can be recovered with direct control bias alone.

Artifacts:
- `run_logs/tide_search_phase0_l0_l3_direct_pose_20260411.log`
- `run_logs/tide_search_phase0_offlegs_direct_pose_20260411.log`
- `run_logs/tide_search_phase2_l1_l2_direct_pose_20260411.log`
- `run_logs/tide_search_phase2_l2_support_pose_20260411.log`

What they showed:
- phase `0` can be pushed from `[1,0,0,0]` toward the canonical target, but the best outer-support search only reached `[1,0,1,1]`
- once the off-leg search was widened, phase `0` improved again to `[1,1,0,1]`
- phase `2` can be pushed from `[1,0,0,0]` toward the canonical target, first reaching `[0,1,0,0]`
- a focused `l2` support search then improved phase `2` to `[0,1,1,1]`

Interpretation:
- the 4-phase gait is no longer failing by full support collapse in phases `0` and `2`
- both missing phases are now blocked by one stubborn extra ground contact:
  - phase `0`: `l1` stays down
  - phase `2`: `l3` stays down
- that means the remaining blocker has narrowed to off-leg unloading / retraction for those specific legs, not total inability to realize the intended support pair
- action-space search around legs `0` and `1` did not matter
  - 625 tested action combinations all produced the same failure timeline
  - `first_l1_loss = 9`
  - `first_l0_touch = 13`
- alternate phase pairing `phase0=(0,2), phase1=(0,3)` also failed
  - phase 1 collapsed to `[1,0,0,0]`
- direct MJCF local-transform candidates for `servo_l1` / `fixed_carriage_l1` did not fix phase 1
  - `current`, `l3_z_only`, `l3_full_local`, and `l0_style_local` all still ended as `[1,0,0,1]`

Current strongest conclusion:
- the phase-1 failure is a real support takeover, not a false contact-registration artifact
- it is not being caused by:
  - reward shaping
  - phase-switch deadlock alone
  - early contact capture during reset
  - simple plant preload
  - nearby leg-0 / leg-1 action adjustments
  - simple alternate second-phase plant pair
  - shallow `servo_l1` / `fixed_carriage_l1` local transform edits
- the remaining blocker is a deeper support-geometry / load-path issue in the physical model, with `l1` unloading and `l0` inheriting support during phase 1

## 2026-04-11: Authoritative 4-Phase Gait Definition

The project gait definition has been superseded by the user-provided 4-phase cycle below.
This is now the canonical reference and should override the older 2-phase alternating assumptions.

Important contact semantics:
- `1` means the leg is on the ground
- `0` means the leg is off the ground

Authoritative contact sequence in leg order `[l0, l1, l2, l3]`:
1. `phase 0 = [1, 0, 0, 1]`
   - legs `0` and `3` on the ground
   - legs `1` and `2` off the ground / leaned behind
2. `phase 1 = [1, 1, 1, 1]`
   - all legs on the ground
3. `phase 2 = [0, 1, 1, 0]`
   - legs `1` and `2` on the ground
   - legs `0` and `3` off the ground
4. `phase 3`
   - returns to `phase 0`

Cycle:
- `[1,0,0,1] -> [1,1,1,1] -> [0,1,1,0] -> [1,0,0,1]`

Implication:
- the current 2-phase diagnostics and controller repairs were useful for isolating the mechanical blocker,
  but the env must now be migrated to this 4-phase gait before future training results are considered authoritative.

## 2026-04-11: Latest Canonical Gait State

This is the authoritative current state and should be referenced ahead of the older intermediate notes above.

Current env architecture:
- runtime cycle is now `phase 0 -> phase 1 -> phase 2 -> phase 0`
- the duplicate full-duration return phase was removed
- per-leg targets are now stored per phase instead of being collapsed into shared `plant` / `swing` role snapshots
- transition roles are now explicit:
  - support
  - touchdown
  - liftoff
  - air
- reference control now interpolates between consecutive phase poses
- reset grounding now aligns the lowest planted foot to the floor to reduce multi-foot embedding

Local verification completed:
- `python3 -m py_compile tars_env.py tide_tars.py train.py training_helpers.py test_gait_phase.py`

Remaining limitation:
- runtime validation still requires TIDE because local `python3` in this shell does not have `mujoco`
## 2026-04-11 Canonical Gait Follow-up

- Preserved the user-defined canonical gait requirements:
  - phase 0: `[1,0,0,1]`
  - phase 1: `[1,1,1,1]`
  - phase 2: `[0,1,1,0]`
  - `1 = on ground`, `0 = off ground`
- Fixed a code-level timing bug where touchdown legs were being pulled toward the next phase from step 0.
- Refactored reference control interpolation to respect phase roles:
  - support legs interpolate normally
  - liftoff legs wait until phase-specific liftoff start
  - touchdown legs wait until late-phase touchdown start
- Added phase-specific liftoff timing with earlier outer-leg liftoff for `phase 1 -> phase 2`.
- Re-ran TIDE diagnostics:
  - `run_logs/tide_diag_canonical_support_20260411_refactor_fix3.log`
  - `run_logs/tide_diag_canonical_support_20260411_refactor_fix4.log`
  - `run_logs/tide_diag_phase_control_consistency_20260411_canonical_fix4.log`
- Re-ran TIDE pose searches:
  - `run_logs/tide_search_phase0_l0_l3_direct_pose_20260411_canonical.log`
  - `run_logs/tide_search_phase0_offlegs_direct_pose_20260411_canonical.log`
  - `run_logs/tide_search_phase2_l1_l2_direct_pose_20260411_canonical.log`
- Current state after these fixes:
  - phase 1 static hold is correct
  - phase 2 improved from all-four-down failure to a one-sided drag problem in training
  - phase 0 still has a marginal extra `l1` contact
  - zero-action control matches the phase reference exactly across all canonical phases
- Training evidence:
  - `run_logs/tide_train_20k_20260411_canonical_fix5.log` improved the phase-2 failure from both outer legs dragging to mainly `l3` dragging
  - `run_logs/tide_train_20k_20260411_canonical_fix6.log` showed early-liftoff helping but phase 2 still blocking on incomplete swing clearance
- Latest code tuning in progress:
  - increased swing vertical authority
  - added leg-3-specific extra lift / later landing
  - validation run: `run_logs/tide_train_20k_20260411_canonical_fix7.log`
- Gait intuition note from user:
  - visualize TARS like a human on crutches
  - legs `0` and `3` behave like the crutches
  - legs `1` and `2` behave like the legs/body support transfer pair
  - this is a useful mental model for debugging canonical phase timing and load transfer

## 2026-04-11 Transition Trace Deep Dive

- Kept the canonical gait requirements unchanged:
  - phase 0: `[1,0,0,1]`
  - phase 1: `[1,1,1,1]`
  - phase 2: `[0,1,1,0]`
  - `1 = on ground`, `0 = off ground`
- Added stricter planted-leg tracing and target logging in `diagnose_phase_transition_trace.py` so each step now records:
  - raw contacts
  - full-plant flags and qualities
  - foot heights
  - target track heights
  - motion roles
  - switch gating fractions / block code
- Reworked touchdown handling in `tars_env.py`:
  - touchdown legs now use touchdown-specific descent limits
  - touchdown legs no longer use the generic swing-lift vertical arc
  - touchdown timing can vary by leg in phase 0
  - touchdown IK authority is stronger than free-swing IK
  - touchdown controller can receive phase/leg-specific extra control offsets during the transition
- Key TIDE traces during this pass:
  - `run_logs/tide_diag_phase0_to_1_trace_20260411_fullplant2.log`
  - the updated inline trace runs from this session showed:
    - original blocker: leg `2` never became fully planted during `0 -> 1`
    - after a search-backed static phase-0 leg-2 bias, leg `2` did plant, but that violated canonical phase 0 by starting with `[1,0,1,1]`
    - that static bias was reverted to preserve the user’s requirements
    - the search result was instead moved into a transition-only touchdown control offset for leg `2`
- TIDE search results used for diagnosis:
  - `run_logs/tide_search_phase0_offlegs_direct_pose_20260411.log`
  - best static search candidates showed leg `2` can be lowered substantially, but using that as a permanent phase-0 pose breaks canonical phase 0 and is therefore not acceptable
  - `run_logs/tide_search_phase0_l0_l3_direct_pose_20260411.log`
  - outer-support search showed phase-0 support is still fragile and does not have an obvious clean static fix inside the tested l0/l3 offset window
- Current state at the end of this pass:
  - phase 0 is canonical again at reset / start
  - phase `0 -> 1` is still the dominant blocker
  - immediate touchdown on leg `2` is better understood but not solved under canonical constraints
  - once leg `2` is driven down harder, support integrity on leg `0` becomes the next limiting issue
  - no user action is needed yet; this is still an in-code gait/pose/controller problem
## 2026-04-11 - Phase-0 snapshot repair and `0 -> 1` transition narrowing

- Repaired the phase-0 start-state corruption by changing `_capture_grounded_phase_snapshot()` to choose between the immediate grounded snapshot and the settled snapshot based on desired-contact score. This restored the live `phase 0` transition trace to the correct opening support mask `[1,0,0,1]`.
- Confirmed on TIDE with `run_logs/tide_diag_phase0_to_1_trace_20260411_snapshotselect.log` that the old blocker is no longer "bad phase-0 start pose." The trace now starts correctly, and `phase 0 -> 1` initially reaches `next_contact = 0.75`, `next_low = 0.75`.
- Added `x` tracing to `diagnose_phase_transition_trace.py` and validated that leg 2's touchdown target had been getting pulled far behind the body in track-target space even though the phase-1 grounded support snapshot itself was reasonable.
- Patched touchdown fore-aft clamping in `_swing_target_world()`. This removed the worst rearward target excursion for leg 2, but did not by itself produce a valid phase transition.
- Confirmed on TIDE with `run_logs/tide_diag_phase_snapshot_geometry_20260411_touchdownx.log` that the phase-1 grounded support snapshot is now mechanically sane: all four feet settle planted, and leg 2's support offset is forward in body coordinates rather than behind.
- Re-ran both transition search harnesses on the repaired build:
  - `run_logs/tide_search_phase0_transition_touchdown_target_offsets_20260411_touchdownxclip.log`
  - `run_logs/tide_search_phase0_transition_touchdown_offsets_20260411_postxfix.log`
- Both searches are still completely flat on the repaired build. Small touchdown target offsets and small touchdown control offsets do not change the `0 -> 1` outcome at all, which means the remaining blocker is deeper than a local controller tweak.
- Tested `SUPPORT_REFERENCE_CTRL_BLEND = 0.0` to force planted-leg IK targets to dominate. This made the `0 -> 1` handoff worse and was reverted. Log: `run_logs/tide_diag_phase0_to_1_trace_20260411_supportblend0.log`.
- Current best diagnosis: `phase 0 -> 1` is still blocked by deeper leg-2 transition geometry / kinematic conversion rather than reward shaping, switch hysteresis, or a small touchdown offset. The next likely investigation is model- or geometry-side around leg 2's track/contact conversion path rather than another small PPO/controller tweak.
