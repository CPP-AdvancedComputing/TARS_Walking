# TARS Project And Experiment Notes

Primary running record:
- `OverallProgress.md`
- Treat that file as the canonical current-status document. Keep updating it on every meaningful change.

## Project Snapshot
- Goal: train TARS to learn a visually coherent alternating gait in MuJoCo and run reliably both locally in VS Code and remotely on TIDE/JupyterHub.
- Main env: `tars_env.py`
- Main trainer: `train.py`
- Remote runner: `tide_tars.py`
- Key diagnostics:
  - `diagnose_phase_support.py`
  - `audit_leg1_geometry.py`
  - `debug_gait_inspector.py`

## Current Defaults
- Training device defaults to CPU in `train.py`.
- Reward profile defaults to `foundation`.
- Max episode steps default to `2000`.
- Live viewer is enabled locally unless `LIVE_VIEWER=0`.

## Current Findings
- Phase 0 is the healthy half of the gait.
- Phase 1 is still the weak half.
- The failure is not just viewer noise.
- The remaining blocker is centered on phase-1 support integrity, especially leg `l1`.
- Raw URDF/MJCF inspection did not show an obvious axis-sign mismatch between `l1` and `l3`.
- Several attempted “fixes” made training worse and were reverted:
  - forced timeout phase switching
  - aggressive body-centering over support centroid
  - much stronger actuator gains

## Kept Changes
- Restored per-leg target generation instead of flattening all legs toward pair means.
- Made paired swing timing symmetric.
- Fixed stale diagnostics that were calling helper methods incorrectly.
- Added persistent diagnostics for phase support and leg-1 geometry.
- Added simpler reward mode and longer episodes.

## Latest Remote Smoke Result
- Date: 2026-04-11
- Reward profile: `foundation`
- Episode max steps: `2000`
- Result:
  - `ep_rew_mean ~277`
  - `progress_gate ~0.826`
  - phase-0 contacts looked coherent

## Open Problems
- Phase 1 still tends to come up as:
  - desired `[0,1,0,1]`
  - actual `[1,0,0,1]`
- `l1` unloads while `l0` re-contacts the floor.
- This still points to a leg-1-specific geometry / support / target-construction issue under load.

## Next High-Value Experiments
1. Compare `l1` vs `l3` support behavior under identical commanded plant targets.
2. Audit whether `l1` needs mirrored target construction relative to `l3`.
3. Render controlled phase-1 hold videos locally with the viewer and save frames.
4. Run longer training only after phase-1 static support is improved.

## Experiment Log

### Template
- Date:
- Goal:
- Code changes:
- Command:
- Result:
- Interpretation:
- Keep / revert:

### 2026-04-11: Simpler Reward + Longer Episodes
- Goal: reduce reward conflicts and allow longer gait attempts.
- Code changes:
  - default reward profile set to `foundation`
  - default max episode steps set to `2000`
- Command:
  - `python3 tide_tars.py smoke-train --timesteps 1024`
- Result:
  - remote smoke completed
  - `ep_rew_mean ~277`
- Interpretation:
  - cleaner baseline than the more bloated reward stack
  - still not enough to solve phase-1 support
- Keep / revert:
  - keep

### 2026-04-11: Phase-Support Diagnostics
- Goal: determine whether failures are real physics failures or viewer artifacts.
- Code changes:
  - added `diagnose_phase_support.py`
  - added `audit_leg1_geometry.py`
- Result:
  - phase 0 coherent
  - phase 1 still weak
- Keep / revert:
  - keep

### 2026-04-11: Switch-Deadlock Instrumentation
- Goal: determine exactly why phase changes still never complete after timeout termination was added.
- Code changes:
  - added `phase_switch_block_code`
  - added `next_plant_contact_fraction`
  - added `next_plant_low_fraction`
  - added `switch_swing_clear_fraction`
  - added hysteresis-based switch diagnostics to TensorBoard/logging
- Command:
  - `python3 tide_tars.py smoke-train --timesteps 100000 > run_logs/tide_train_100k_20260411_switchdebug.log 2>&1`
- Result:
  - `ep_len_mean` remained pinned at `60`
  - `phase_transition_reward` remained `0`
  - the dominant failure pattern was:
    - next plant pair ready
    - swing clearance still `0`
    - transition blocked
- Interpretation:
  - switching was still deadlocked on swing-foot drag even when the incoming support pair was already established
- Keep / revert:
  - keep the diagnostics

### 2026-04-11: Stable-Support Phase Switching
- Goal: stop phase changes from depending on a single-frame perfect swing-clear condition.
- Code changes:
  - switched phase readiness to short history over:
    - next-plant contact fraction
    - next-plant low fraction
    - swing-clear fraction
  - added force-advance arming after stable support persists past `PHASE_STEPS + 8`
  - logged:
    - `history_next_plant_contact_fraction`
    - `history_next_plant_low_fraction`
    - `history_swing_clear_fraction`
    - `phase_switch_force_armed`
- Result:
  - pending fresh TIDE validation
- Interpretation:
  - this is the cleanest way to distinguish “support never becomes viable” from “support is viable but the switch gate is too strict”
- Keep / revert:
  - pending run results

### 2026-04-11: Phase-Control Consistency Fix
- Goal: verify whether `zero_action()` actually reproduces the intended phase reference pose.
- Code changes:
  - added per-phase control calibration so zero-action control matches `phase_reference_ctrl`
- Diagnostics:
  - `run_logs/tide_diag_phase_control_consistency_20260411.log`
- Result:
  - before fix: phase-zero mismatches up to about `0.10 rad`
  - after fix: exact `0.000000` max delta in both phases
- Keep / revert:
  - keep

### 2026-04-11: Support-Pair And Support-Integrity Sweep
- Goal: determine whether the current two-phase support template is simply the wrong gait decomposition.
- Diagnostics:
  - `run_logs/tide_diag_supportpair_02_20260411.log`
  - `run_logs/tide_diag_supportpair_03_20260411.log`
  - `run_logs/tide_diag_supportpair_12_20260411.log`
  - `run_logs/tide_diag_single_swing_support_20260411.log`
- Result:
  - current `[0,2] <-> [1,3]` template has one viable support phase and one failing phase
  - tested alternatives were worse
  - even three-feet-down patterns still drop `l1`
- Interpretation:
  - the remaining blocker is not “pick a different simple contact template”
  - `l1` itself is failing to support load
- Keep / revert:
  - keep the diagnostics

### 2026-04-11: l1-Specific Sweeps
- Goal: test whether the remaining blocker is a controllable `l1` target mismatch or a deeper model/support issue.
- Diagnostics:
  - `run_logs/tide_sweep_phase1_l1_joint_bias_20260411.log`
  - `run_logs/tide_sweep_l1_contact_geometry_z_20260411.log`
  - `run_logs/tide_sweep_l1_actuator_strength_20260411.log`
  - `run_logs/tide_test_l1_geometry_candidates_20260411.log`
- Result:
  - `l1` remained non-load-bearing across:
    - phase-1 joint bias changes
    - lower foot/track z shifts
    - stronger `l1` actuator gains
    - more mirrored anchor/contact candidates
- Interpretation:
  - the remaining issue is no longer well explained by reward, switch logic, or simple controller tuning
  - the strongest current interpretation is a deeper model/support-integrity problem centered on `l1`
- Keep / revert:
  - keep the diagnostics and current code fixes

### 2026-04-11: Reset/Transient And Support-Takeover Diagnosis
- Goal: determine whether the remaining failure is caused by bad reset transients, false early contacts, or a real phase-1 support collapse.
- Code changes:
  - reset now begins directly from the selected phase reference pose
  - plant-target contact capture is delayed after phase start
  - phase support offsets and rod-role targets now come from grounded, settled phase snapshots
  - per-phase plant pairs are configurable with `TARS_PHASE0_PAIR` / `TARS_PHASE1_PAIR`
  - overlapping plant pairs across phases are now legal in controller initialization
- Diagnostics:
  - `run_logs/tide_diag_reset_transient_phasepose_20260411.log`
  - `run_logs/tide_train_20k_20260411_resetquarantine.log`
  - `run_logs/tide_search_phase1_hold_action_20260411.log`
  - `run_logs/tide_diag_phasepairs_02_03_20260411.log`
  - `run_logs/tide_test_l1_mjcf_pose_candidates_20260411.log`
- Result:
  - reset/contact quarantine did not improve early training metrics
    - `tide_train_20k_20260411_resetquarantine.log` started at `ep_len_mean = 52.1`
  - direct phase-start trace showed:
    - phase 1 starts correctly as `[0,1,0,1]`
    - `l1` loses contact by step `9`
    - `l0` takes over by step `13`
    - phase 1 settles to `[1,0,0,1]`
  - 625 leg-0 / leg-1 action combinations produced the same phase-1 failure timing
  - alternate phase pairing `phase0=(0,2), phase1=(0,3)` failed as well, collapsing to `[1,0,0,0]`
  - shallow MJCF local-transform candidates for `servo_l1` / `fixed_carriage_l1` did not restore phase-1 support
- Interpretation:
  - the remaining problem is not just reset bounce or false contact registration
  - phase 1 contains a real support takeover:
    - `l1` unloads
    - `l0` inherits support
  - this failure now appears insensitive to:
    - nearby controller actions
    - simple support-pair changes
    - shallow `l1` MJCF local-transform edits
- Keep / revert:
  - keep the reset/contact-quarantine cleanup
  - keep the settled support-reference snapshots
  - keep the per-phase pair configurability for future experiments
  - treat the blocker as a deeper model/support-geometry problem until disproven

### 2026-04-11: Canonical Gait Spec Updated
- User clarified that the project gait is not the old 2-phase alternating diagonal gait.
- Canonical contact semantics:
  - `1 = on the ground`
  - `0 = off the ground`
- Canonical 4-phase cycle in `[l0, l1, l2, l3]` order:
  - `phase 0 = [1, 0, 0, 1]`
    - legs `0` and `3` on the ground
    - legs `1` and `2` off the ground / leaned behind
  - `phase 1 = [1, 1, 1, 1]`
    - all legs on the ground
  - `phase 2 = [0, 1, 1, 0]`
    - legs `1` and `2` on the ground
    - legs `0` and `3` off the ground
  - `phase 3`
    - returns to `phase 0`
- Implication:
  - all future environment logic should be migrated to this 4-phase definition
  - older 2-phase experiments remain useful for diagnosing the `l1` mechanical issue, but they are no longer the authoritative gait design

### 2026-04-11: 4-Phase Env Conversion And First Static Validation
- Goal: migrate the env off the old two-phase diagonal gait and validate the canonical 4-phase support masks on TIDE.
- Code changes:
  - `tars_env.py` now sets:
    - `PHASE_COUNT = 4`
    - `PHASE_CONTACT_MASKS = {0:[1,0,0,1], 1:[1,1,1,1], 2:[0,1,1,0], 3:[1,0,0,1]}`
  - phase pair derivation now comes from the contact masks instead of the old fixed pair template
  - oscillator timing, reset phase sampling, support-calibration loops, and the observation phase clock were migrated to the 4-phase cycle
  - phase-specific control bias was expanded to 4 indexed phases
  - added direct-pose search scripts for:
    - `search_phase0_l0_l3_direct_pose.py`
    - `search_phase2_l1_l2_direct_pose.py`
- Diagnostics:
  - `run_logs/tide_diag_4phase_support_20260411.log`
- Result:
  - phase `0` desired `[1,0,0,1]`, actual `[1,0,0,0]`
  - phase `1` desired `[1,1,1,1]`, actual `[1,1,1,1]`
  - phase `2` desired `[0,1,1,0]`, actual `[1,0,0,0]`
  - phase `3` desired `[1,0,0,1]`, actual `[1,0,0,0]`
- Interpretation:
  - the canonical gait masks are now active end-to-end
  - the old mechanical `l1` collapse is no longer the only issue under test
  - the next blocker is physical pose calibration for the outer-leg support phase (`0`) and middle-leg support phase (`2`)
- Keep / revert:
  - keep the 4-phase migration
  - continue with targeted TIDE pose searches before any new training run

### 2026-04-11: Focused 4-Phase Pose Search Results
- Goal: determine whether phase `0` and phase `2` can statically realize the canonical 4-phase masks through direct joint-pose bias.
- Diagnostics:
  - `run_logs/tide_search_phase0_l0_l3_direct_pose_20260411.log`
  - `run_logs/tide_search_phase0_offlegs_direct_pose_20260411.log`
  - `run_logs/tide_search_phase2_l1_l2_direct_pose_20260411.log`
  - `run_logs/tide_search_phase2_l2_support_pose_20260411.log`
- Result:
  - phase `0` best coarse outer-leg search:
    - desired `[1,0,0,1]`
    - actual `[1,0,1,1]`
  - phase `0` best widened off-leg search:
    - desired `[1,0,0,1]`
    - actual `[1,1,0,1]`
  - phase `2` best coarse middle-leg search:
    - desired `[0,1,1,0]`
    - actual `[0,1,0,0]`
  - phase `2` best focused `l2` support search:
    - desired `[0,1,1,0]`
    - actual `[0,1,1,1]`
- Interpretation:
  - the canonical gait is now much closer to realizable than the first 4-phase diagnostic suggested
  - phase `0` is blocked by `l1` refusing to unload
  - phase `2` is blocked by `l3` refusing to unload
  - both failures are now “one extra foot still grounded,” not “support pair never forms”
- Keep / revert:
  - keep the 4-phase migration and variable-cardinality reward fixes
  - do not start PPO again yet
  - next step is explicit off-leg unloading calibration for `l1` in phase `0` and `l3` in phase `2`

### 2026-04-11: Latest Canonical Gait Refactor State
- This section supersedes the intermediate 4-slot phase notes above.
- Current code state:
  - runtime cycle is `0 -> 1 -> 2 -> 0`
  - the duplicate full-duration return phase was removed
  - per-phase leg targets replaced the old role-collapsed target store
  - transition roles now distinguish support, touchdown, liftoff, and air
  - reference control interpolates between consecutive phase poses
  - switch readiness evaluates the next phase’s desired grounded and airborne sets directly
  - reset grounding now aligns the lowest planted support foot to the floor
- Local verification:
  - `python3 -m py_compile tars_env.py tide_tars.py train.py training_helpers.py test_gait_phase.py`
- Remaining work:
  - TIDE runtime validation
  - then recalibration only if the remaining extra-contact legs (`l1` in phase 0, `l3` in phase 2) still refuse to unload
## 2026-04-11 Canonical Gait Status

Canonical gait requirement in force:
- phase 0: `[1,0,0,1]`
- phase 1: `[1,1,1,1]`
- phase 2: `[0,1,1,0]`
- `1 = on ground`, `0 = off ground`

Recent code fixes:
- touchdown motion is delayed until late in phase instead of starting immediately
- phase-1 liftoff starts earlier so legs `0` and `3` can clear before phase 2 stalls
- support/pre-touchdown/pre-liftoff controller blending now anchors more strongly to the canonical phase reference
- swing lift authority increased and leg `3` gets extra clearance / later landing

Key diagnostics and results:
- Static support / control consistency:
  - `run_logs/tide_diag_canonical_support_20260411_refactor_fix3.log`
  - `run_logs/tide_diag_canonical_support_20260411_refactor_fix4.log`
  - `run_logs/tide_diag_phase_control_consistency_20260411_canonical_fix4.log`
- Pose searches:
  - `run_logs/tide_search_phase0_l0_l3_direct_pose_20260411_canonical.log`
  - `run_logs/tide_search_phase0_offlegs_direct_pose_20260411_canonical.log`
  - `run_logs/tide_search_phase2_l1_l2_direct_pose_20260411_canonical.log`
- Short train validations:
  - `run_logs/tide_train_20k_20260411_canonical_fix5.log`
  - `run_logs/tide_train_20k_20260411_canonical_fix6.log`
  - `run_logs/tide_train_20k_20260411_canonical_fix7.log`

Latest interpretation:
- The dominant blocker is no longer reward composition.
- The dominant blocker is still gait execution timing and asymmetrical swing clearance, especially leg `3` during `phase 1 -> phase 2`.
- There is no current evidence that the gait requirements themselves need to change.

Design intuition note:
- User analogy: TARS should move like a human on crutches.
- In that analogy, legs `0` and `3` are the crutches and legs `1` and `2` are the legs.
- This should be used as a debugging reference for load transfer, phase timing, and touchdown/liftoff sequencing.

## 2026-04-11 Transition-State Investigation

Canonical gait remains fixed:
- phase 0: `[1,0,0,1]`
- phase 1: `[1,1,1,1]`
- phase 2: `[0,1,1,0]`
- `1 = on ground`, `0 = off ground`

What was instrumented:
- `diagnose_phase_transition_trace.py` now prints per-step:
  - raw contacts
  - fully planted flags / qualities
  - foot heights
  - target track heights
  - motion roles
  - phase-switch fractions and block code

What changed in code during this pass:
- touchdown legs use touchdown-specific vertical motion instead of the generic swing-lift arc
- touchdown timing can vary by leg in phase 0
- touchdown IK authority was increased
- touchdown can receive phase/leg-specific extra control offsets inside `_reference_ctrl_targets`
- static search-backed phase-0 leg-2 bias was tested but reverted because it broke canonical phase 0 by starting with leg 2 on the ground

Key findings from TIDE:
- `0 -> 1` is still the dominant transition blocker under the strict planted-leg rule.
- Leg `2` was originally the main touchdown failure:
  - it stayed high through the transition and never qualified as fully planted
- A static pose search (`run_logs/tide_search_phase0_offlegs_direct_pose_20260411.log`) showed leg `2` can be lowered, but only by changing the static phase-0 pose in a way that violates the canonical gait
- That result is now being used only as a transition-time touchdown control hint, not as a permanent phase pose
- Once leg `2` is improved, support integrity shifts to the outer pair:
  - leg `0` becomes the next main support-quality failure
- Outer-support search (`run_logs/tide_search_phase0_l0_l3_direct_pose_20260411.log`) did not find a clean static l0/l3 pose that preserves both required planted legs inside the tested search window

Current interpretation:
- The user’s gait requirements do not need to change
- Reward is not the blocker
- The blocker is still transition execution:
  - first on touchdown completion for leg `2`
  - then on sustained full-plant support quality for leg `0`
## 2026-04-11 - Transition-state deep dive after snapshot fix

- Canonical gait requirements remain unchanged:
  - phase 0: `[1,0,0,1]`
  - phase 1: `[1,1,1,1]`
  - phase 2: `[0,1,1,0]`
  - `1 = on ground`, `0 = off ground`
- `_capture_grounded_phase_snapshot()` was upgraded to score the immediate grounded pose against the high-gain settled pose and keep the better one. This fixed the live `phase 0` start pose, which had previously been collapsing before any transition logic even ran.
- TIDE trace `run_logs/tide_diag_phase0_to_1_trace_20260411_snapshotselect.log` shows the repaired behavior:
  - step 0 starts as `[1,0,0,1]`
  - step 3 reaches `[1,1,0,1]`
  - the transition still fails because leg 2 never becomes a valid planted next-phase foot
- Added `x` / `target_x` logging to the transition trace. This exposed a real control-path issue: leg 2's touchdown track target was being driven far behind the body during `phase 0 -> 1`, even though the phase-1 grounded support snapshot itself was not rearward.
- Patched touchdown `x` clamping so touchdown targets cannot be yanked backward in one step. Log: `run_logs/tide_diag_phase0_to_1_trace_20260411_touchdownxclip.log`.
- That patch cleaned up the commanded fore-aft target, but leg 2 still failed to plant. This means the remaining problem is not just fore-aft target insanity.
- Re-ran the two existing touchdown search harnesses on the repaired build:
  - target-offset search remained completely flat
  - control-offset search remained completely flat
- Interpretation:
  - the remaining `0 -> 1` blocker is not a small per-leg touchdown offset
  - the next likely issue is deeper geometry or kinematic conversion around leg 2, similar in spirit to the earlier leg-1 support problem
- A support-target blend experiment (`SUPPORT_REFERENCE_CTRL_BLEND = 0.0`) was tested and reverted after it made the `0 -> 1` trace worse. It is not the right fix.
