# Overall Progress

## Purpose

This file is the canonical running record for the TARS walking project.

It exists to answer four questions at any point in time:

1. What problem are we solving right now?
2. What requirements are fixed and must not be changed?
3. What code changes and experiments have already been tried?
4. What is the current blocker?

From this point forward, every meaningful change should be added here.

## Process Rule

- Update this file every time a meaningful code change, diagnostic, or TIDE run is made.
- Keep this file even when context is compacted.
- When the gait is materially better and the workspace is inside a real Git repository, commit and push that improved version.
- Every validation run must be cross-referenced against the canonical reference video:
  - [Screen Recording 2026-04-09 211454.mp4](/mnt/c/Users/anike/tars-urdf/Screen%20Recording%202026-04-09%20211454.mp4)
  - the immediate requirement is faithful phase-change behavior without bypassing transitions
  - exact final walking style can improve later, but no run should be treated as acceptable if it is skipping or faking the intended phase sequence
- Current limitation:
  - from this shell, `/mnt/c/Users/anike/tars-urdf` is not currently recognized as a Git repository, so I can maintain this file now but cannot commit or push from here until the workspace is attached to a real Git checkout.

## Resolved / Do Not Repeat

- Do not repeat pre-pairlock per-leg phase-0 pose searches:
  - canonical phase-0 reset was already repaired by the pair-shared phase-0 offsets plus stronger middle-pair feedback
  - use the deterministic verifier from `start_phase = 0` as the reference check instead
- Do not treat raw touch-only contact as planted support:
  - the project now uses stricter planted-quality semantics
  - edge touch during return does not automatically count as valid support
- Do not use stale return-search harnesses as primary evidence:
  - older timing-only and chain-return sweeps can report flat or misleading results under the current env logic
  - prefer `search_phase2_return_mechanics.py` plus chained trace / verifier output
- Do not accept runs that bypass transitions:
  - contact timing alone is not success
  - every meaningful validation should be cross-checked against the reference video and the current phase mechanics requirements

## Canonical Reference Media

- High-priority visual reference video to cross-check the intended gait:
  - [Screen Recording 2026-04-09 211454.mp4](/mnt/c/Users/anike/tars-urdf/Screen%20Recording%202026-04-09%20211454.mp4)
- This video is now part of the verification workflow and should be used to cross-reference future gait/render checks, especially for:
  - phase timing
  - effective tripedal grouping
  - paired motion of `l0/l3` and `l1/l2`

## Fixed Requirements

These requirements are authoritative and must not be changed unless the user explicitly changes them.

### High-Priority Pairing Constraint

This is a top-priority invariant and must be checked continuously going forward.

- TARS should be evaluated as an effective tripedal walker:
  - leg `l3` acts as one member
  - leg `l0` acts as the second member
  - legs `l1` and `l2` are bound together and act as the third member
- Pair-lock invariants:
  - the theta difference between `l0` and `l3` should be `0`
  - the theta difference between `l1` and `l2` should be `0`
  - all other movements of `l0` and `l3` should stay paired
  - all other movements of `l1` and `l2` should stay paired
  - Verification requirement:
  - build and keep using a dedicated render/inspection tool that explicitly measures these pair-lock constraints during live or rendered rollouts
  - this tool is for ongoing internal verification, not a one-off experiment
  - implementation added:
    - [verify_tripedal_pair_gait.py](/mnt/c/Users/anike/tars-urdf/verify_tripedal_pair_gait.py)
    - supports live viewing, optional GIF rendering, JSON reporting, and explicit pair-lock metrics for:
      - `l0` vs `l3`
      - `l1` vs `l2`
    - updated so the default mode is the built-in reference gait path and it also reports:
      - outer-pair mean theta
      - middle-pair mean theta
      - the gap between those mean thetas
    - updated again so verifier resets can be pinned to a canonical start phase instead of using the env's random training reset
    - updated again so foot pairing is measured as paired body-frame movement deltas from baseline, not naive absolute foot-position distance
  - TIDE integration updated:
    - [tide_tars.py](/mnt/c/Users/anike/tars-urdf/tide_tars.py) now syncs `verify_tripedal_pair_gait.py` to the remote workspace so the verifier can be run on the actual MuJoCo environment
- first real remote verifier result:
  - current built-in reference gait fails the pair-lock requirement badly
  - TIDE verifier report:
      - `outer_pair.theta_diff max ≈ 0.370`
      - `middle_pair.theta_diff max ≈ 0.413`
      - `pair_mean_theta_gap max ≈ 0.347`
      - all verifier verdicts currently fail
    - this makes pair-symmetry enforcement a current top-priority controller fix, not a later refinement
- Pair-symmetry enforcement added in the controller path:
  - [tars_env.py](/mnt/c/Users/anike/tars-urdf/tars_env.py) now applies an explicit tripedal pair-lock projection for:
    - phase reset poses
    - interpolated reference controls
    - final controller targets
  - enforced pairs:
    - outer pair: `l0` and `l3`
    - middle pair: `l1` and `l2`
  - this is the first direct code change aimed specifically at matching the new high-priority tripedal/pair-lock requirement
- Post-pairlock verifier result:
  - the theta requirement improved sharply and is now effectively satisfied in the reference gait path:
    - `outer_pair.theta_diff max ≈ 0.020`
    - `middle_pair.theta_diff max ≈ 0.0034`
  - but actual paired motion is still not good enough:
    - outer shoulder diff max ≈ `0.083`
    - middle shoulder diff max ≈ `0.111`
    - outer foot-track diff max ≈ `0.068`
    - middle foot-track diff max ≈ `0.135`
    - pair-mean-theta gap is still large at ≈ `0.352`
  - conclusion:
    - pair-locked setpoints helped
    - actual dynamic pairing still needs stronger state-following / pair-lock enforcement
- Post-pair-feedback verifier result:
  - outer-pair joint pairing now passes
  - remaining failures are concentrated in:
    - middle-pair joint pairing
    - outer and middle foot-track pairing
  - next adjustment:
    - use stronger tripedal pair-state feedback on the middle pair (`l1/l2`) than on the outer pair (`l0/l3`)
- Post-delta-metric remote verifier result:
  - the newer body-frame baseline-delta foot metric confirms the remaining asymmetry is real, not just a bad metric
  - remote report:
    - `outer_pair.theta_diff max ≈ 0.0024`
    - `middle_pair.theta_diff max ≈ 0.0524`
    - `outer_pair_delta_diff max ≈ 0.223`
    - `middle_pair_delta_diff max ≈ 0.223`
    - `pair_mean_theta_gap max ≈ 0.319`
  - verdict:
    - `outer_theta_ok = true`
    - `middle_theta_ok = false`
    - `outer_joint_pairing_ok = false`
    - `middle_joint_pairing_ok = false`
    - `outer_foot_pairing_ok = false`
    - `middle_foot_pairing_ok = false`
  - interpretation:
    - pair-locked setpoints alone are not enough
    - the live reference / IK target path is still producing real dynamic asymmetry, especially in the middle pair (`l1/l2`)
    - future controller fixes should target upstream target generation and transition control, not just stronger feedback gains
- Forced phase-0 verifier result:
  - reran the tripedal verifier from a deterministic canonical `start_phase = 0` to remove reset-phase randomness from the diagnosis
  - result was worse and more informative than the prior random-start run:
    - the rollout never left phase `0`
    - first stepped trace already showed the wrong effective support pattern: contacts/planted were closer to the middle pair than the required outer pair
    - remote report:
      - `outer_pair.theta_diff max ≈ 0.162`
      - `middle_pair.theta_diff max ≈ 0.051`
      - `outer_pair_delta_diff max ≈ 0.085`
      - `middle_pair_delta_diff max ≈ 0.050`
      - `pair_mean_theta_gap max ≈ 0.303`
    - all verifier verdicts failed
  - interpretation:
    - the reference gait path is not just drifting later in the cycle
    - it is already inconsistent with canonical phase `0` behavior under live stepping
    - next diagnostic priority is to distinguish:
      - bad phase-0 reset pose
      - versus immediate collapse on the first control step
  - verifier improvement added for that:
    - `verify_tripedal_pair_gait.py` now records and prints a dedicated pre-step `initial_state` snapshot so reset-state validity can be separated from first-step instability
  - follow-up result with `initial_state` capture:
    - canonical `start_phase = 0` is already wrong at reset, before the first control step
    - initial state report:
      - `contacts = [0,1,1,0]`
      - `planted = [0,1,1,0]`
    - interpretation:
      - the phase-0 reset/reference pose itself is currently biased toward the middle pair instead of the required outer pair
      - next fix should target phase-0 reset/reference pose generation directly, not only later transition logic
  - new targeted diagnostic added:
    - [search_phase0_pairlock_pose.py](/mnt/c/Users/anike/tars-urdf/search_phase0_pairlock_pose.py)
    - purpose:
      - search phase-0 reset poses using shared outer-pair and shared middle-pair offsets under the current enforced pair-lock architecture
      - avoids redoing stale pre-pairlock per-leg searches
  - result of that new search:
    - TIDE search found multiple pair-shared phase-0 offset combinations that restore the canonical settled phase-0 support pattern exactly:
      - `desired = [1,0,0,1]`
      - `actual = [1,0,0,1]`
      - `planted = [1,0,0,1]`
    - best candidate:
      - outer pair `(l0,l3)` offset: `(shoulder=+0.00, hip=+0.25, knee=-0.05)`
      - middle pair `(l1,l2)` offset: `(shoulder=-0.10, hip=-0.25, knee=-0.10)`
  - code change kept:
    - added phase-specific pair-shared reset/reference offset support in [tars_env.py](/mnt/c/Users/anike/tars-urdf/tars_env.py)
    - applied the best phase-0 pair-shared offsets through that new path instead of baking more asymmetry into per-leg bias tables
  - validation after that fix:
    - forced phase-0 verifier now starts in an effectively canonical planted state:
      - `planted = [1,0,0,1]`
      - one extra raw contact on leg `2` can still appear, but it does not qualify as planted under the stricter rule
    - verifier status improved sharply:
      - `outer_theta_ok = true`
      - `outer_joint_pairing_ok = true`
      - `middle_joint_pairing_ok = true`
      - `outer_foot_pairing_ok = true`
      - `middle_foot_pairing_ok = true`
      - only `middle_theta_ok` still fails, and only narrowly (`max ≈ 0.053` vs `tol = 0.05`)
  - next kept adjustment:
    - raised middle-pair tripedal state feedback gain from `1.25` to `1.50` to try to pull that remaining `l1/l2` theta drift inside tolerance without disturbing the now-correct phase-0 support geometry
  - follow-up validation after the middle-pair gain change:
    - forced phase-0 verifier now fully passes
    - report:
      - `outer_theta_ok = true`
      - `middle_theta_ok = true`
      - `outer_joint_pairing_ok = true`
      - `middle_joint_pairing_ok = true`
      - `outer_foot_pairing_ok = true`
      - `middle_foot_pairing_ok = true`
      - `pass = true`
    - important nuance:
      - raw contact on leg `2` can still appear transiently, but the stricter planted-quality logic still classifies phase-0 support correctly as `[1,0,0,1]`
    - this is the first fully passing deterministic tripedal verifier result for canonical phase `0`

- Contact semantics:
  - `1 = on ground`
  - `0 = off ground`
- Canonical gait phases in `[l0, l1, l2, l3]` order:
  - `phase 0 = [1, 0, 0, 1]`
  - `phase 1 = [1, 1, 1, 1]`
- `phase 2 = [0, 1, 1, 0]`
- runtime cycle returns to `phase 0`
- Interpretation:
  - legs `0` and `3` act like crutches
  - legs `1` and `2` act like the walking legs
- Off-ground meaning:
  - “not touching the ground” does not mean the leg should translate straight upward
  - it should unload by rotating up into swing, analogous to human crutch gait / normal gait swing mechanics
  - support legs should stay planted and stable while the swing legs rotate up and clear the ground
- Return transition clarification:
  - for `phase 2 -> 0`, the edge of the middle legs should touch the ground while the outer legs rotate forward, returning to phase `0`
  - this clarification is authoritative and should be used in future transition tuning and verification
- Phase-2 geometry clarification:
  - in `phase 2`, the outer legs (`l0`, `l3`) should be perpendicular to the ground
  - the inner legs (`l1`, `l2`) should be rotated outward
  - during `2 -> 0`, `l1` and `l2` should lean forward to become perpendicular on the ground
  - at the same time, `l0` and `l3` should rotate forward
  - TARS should lean into that transfer so the outer pair (`l0`, `l3`) become perpendicular to the ground as the return completes
  - this is a high-priority mechanical requirement and should be checked against the reference video, not inferred only from contact timing
  - verification tooling must explicitly log:
    - outer-pair forward offset
    - middle-pair forward offset
    - pairwise forward-offset symmetry
    - these metrics are now part of the ongoing verifier / chain-trace path
- Desired planted-foot interpretation:
  - not just “touching the ground”
  - a planted foot should be meaningfully on the ground, approximated in code by a stricter planted-quality test rather than raw contact alone

## Current Problem Definition

The current dominant problems are:

- `phase 2 -> 0` still needs foundation-level cleanup so the middle pair unloads cleanly on the return transition.
- the dedicated tripedal verifier now also shows that the built-in reference gait path is still dynamically asymmetric even when the commanded setpoints are pair-locked.

More specifically:

- `phase 0 -> phase 1` now has a validated real chained-runtime switch after the pre-step switch fix
- `phase 1 -> phase 2` now also switches in the chained runtime, but too late and only by force-advance
- current direct trace for `phase 1 -> 2` shows:
  - leg `2` does not become a fully planted next-support foot early enough
  - the outer pair unload asymmetrically, with leg `3` clearing much earlier than leg `0`
- the transition must preserve the user’s movement requirement:
  - legs `0` and `3` should remain in place as supports
  - legs `1` and `2` should be the pair that rotate up / transition on their own
- “rotate up” means rotational swing/unload, not pure vertical translation

So the current blockers are no longer "the phase machine will not switch." They are now:

- the quality and timing of the `2 -> 0` return handoff
- and the fact that the reference / target-generation path still violates the high-priority pair-lock gait requirement in live motion, especially for `l1/l2`

Current working hypothesis:

- `phase 2 -> 0` is limited more by middle-pair unload / clearance quality than by the raw phase machine
- the verifier evidence says the middle pair (`l1/l2`) still diverges dynamically under the current live reference / IK path
- static support searches are no longer enough
- the right next fixes should act upstream in target generation and phase-transition control, then be checked with the deterministic tripedal verifier

## Latest Update

- Added geometry-specific verification for the new phase-2 / `2 -> 0` requirement:
  - [verify_tripedal_pair_gait.py](/mnt/c/Users/anike/tars-urdf/verify_tripedal_pair_gait.py) now logs:
    - outer/middle forward mean
    - outer/middle forward pair difference
  - [diagnose_phase_chain_trace.py](/mnt/c/Users/anike/tars-urdf/diagnose_phase_chain_trace.py) now logs:
    - per-leg foot forward offsets
    - outer/middle forward-offset means
  - purpose:
    - stop relying on contact-mask-only interpretation for phase-2 posture and return mechanics
    - make the next return fixes measurable against the user’s clarified geometry
- New direct phase-2 mechanics patch applied:
  - added explicit phase-2 pair-shared offsets in [tars_env.py](/mnt/c/Users/anike/tars-urdf/tars_env.py)
    - outer pair (`l0`,`l3`) biased toward a more perpendicular return-receiver pose
    - middle pair (`l1`,`l2`) biased to stay outward before the lean-forward handoff
  - shifted phase-2 timing:
    - later middle-pair liftoff start (`phase 2` liftoff now starts at `0.50`)
    - earlier outer-pair touchdown start (`phase 2` touchdown now starts at `0.35`)
  - added phase-2 touchdown control offsets for both outer legs so the return tries to rotate them forward instead of waiting for a late contact-only catch
  - rationale:
    - the current non-redundant blocker is still weak phase-2 posture / return geometry
    - this patch is the first direct code change aimed at the newly clarified requirement:
      - phase 2 outer pair perpendicular
      - middle pair outward
      - `2 -> 0` middle pair leans forward while the outer pair rotates forward into perpendicular support
- Follow-up validation result:
  - [run_logs/tide_diag_phase_chain_0_1_2_20260412_phase2posturefix.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase_chain_0_1_2_20260412_phase2posturefix.log)
  - this patch was a regression and has been reverted
  - observed failure:
    - the chained trace never left phase `0`
    - phase `0 -> 1` collapsed to `block=1`
    - support degraded toward `[1,1,1,0]` instead of reaching the clean all-feet-down handoff
    - the phase-2-oriented posture changes contaminated earlier phase behavior through the shared reference path
  - do not repeat:
    - do not apply broad phase-2 pair offsets or aggressive phase-2 touchdown timing changes without first isolating them from phase-0 / phase-1 reference integrity
  - updated interpretation:
    - the next fix for `2 -> 0` should be more localized than broad phase-2 pose forcing
    - likely candidates are transition-only return targets / return-only control offsets rather than phase-wide reference reshaping

- Found a real return-path regression in the support-latch implementation:
  - `_begin_phase()` was not resetting `phase_switch_support_latched`
  - that allowed support readiness from the previous phase to leak into the next phase
  - the result was pathological chain-switching like `2 -> 0 -> 1 -> 2` on partial support
- Fixed in [tars_env.py](/mnt/c/Users/anike/tars-urdf/tars_env.py) by resetting `self.phase_switch_support_latched = False` at phase start
- This bug has to be validated before any further return-transition tuning, because it can make every later trace interpretation invalid
- Clean validation after that fix:
  - `0 -> 1` still switches cleanly
  - `1 -> 2` still switches
  - `2 -> 0` is now blocked for the correct reason instead of false chain-switching
  - current precise return blocker is `swing_clear = 0.00` in phase 2 while support is already good
- New switch-quality tightening:
  - phases `1` and `2` now require full outgoing-pair clearance before a switch is allowed
  - this is intended to stop `1 -> 2` from entering phase 2 with one outer support leg still dragging on the floor
- New timing correction after that:
  - phase `1` was reaching a real fully-valid `[0,1,1,0]` window, but the history+hysteresis path was missing it and only switching later via force-advance after the window had already decayed
  - added a phase-specific instant full-match switch path for phases `1` and `2`
  - this allows those transitions to switch immediately when current-frame support and clearance are all genuinely satisfied, instead of waiting for the slower averaged path
- New phase-2 descent bug found:
  - legs marked as `touchdown` were still being commanded downward from the start of the phase, even before their touchdown window
  - that explains why leg `3` was recontacting the floor immediately after the now-clean `1 -> 2` switch
  - fixed so touchdown legs stay on the swing path until `touchdown_start`, and only then begin the descent path
- Follow-up after validation:
  - that descent-path fix alone was not enough
  - pre-touchdown outer legs were still sagging back to the floor because their target path was not explicitly preserving the airborne pose
  - added a pre-touchdown airborne hold so touchdown legs keep their current off-ground pose until the touchdown window actually begins
- Root cause behind that failed hold:
  - phase-2 pre-touchdown legs were still being overridden by `PRETOUCHDOWN_REFERENCE_CTRL_BLEND = 1.0`
  - that meant the controller was snapping legs `0` and `3` back toward the phase reference even while their foot targets were trying to hold them airborne
  - added a phase-specific override so phase `2` pre-touchdown uses `PRETOUCHDOWN_REFERENCE_CTRL_BLEND = 0.0`
- Latest validation after that:
  - phase `2` no longer looks dominated by immediate outer-leg replanting
  - the real remaining blocker is now correctly isolated to the middle pair (`1` and `2`) still not unloading during the return
  - added the matching phase-specific override so phase `2` pre-liftoff uses `PRELIFTOFF_REFERENCE_CTRL_BLEND = 0.0`

## Current Best Diagnosis

As of the latest diagnostics:

- reward shaping was a real earlier issue, but it is no longer the main blocker
- snapshot capture was a real issue, and phase-0 start corruption has now been repaired
- the `0 -> 1` failure is now narrowed to deeper leg-2 transition geometry / kinematic conversion
- direct leg-2 track-site geometry changes also do not change the outcome
- direct raw leg-2 transition-pose search also does not change the outcome
- phase-0 Jacobian diagnostics do not show leg `2` as obviously kinematically singular or vertically powerless
- phase-0 foot-alignment diagnostics show leg `2`'s track-to-foot vertical separation is dramatically smaller than the other legs
- lowering leg `2`'s foot collision geom by about `0.28 m` materially fixes early `0 -> 1` touchdown quality
- phase `0 -> 1` also needed an earlier switch threshold because the all-feet-down stance is achieved early and then decays
- small touchdown control offsets do not change the outcome
- small touchdown target offsets do not change the outcome
- forcing support legs to use pure support IK instead of the phase reference made the trace worse and was reverted
- the original "switchable" `0 -> 1` trace was only showing readiness, not actual runtime switching
- the live `step()` path was still blocking the transition until `phase_timer >= PHASE_STEPS`
- because `PHASE_STEPS = 30`, phase `0` could become ready early, lose the good four-foot window, and never actually advance
- phase-1 reset/support also regressed after the leg-2 foot fix, so `1 -> 2` has to be evaluated through both:
  - standalone phase-1 support diagnostics
  - real chained runtime traces

The strongest current interpretation is:

- leg `2` is still not receiving a touchdown/support trajectory that it can convert into a real planted support state during `phase 0 -> 1`
- the remaining issue is likely deeper than a local reward tweak or a tiny controller offset
- after the leg-2 foot fix, the next runtime blocker is the mismatch between:
  - per-phase readiness criteria
  - actual phase-switch timing inside `step()`

## What Was Already Fixed Earlier

These issues were already identified and addressed earlier in the project:

- reward was previously dominated by raw progress and allowed non-walking exploit behavior
- progress gating and broader gait-related shaping were restored
- fall/stall exploitation paths were reduced
- absolute world `x/y` leakage in observations was removed
- reset phase randomization was restored
- action space was expanded away from the old shared-pair structure
- pair-locking and per-leg target logic were improved
- phase-switch timeout farming was removed
- multiple diagnostics were added so behavior can be traced directly instead of inferred from PPO reward alone

## Important Diagnostics And What They Mean

### 1. Phase-0 start-state corruption was repaired

Log:
- [run_logs/tide_diag_phase0_to_1_trace_20260411_snapshotselect.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_snapshotselect.log)

Meaning:
- the trace now starts correctly in `phase 0` as `[1,0,0,1]`
- this rules out “bad initial phase support state” as the current dominant `0 -> 1` blocker

### 2. Phase-1 grounded snapshot is mechanically sane

Log:
- [run_logs/tide_diag_phase_snapshot_geometry_20260411_touchdownx.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase_snapshot_geometry_20260411_touchdownx.log)

Meaning:
- phase 1 grounded snapshot settles with all four feet planted
- so the phase-1 support snapshot itself is not obviously rearward or degenerate

### 3. Leg-2 touchdown target had a bad fore-aft command path

Log:
- [run_logs/tide_diag_phase0_to_1_trace_20260411_touchdownx2.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_touchdownx2.log)

Meaning:
- added `x` and `target_x` tracing showed leg `2` had been commanded far behind the body during touchdown
- that was a real control-path bug

### 4. Touchdown `x` clamping improved commanded geometry but did not solve planting

Log:
- [run_logs/tide_diag_phase0_to_1_trace_20260411_touchdownxclip.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_touchdownxclip.log)

Meaning:
- the worst rearward `x` target excursion was removed
- leg `2` still did not plant
- therefore the remaining issue is not just bad fore-aft touchdown target placement

### 5. Small controller and target-offset searches are still flat

Logs:
- [run_logs/tide_search_phase0_transition_touchdown_offsets_20260411_postxfix.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_transition_touchdown_offsets_20260411_postxfix.log)
- [run_logs/tide_search_phase0_transition_touchdown_target_offsets_20260411_touchdownxclip.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_transition_touchdown_target_offsets_20260411_touchdownxclip.log)

Meaning:
- small transition-only control offsets for leg `2` do not materially change the outcome
- small transition-only target offsets for leg `2` do not materially change the outcome
- the blocker is deeper than a small local touchdown tweak

### 6. Pure support IK for support legs was worse

Log:
- [run_logs/tide_diag_phase0_to_1_trace_20260411_supportblend0.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_supportblend0.log)

Meaning:
- setting `SUPPORT_REFERENCE_CTRL_BLEND = 0.0` destabilized support during `0 -> 1`
- that change was reverted

### 7. Direct leg-2 track-site geometry sweep is also flat

Log:
- [run_logs/tide_search_phase0_l2_tracksite_offsets_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_l2_tracksite_offsets_20260411.log)

Meaning:
- leg `2` does have a suspiciously different hardcoded local track/contact conversion than the neighboring legs
- however, directly sweeping substantial local `x/z` changes in the leg-2 track-site anchor did not change the `0 -> 1` transition outcome at all
- so the remaining blocker is deeper than the hardcoded track-site anchor table by itself

### 8. Direct raw leg-2 transition-pose search is also flat

Log:
- [run_logs/tide_search_phase0_l2_direct_transition_pose_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_l2_direct_transition_pose_20260411.log)

Meaning:
- this search bypassed the env's higher-level touchdown conversion and directly blended from a true phase-0 start toward candidate raw leg-2 joint poses during the `0 -> 1` handoff
- even broad leg-2 shoulder/hip/knee transition poses remained flat
- leg `2` still failed to plant and remained high at the end of the transition window
- that makes the blocker look deeper than both:
  - touchdown target conversion
  - small leg-2 controller offsets

### 9. Leg-2 phase-0 Jacobian is not obviously the limiting factor

Log:
- [run_logs/tide_diag_phase0_leg_jacobians_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_leg_jacobians_20260411.log)

Meaning:
- leg `2` remains airborne during the early `phase 0 -> 1` window
- but its foot-track Jacobian does not stand out as catastrophically worse than the other legs
- its vertical sensitivity is in-family with the other legs while it still fails to plant
- that argues against the current blocker being a simple “leg 2 cannot move down because its Jacobian is bad” explanation

### 10. Leg-2 foot-contact alignment looks physically inconsistent

Log:
- [run_logs/tide_diag_phase0_foot_alignment_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_foot_alignment_20260411.log)

Meaning:
- compared each leg's controller track point against the actual foot collision geom center during the early `phase 0 -> 1` window
- leg `2` is a strong outlier in vertical separation:
  - leg 0 track-to-foot `z` delta is about `-0.353`
  - leg 1 track-to-foot `z` delta is about `-0.440`
  - leg 2 track-to-foot `z` delta is about `-0.155` to `-0.160`
  - leg 3 track-to-foot `z` delta is about `-0.422`
- that means the control point used for leg `2` touchdown is much closer to the foot collision geom than for the other legs
- this is the first result that directly supports a physically meaningful explanation for why leg `2` stays high even when the controller is trying to move it down

### 11. Leg-2 foot collision height fix produces a real `0 -> 1` transition window

Logs:
- [run_logs/tide_search_phase0_l2_large_vertical_alignment_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_l2_large_vertical_alignment_20260411.log)
- [run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix.log)
- [run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix_earlyswitch.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix_earlyswitch.log)

Meaning:
- a broad vertical-alignment search found the decisive lever:
  - lowering only leg `2`'s foot collision geom by about `0.28 m`
- that change produces:
  - full four-foot contact during `phase 0 -> 1`
  - full planted quality for all four feet in the early transition window
- after that, the generic switch threshold became the new blocker:
  - the all-feet-down stance was reached early, but the env waited too long to allow switching
- adding a phase-specific earlier switch threshold for `phase 0` made the trace reach `ready=1`
- this is the first validated canonical `0 -> 1` trace that becomes actually switchable

### 12. Current `phase 1 -> 2` baseline is structurally wrong

Log:
- [run_logs/tide_diag_phase1_to_2_trace_20260412_baseline.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase1_to_2_trace_20260412_baseline.log)

Meaning:
- the current `phase 1 -> 2` trace is not respecting the intended movement structure
- it starts from a broken phase-1 stance instead of a stable all-feet-down transition stance
- then it unloads into an all-feet-air failure rather than a controlled rotational swing
- this is the next foundation blocker to solve

### 13. Phase-1 support regressed after the leg-2 foot fix

Logs:
- [run_logs/tide_diag_phase1_snapshot_geometry_20260412_after_l2fix.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase1_snapshot_geometry_20260412_after_l2fix.log)
- [run_logs/tide_search_phase1_outer_support_pose_20260412.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase1_outer_support_pose_20260412.log)
- [run_logs/tide_diag_phase1_snapshot_geometry_20260412_outerfix2.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase1_snapshot_geometry_20260412_outerfix2.log)

Meaning:
- after the leg-2 foot collision-height correction, direct phase-1 support no longer settled into `[1,1,1,1]`
- an outer-leg support search found plausible phase-1 outer-leg biases, but the corrected bias patch still did not restore robust standalone all-feet-down phase-1 support
- that means phase 1 must now be judged through both standalone support diagnostics and true chained runtime entry, not by snapshot assumptions alone

### 14. The earlier `0 -> 1` "switchable" trace exposed a real runtime bug

Log:
- [run_logs/tide_diag_phase_chain_0_1_2_20260412.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase_chain_0_1_2_20260412.log)

Meaning:
- the earlier standalone `0 -> 1` trace showed `ready=1` by step 6
- but the real chained runtime trace never actually left phase 0 within 40 steps
- comparing that against the code showed the reason:
  - readiness was computed using phase-specific `min_progress`
  - but actual switching in `step()` still required `phase_timer >= PHASE_STEPS`
- because `PHASE_STEPS = 30`, the runtime was waiting far longer than the phase-0 readiness window lasted
- this is a true code-path bug, not a diagnostic artifact

### 15. Runtime switching now advances immediately once readiness is met

Code change:
- [tars_env.py](/mnt/c/Users/anike/tars-urdf/tars_env.py)

Meaning:
- phase transitions no longer wait for the full nominal phase duration once the per-phase readiness gate has been satisfied
- the readiness gate already enforces:
  - phase-specific minimum progress
  - planted-foot requirements
  - swing-clear requirements
  - hysteresis
- this removes the false-positive distinction between:
  - "ready in the diagnostic trace"
  - "still not allowed to switch in the real env step path"

### 16. A second timing bug remained inside `step()`

Meaning:
- even after removing the `phase_timer >= PHASE_STEPS` gate, the env still only checked switching after simulating one more control step
- the chained trace showed the valid `phase 0 -> 1` window existed at the start of step 6, then decayed during the extra simulated step before the runtime ever advanced
- `step()` now performs a pre-step readiness check and switches immediately when the current state is already ready
- this aligns the live runtime with the diagnostic meaning of:
  - "the current state is ready to switch now"

### 17. Real chained runtime now confirms both `0 -> 1` and `1 -> 2` can advance

Log:
- [run_logs/tide_diag_phase_chain_0_1_2_20260412_preswitchfix.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase_chain_0_1_2_20260412_preswitchfix.log)

Meaning:
- `phase 0 -> 1` now switches in the real chained runtime at step 6
- `phase 1 -> 2` also switches, at step 50
- but `phase 1 -> 2` is still not healthy:
  - it only advances after the force-advance path arms
  - leg `2` does not become a fully planted next-support foot until very late
  - outer-leg unloading is asymmetric, with leg `3` clearing much earlier than leg `0`
- this changes the blocker from:
  - "phase transitions do not happen"
  to:
  - "`phase 1 -> 2` happens too late and with the wrong support-transfer shape"

### 18. Phase-1 support legs were still being driven by reference interpolation

Meaning:
- in `phase 1`, legs `1` and `2` are the support pair for the next phase
- but `_ctrl_targets_from_action()` was still blending support legs fully toward the interpolated phase reference
- that means leg `2` could start following the `phase 1 -> 2` reference path instead of holding its captured planted target, even while it was supposed to be the stabilizing support foot
- this is a controller-side cause for the late/weak `1 -> 2` handoff
- attempted fix:
  - phase-specific support-reference blend override
  - `phase 1` support legs used planted-target IK directly instead of full reference blending
- result:
  - regression
  - [run_logs/tide_diag_phase_chain_0_1_2_20260412_phase1supporthold.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase_chain_0_1_2_20260412_phase1supporthold.log)
  - `phase 1 -> 2` no longer switched within 60 steps
  - leg `2` stayed weakly planted and never recovered to full planted quality
- action taken:
  - reverted
  - do not retry this exact phase-1 support-blend removal

### 19. New diagnostic direction: runtime-scored phase-1 leg-2 bias search

File:
- [search_phase1_chain_support_offsets.py](/mnt/c/Users/anike/tars-urdf/search_phase1_chain_support_offsets.py)

Meaning:
- this search does not evaluate a static snapshot
- it runs the real chained `0 -> 1 -> 2` transition
- it scores candidates by:
  - whether `phase 2` is reached
  - how early the switch happens
  - best next-support contact fraction
  - best next-support low fraction
  - swing-clear fraction
  - leg `2` planted quality during phase 1
- this is the current non-redundant next step for fixing the `1 -> 2` blocker

### 20. TIDE sync gap fixed for the new phase-1 runtime search

Meaning:
- the first launch of `search_phase1_chain_support_offsets.py` failed because `tide_tars.py` did not include it in `SYNC_FILES`
- this was an infrastructure issue, not a gait result
- fixed by adding the script to the TIDE sync list

### 21. Runtime-scored phase-1 leg-2 search found a useful control bias

Log:
- [run_logs/tide_search_phase1_chain_support_offsets_20260412.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase1_chain_support_offsets_20260412.log)

Meaning:
- the chained runtime objective is not flat
- the clear winning family is a more positive leg-2 hip offset in `phase 1`
- best class found:
  - leg `2` hip offset `+0.25`
  - shoulder and knee offsets were largely irrelevant in the tested range
- outcome improved from:
  - `phase 1 -> 2` switching at step `50`
  to:
  - switching at step `37`
  while also reaching:
  - `best_next_contact = 1.00`
  - `best_next_low = 1.00`
  - `best_swing_clear = 1.00`
  - `best_leg2_quality = 0.850`
- minimal code action taken:
  - updated `PHASE_CTRL_BIAS[1]` so leg `2` hip bias is `+0.25`

### 22. The phase-1 leg-2 hip fix materially improves `1 -> 2`

Log:
- [run_logs/tide_diag_phase_chain_0_1_2_20260412_leg2hipfix.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase_chain_0_1_2_20260412_leg2hipfix.log)

Meaning:
- `phase 1 -> 2` no longer relies on the late force-advance path from the older trace
- it now switches at step `35` instead of step `50`
- leg `2` reaches strong planted quality before the switch:
  - step `033`: leg-2 quality `0.849`
  - step `034`: leg-2 quality `0.849`
- the next-support readiness metrics are clean before switching:
  - `next_contact = 1.00`
  - `next_low = 1.00`
  - `swing_clear = 1.00`
- this means the `1 -> 2` blocker is no longer “leg 2 never becomes a valid next-support foot”
- the next exposed transition weakness is now `phase 2 -> 0`, where swing-clear remains `0.00` later in phase 2

### 23. Next adjustment: earlier phase-2 liftoff timing

Meaning:
- after the leg-2 phase-1 fix, the limiting issue moves to `phase 2 -> 0`
- in the chained trace, the next-support pair (`0`, `3`) becomes available, but the next-air pair (`1`, `2`) does not clear
- that points to a timing problem first, not a missing-support problem
- attempted code change:
  - `PHASE_LIFTOFF_START_PROGRESS[2]` changed from `0.60` to `0.25`
- result:
  - regression
  - [run_logs/tide_diag_phase_chain_0_1_2_20260412_phase2liftofffix.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase_chain_0_1_2_20260412_phase2liftofffix.log)
  - leg `2` shot upward aggressively in phase 1 and then stayed too airborne/high
  - `phase 1 -> 2` was no longer the improved version from the leg-2 hip fix
- action taken:
  - reverted
  - do not retry this exact global phase-2 liftoff timing change

### 24. New diagnostic direction: runtime-scored `phase 2 -> 0` return search

Files:
- [search_phase2_chain_return_offsets.py](/mnt/c/Users/anike/tars-urdf/search_phase2_chain_return_offsets.py)
- [tide_tars.py](/mnt/c/Users/anike/tars-urdf/tide_tars.py)

Meaning:
- the next blocker is no longer a generic timing guess
- the new search evaluates the real chained return transition
- it scores candidates by:
  - whether phase `0` is reached after phase `2`
  - how early the return switch happens
  - next-support contact and low fractions
  - swing-clear quality
  - direct clearance of legs `1` and `2`
- this is the current non-redundant path for fixing the return transition / user-described `phase 2 -> 3`

### 25. First return search result: no full `2 -> 0` fix yet

Log:
- [run_logs/tide_search_phase2_chain_return_offsets_20260412.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase2_chain_return_offsets_20260412.log)

Meaning:
- no candidate in the first phase-2 return search actually completed the return switch
- the best family improved the ingredients:
  - `best_next_contact = 1.00`
  - `best_next_low = 1.00`
  - `best_swing_clear = 0.50`
- but that was still not enough to reach phase `0`
- best candidate family from the first search:
  - leg `1` hip `+0.50`
  - leg `2` hip `-0.50`
  - leg `2` knee `-0.10`
- next action:
  - patch the best candidate directly into `PHASE_CTRL_BIAS[2]`
  - inspect the full chained trace before deciding whether to keep or revert it

### 26. First direct phase-2 candidate was not good enough

Log:
- [run_logs/tide_diag_phase_chain_0_1_2_20260412_phase2candidate1.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase_chain_0_1_2_20260412_phase2candidate1.log)

Meaning:
- the candidate changed the return-transition shape, but it still did not complete `phase 2 -> 0`
- it temporarily improved phase-2 partial clearance, but only one of the returning swing legs cleared at a time
- `swing_clear` never became valid for the full pair, so the return transition still failed
- action taken:
  - reverted
  - keep the cleaner `phase 1` leg-2 hip fix as baseline

### 27. New return diagnostic: per-leg phase-2 liftoff timing search

Files:
- [search_phase2_chain_timing_offsets.py](/mnt/c/Users/anike/tars-urdf/search_phase2_chain_timing_offsets.py)
- [tide_tars.py](/mnt/c/Users/anike/tars-urdf/tide_tars.py)

Meaning:
- the current return blocker is about overlap, not just static bias
- this search explores per-leg phase-2 liftoff timing for legs `1` and `2`, plus phase-2 touchdown timing
- it scores the real chained return by:
  - whether phase `0` is reached
  - how early the return switch happens
  - next-support contact / low fractions
  - swing-clear fraction
  - direct per-leg clearance of legs `1` and `2`
- this is the current non-redundant next attempt after the first phase-2 bias search failed to complete the return

### 28. Phase-2 timing search result: leg `1`, not leg `2`, is the remaining return blocker

Log:
- [run_logs/tide_search_phase2_chain_timing_offsets_20260412.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase2_chain_timing_offsets_20260412.log)

Meaning:
- no timing candidate completed the return switch
- the best family still only reached:
  - `best_swing_clear = 0.50`
  - `best_leg1_clear = 0.00`
  - `best_leg2_clear = 1.00`
- this isolates the return blocker further:
  - leg `2` can clear
  - leg `1` is the leg that refuses to clear in phase 2
- next action:
  - switch from timing search to a leg-1-specific phase-2 runtime bias search

### 29. New return diagnostic: phase-2 leg-1 runtime bias search

Files:
- [search_phase2_leg1_return_offsets.py](/mnt/c/Users/anike/tars-urdf/search_phase2_leg1_return_offsets.py)
- [tide_tars.py](/mnt/c/Users/anike/tars-urdf/tide_tars.py)

Meaning:
- this search targets only the remaining stuck return leg
- it scores the real chained return with extra weight on:
  - leg `1` clearance
  - full swing-clear
  - actual return completion to phase `0`

### 30. Phase-2 leg-1 search shows the return window is achievable

Log:
- [run_logs/tide_search_phase2_leg1_return_offsets_20260412.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase2_leg1_return_offsets_20260412.log)

Meaning:
- no candidate completed the return in the coarse search
- but many candidates achieved the full ingredient set:
  - `best_next_contact = 1.00`
  - `best_next_low = 1.00`
  - `best_swing_clear = 1.00`
  - `best_leg1_clear = 1.00`
  - `best_leg2_clear = 1.00`
- this is an important shift:
  - the return transition is no longer blocked by "leg 1 cannot clear"
  - the remaining issue is now how the valid return window aligns with the actual phase-2 switch path
- direct validation action:
  - patched a minimal top candidate into `PHASE_CTRL_BIAS[2]`
  - leg `1` shoulder `-0.20`
  - leg `1` knee `-0.10`
  - keeping the rest of the phase-2 bias unchanged

### 31. Return-switch support latch added

Meaning:
- the phase-2 return search showed the full return ingredients can occur, but not necessarily at the exact same instant
- the env previously required:
  - next support pair valid now
  - swing pair clear now
- if the next support pair became valid first and stayed valid, the env could still lose the window before the swing pair cleared
- code change:
  - added `phase_switch_support_latched`
  - once next-support readiness is achieved within a phase, it remains latched until the phase changes
- goal:
  - allow the return transition to use a valid support window that occurs slightly before the swing-clear window
  - without changing the canonical gait requirements

## Current Code Changes Kept

These are meaningful recent changes that remain in the code:

- canonical gait masks and 3-phase runtime cycle for:
  - phase `0 = [1,0,0,1]`
  - phase `1 = [1,1,1,1]`
  - phase `2 = [0,1,1,0]`
- stricter planted-foot proxy instead of raw touch-only logic
- phase-aware `_ground_support_feet()`
- `_capture_grounded_phase_snapshot()` now selects the better of:
  - immediate grounded snapshot
  - settled snapshot
- runtime phase switching now advances immediately when `_phase_switch_ready()` succeeds, instead of waiting for `phase_timer >= PHASE_STEPS`
- `step()` now does a pre-step readiness check so already-valid contact windows are not lost during one extra simulated control step
- transition tracing now logs:
  - contacts
  - planted flags
  - planted qualities
  - heights
  - `x`
  - `target_x`
  - `target_z`
  - phase switch readiness fractions
  - block code
- touchdown target path fixes:
  - touchdown legs now have their own fore-aft interpolation path
  - touchdown fore-aft motion is clamped with `TOUCHDOWN_TARGET_MAX_X_STEP`
- added diagnostic search:
  - `search_phase0_l2_tracksite_offsets.py`
  - `search_phase0_l2_direct_transition_pose.py`
- added diagnostic trace:
  - `diagnose_phase0_leg_jacobians.py`
  - `diagnose_phase0_foot_alignment.py`
- added diagnostic search:
  - `search_phase0_l2_large_vertical_alignment.py`
- added transition diagnostic reference for the next blocker:
  - `run_logs/tide_diag_phase1_to_2_trace_20260412_baseline.log`

## Changes Tried And Reverted

These were tried, validated, and then reverted or left unused because they made behavior worse or did not help:

- `SUPPORT_REFERENCE_CTRL_BLEND = 0.0`
  - made the `0 -> 1` trace worse
  - support destabilized
- several earlier global strategies from prior sessions:
  - forced timeout switching
  - aggressive body-centering over support centroid
  - much stronger actuator gains
- using static pose-search results directly as canonical phase poses when they violated the user’s gait requirements

## TIDE Runs And Artifacts To Reference First

If resuming later, these are the most relevant logs for the current blocker:

- repaired `0 -> 1` baseline:
  - [run_logs/tide_diag_phase0_to_1_trace_20260411_snapshotselect.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_snapshotselect.log)
- `x`-diagnostic trace:
  - [run_logs/tide_diag_phase0_to_1_trace_20260411_touchdownx2.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_touchdownx2.log)
- post-`x`-clamp trace:
  - [run_logs/tide_diag_phase0_to_1_trace_20260411_touchdownxclip.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_touchdownxclip.log)
- failed support-IK experiment:
  - [run_logs/tide_diag_phase0_to_1_trace_20260411_supportblend0.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_supportblend0.log)
- current flat search evidence:
  - [run_logs/tide_search_phase0_transition_touchdown_offsets_20260411_postxfix.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_transition_touchdown_offsets_20260411_postxfix.log)
  - [run_logs/tide_search_phase0_transition_touchdown_target_offsets_20260411_touchdownxclip.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_transition_touchdown_target_offsets_20260411_touchdownxclip.log)
  - [run_logs/tide_search_phase0_l2_tracksite_offsets_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_l2_tracksite_offsets_20260411.log)
  - [run_logs/tide_search_phase0_l2_direct_transition_pose_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_l2_direct_transition_pose_20260411.log)
  - [run_logs/tide_diag_phase0_leg_jacobians_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_leg_jacobians_20260411.log)
  - [run_logs/tide_diag_phase0_foot_alignment_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_foot_alignment_20260411.log)
  - [run_logs/tide_search_phase0_l2_large_vertical_alignment_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_l2_large_vertical_alignment_20260411.log)
  - [run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix.log)
  - [run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix_earlyswitch.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix_earlyswitch.log)

## Current Status

Current state of the project:

- canonical gait requirements are preserved
- phase-0 start corruption is fixed
- touchdown `x` target corruption was partially fixed
- `phase 0 -> 1` still fails because leg `2` does not become a valid planted support foot
- current blocker is still solvable from code/model work; no user input is required yet

## Active Blockers To Watch

- Contact-timing exploit risk:
  - the robot may be satisfying switch readiness by hitting the floor at the right time rather than expressing the intended support-transfer mechanics
  - this is now an explicitly tracked blocker and must be ruled out in future diagnostics
- Reset/settle bobbing risk:
  - unequal limb loading during spawn or settle can create fake phase structure or misleading contact order
  - this remains a live blocker to watch even when a transition trace looks superficially successful
- Return-transition mechanical requirement:
  - for `phase 2 -> 0`, the edge of the middle legs should touch the ground while the outer legs rotate forward back into phase `0`
  - future diagnostics must measure this directly, not infer it from raw contact timing alone

## Immediate Next Steps

1. Continue leg-2-specific model/geometry investigation rather than more small controller sweeps.
2. Compare leg `2` against the analogous leg under equivalent support conversion paths.
3. Inspect whether the leg-2 foot/track/contact conversion is mechanically or geometrically inconsistent during touchdown.
4. Only resume broader PPO learning once `0 -> 1` establishes phase 1 more reliably in direct traces.

### 2026-04-11 - Leg-2 track-site geometry sweep

- Compared hardcoded local track/contact conversion across legs and found leg `2` is a strong outlier.
- Added `search_phase0_l2_tracksite_offsets.py`.
- Synced it through `tide_tars.py`.
- Ran TIDE sweep:
  - [run_logs/tide_search_phase0_l2_tracksite_offsets_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_l2_tracksite_offsets_20260411.log)
- Result:
  - all tested leg-2 local track-site `x/z` deltas were flat
  - the best result remained the unchanged baseline
- Interpretation:
  - leg-2 hardcoded contact/track geometry is suspicious enough to inspect
  - but changing that local anchor alone does not fix `phase 0 -> 1`
  - the blocker is deeper than the track-site table by itself

### 2026-04-11 - Direct raw leg-2 transition-pose search

- Added `search_phase0_l2_direct_transition_pose.py`.
- Synced it through `tide_tars.py`.
- Ran TIDE search:
  - [run_logs/tide_search_phase0_l2_direct_transition_pose_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_l2_direct_transition_pose_20260411.log)
- Result:
  - all broad leg-2 transition-pose candidates remained effectively flat
  - best pattern still only reached `[1,1,0,1]`
  - final leg-2 height stayed high, around `0.187`
- Interpretation:
  - even bypassing the env’s touchdown conversion and blending directly toward candidate raw joint targets did not make leg `2` plant during `0 -> 1`
  - the blocker is therefore deeper than both:
    - touchdown target conversion
    - small per-leg control offsets

### 2026-04-11 - Phase-0 leg Jacobian diagnostic

- Added `diagnose_phase0_leg_jacobians.py`.
- Synced it through `tide_tars.py`.
- Ran TIDE diagnostic:
  - [run_logs/tide_diag_phase0_leg_jacobians_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_leg_jacobians_20260411.log)
- Result:
  - leg `2` stays airborne while `l1` touches down
  - but leg `2`'s foot-track Jacobian singular values and vertical row norm are not catastrophically worse than the other legs
- Interpretation:
  - the current blocker is not well explained by a simple leg-2 Jacobian / vertical-authority failure
  - next likely layer to inspect is foot-contact alignment and how the actual foot collision geometry relates to the track target during touchdown

### 2026-04-11 - Phase-0 foot-alignment diagnostic

- Added `diagnose_phase0_foot_alignment.py`.
- Synced it through `tide_tars.py`.
- Ran TIDE diagnostic:
  - [run_logs/tide_diag_phase0_foot_alignment_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_foot_alignment_20260411.log)
- Result:
  - leg `2`'s controller track point is much closer vertically to its foot collision geom than the other legs
  - measured track-to-foot `z` delta for leg `2` is only about `-0.16`, while the others are about `-0.35` to `-0.44`
- Interpretation:
  - this is a strong candidate root cause
  - if the controller uses a track point that is too close to the real foot geom, touchdown targeting will under-drive the real foot downward
  - the next correct experiment is a broader leg-2 vertical geometry correction, because the earlier `+0.12` local sweep was too small relative to the measured discrepancy

### 2026-04-11 - Broad leg-2 vertical-alignment search and fix

- Added `search_phase0_l2_large_vertical_alignment.py`.
- Synced it through `tide_tars.py`.
- Ran TIDE search:
  - [run_logs/tide_search_phase0_l2_large_vertical_alignment_20260411.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_search_phase0_l2_large_vertical_alignment_20260411.log)
- Result:
  - lowering leg `2`'s foot collision geom by about `0.28 m` immediately changed the transition qualitatively
  - best candidates reached:
    - `best_next_contact = 1.00`
    - `best_next_low = 1.00`
    - full four-foot contact during `phase 0 -> 1`
- Patched code:
  - lowered `FOOT_CONTACT_GEOMETRY["fixed_carriage_l2"]["pos"][2]` by `0.28`
- Validation traces:
  - [run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix.log)
  - [run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix_earlyswitch.log](/mnt/c/Users/anike/tars-urdf/run_logs/tide_diag_phase0_to_1_trace_20260411_l2footfix_earlyswitch.log)
- Additional code change:
  - added `PHASE_SWITCH_MIN_PROGRESS_BY_PHASE = {0: 0.10}` so the all-feet-down transition stance can switch when it is actually achieved, instead of waiting until it has already decayed
- Interpretation:
  - this is the first concrete fix that makes the canonical `phase 0 -> 1` transition genuinely switchable in direct traces
  - the next step is short training validation rather than more low-level transition guessing

### 2026-04-12 - Clarified off-ground semantics for stage 1 to 2

- User clarified that “not touching the ground” does not mean the leg should go straight upward.
- Correct interpretation:
  - the leg should rotate up into swing / unload, analogous to crutch gait and normal swing mechanics
  - support legs stay planted and stable
  - swing legs clear by rotational motion, not pure vertical translation
- This is now a fixed movement constraint for `phase 1 -> 2`.
- External references used for this interpretation:
  - Physiopedia summary of crutch gait patterns and support sequencing: https://www.physio-pedia.com/Crutches
  - Crutch-assisted gait phase description distinguishing crutch stance from crutch swing: https://pmc.ncbi.nlm.nih.gov/articles/PMC11175161/

### 2026-04-12 - Added mechanics-focused return diagnostics

- Concern logged:
  - the robot may be “learning” phase changes by contacting the ground at the right times and places rather than performing the intended gait mechanics
  - user also observed independent bobbing and apparent unequal ground-connected limb loading
- Diagnostic upgrades added:
  - [diagnose_phase_transition_trace.py](/mnt/c/Users/anike/tars-urdf/diagnose_phase_transition_trace.py)
  - [diagnose_phase_chain_trace.py](/mnt/c/Users/anike/tars-urdf/diagnose_phase_chain_trace.py)
- New metrics added to those traces:
  - centered hip values for all four legs
  - body-frame track `x` values for all four legs
  - outer-pair forward-rotation mean
  - middle-pair forward-rotation mean
  - middle-leg edge-contact proxy flags
- Purpose:
  - directly distinguish real return mechanics from contact-timing exploits
  - specifically for `2 -> 0`, verify that:
    - the middle-leg edge touches the ground
    - while the outer legs rotate forward into the returning phase

### 2026-04-12 - Return-mechanics trace result for `2 -> 0`

- Ran the upgraded phase traces on TIDE for phase `2`:
  - isolated transition trace from phase `2`
  - chained trace starting from phase `2`
- Result:
  - this does not primarily look like a fake switch caused by random contact timing
  - the phase never becomes ready because the required return mechanics do not happen strongly enough
- Specific findings:
  - outer-pair forward rotation stays very small through the whole trace
    - `outer_fwd` only grows from about `+0.023` to about `+0.043`
  - middle pair remains strongly planted for most of the trace
    - leg `2` stays heavily planted throughout
    - leg `1` only briefly shows edge-touch behavior, then returns to full planting
  - middle-leg edge-contact proxy does not sustain the intended return behavior
    - transient `mid_edge=[1,0]` appears briefly
    - but it does not develop into the intended “middle-leg edge touch while outer legs rotate forward” mechanic
  - switch readiness eventually fails with `block=1`
    - by then `next_contact=0.00`
    - `swing_clear=0.00`
    - so the return has fully decayed rather than being narrowly missed
- Interpretation:
  - the current blocker for `2 -> 0` is mechanical/reference-path weakness, not just noisy contact timing
  - the outer pair is not being driven far enough forward during return
  - the middle pair is not unloading into the clarified edge-touch return behavior
  - next fixes should target phase-2 reference/target generation directly, especially:
    - stronger outer-leg forward rotation during return
    - controlled middle-leg unload from full plant into edge-touch
  - additional logic blocker found:
    - `_phase_switch_ready()` was still computing swing clearance from raw foot-floor contact for next-air legs
    - that conflicts with the clarified planted semantics and the `2 -> 0` requirement where middle-leg edge touch can exist during return without counting as true support
  - code change kept:
    - swing-clear readiness now uses `not foot_fully_planted` instead of `not foot_on_ground`
  - rationale:
    - this aligns switch logic with the project's stricter “meaningfully planted” interpretation
    - and prevents edge-touch transients from automatically invalidating an otherwise correct return mechanic
  - follow-up tooling blocker:
    - the old `search_phase2_chain_timing_offsets.py` harness no longer produces meaningful scores against the current refactored env
    - every candidate returned the sentinel failure score without ever observing a valid phase-2 scoring window
    - interpretation:
      - the harness assumptions are stale and should not be used as evidence for current return timing decisions
  - direct timing change kept after that:
    - phase `2` middle-leg liftoff now starts earlier:
      - `PHASE_LIFTOFF_START_PROGRESS[2] = 0.35`
    - phase `2` outer-leg touchdown now starts earlier:
      - `PHASE_TOUCHDOWN_START_PROGRESS[2] = 0.55`
  - rationale:
    - current traces showed middle support lingering too long and outer forward-rotation starting too late
    - this change is intended to force the return mechanics to begin earlier in the phase rather than letting the old support pattern dominate most of phase `2`
  - follow-up result:
    - the earlier timing change only moved the trace marginally
    - outer forward rotation remained weak
    - middle-edge behavior remained brief and unstable
    - conclusion:
      - timing alone is not enough to fix `2 -> 0`
  - new tooling kept:
    - [search_phase2_return_mechanics.py](/mnt/c/Users/anike/tars-urdf/search_phase2_return_mechanics.py)
  - purpose:
    - replace the stale phase-2 search harnesses with a scorer aligned to current project requirements
    - score:
      - outer forward rotation
      - middle edge-touch behavior
      - planted-aware readiness metrics
      - actual return switch if achieved
  - harness fix:
    - initial run of `search_phase2_return_mechanics.py` failed because it did not initialize the env's step bookkeeping fields (`initial_height`, `prev_body_z`, etc.) before calling `step()`
    - fixed in the script and rerun from that corrected state

## Update Log

### 2026-04-11 - File created

- Created `OverallProgress.md` as the canonical running record.
- Consolidated the current problem, requirements, diagnostics, and blocker into one place.
- Going forward, every meaningful change should be added here first.
