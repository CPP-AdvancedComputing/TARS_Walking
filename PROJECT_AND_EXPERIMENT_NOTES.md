# TARS Project And Experiment Notes

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
