from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR


def curriculum_values(
    progress,
    warmup_fraction=0.2,
    nominal_start=1.0,
    nominal_end=0.35,
    pair_start=1.0,
    pair_end=1.0,
):
    progress = float(np.clip(progress, 0.0, 1.0))
    if progress <= warmup_fraction:
        return nominal_start, pair_start
    alpha = (progress - warmup_fraction) / max(1e-9, 1.0 - warmup_fraction)
    nominal = nominal_start + alpha * (nominal_end - nominal_start)
    pair = pair_start + alpha * (pair_end - pair_start)
    return float(nominal), float(pair)


class CurriculumCallback(BaseCallback):
    """Gradually relax the hand-authored gait scaffolding during PPO training."""

    def __init__(
        self,
        total_timesteps,
        warmup_fraction=0.2,
        nominal_end=0.35,
        pair_end=1.0,
    ):
        super().__init__()
        self.total_timesteps_target = max(int(total_timesteps), 1)
        self.warmup_fraction = warmup_fraction
        self.nominal_end = nominal_end
        self.pair_end = pair_end

    def _current_values(self):
        progress = self.num_timesteps / self.total_timesteps_target
        return curriculum_values(
            progress,
            warmup_fraction=self.warmup_fraction,
            nominal_end=self.nominal_end,
            pair_end=self.pair_end,
        )

    def _apply(self):
        env = self.training_env.envs[0].unwrapped
        nominal_scale, pair_blend = self._current_values()
        env.set_curriculum(
            nominal_action_scale=nominal_scale,
            pair_lock_blend=pair_blend,
        )
        self.logger.record("curriculum/nominal_action_scale", nominal_scale)
        self.logger.record("curriculum/pair_lock_blend", pair_blend)

    def _on_training_start(self):
        self._apply()

    def _on_step(self):
        self._apply()
        return True


class BestWalkCallback(BaseCallback):
    """Save the best checkpoint judged by forward progress from reset."""

    def __init__(
        self,
        save_path,
        model_path=DEFAULT_MODEL_PATH_STR,
        eval_freq=20_000,
        n_eval_episodes=3,
        max_steps=1000,
    ):
        super().__init__()
        self.save_path = Path(save_path)
        self.model_path = model_path
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = max(int(n_eval_episodes), 1)
        self.max_steps = max(int(max_steps), 1)
        self.best_score = -float("inf")
        self.eval_env = None

    def _on_training_start(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.eval_env = TARSEnv(self.model_path)

    def _evaluate(self):
        train_env = self.training_env.envs[0].unwrapped
        self.eval_env.set_curriculum(
            nominal_action_scale=train_env.nominal_action_scale,
            pair_lock_blend=train_env.pair_lock_blend,
        )

        forward_progresses = []
        survivals = []
        lateral_drifts = []
        tilts = []
        contact_sums = np.zeros(4, dtype=np.float64)
        phase_contact_sums = {
            0: np.zeros(4, dtype=np.float64),
            1: np.zeros(4, dtype=np.float64),
        }
        phase_contact_counts = {0: 0, 1: 0}
        contact_samples = 0

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            initial_x = float(self.eval_env.data.qpos[0])
            for step in range(self.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                contacts = np.asarray(self.eval_env.last_foot_contacts, dtype=np.float64)
                scored_phase = int(self.eval_env.last_scored_phase)
                contact_sums += contacts
                phase_contact_sums[scored_phase] += contacts
                phase_contact_counts[scored_phase] += 1
                contact_samples += 1
                if terminated or truncated:
                    break

            forward_progresses.append(float(self.eval_env.data.qpos[0]) - initial_x)
            survivals.append((step + 1) / self.max_steps)
            lateral_drifts.append(abs(float(self.eval_env.data.qpos[1])))
            quat = self.eval_env.data.body("internals").xquat
            tilts.append(float(2 * np.arccos(np.clip(abs(quat[0]), 0.0, 1.0))))

        mean_x = float(np.mean(forward_progresses))
        mean_survival = float(np.mean(survivals))
        mean_lateral = float(np.mean(lateral_drifts))
        mean_tilt = float(np.mean(tilts))
        foot_contact_rates = contact_sums / max(contact_samples, 1)
        phase0_contact_rates = phase_contact_sums[0] / max(phase_contact_counts[0], 1)
        phase1_contact_rates = phase_contact_sums[1] / max(phase_contact_counts[1], 1)
        score = mean_x
        return (
            score,
            mean_x,
            mean_survival,
            mean_lateral,
            mean_tilt,
            foot_contact_rates,
            phase0_contact_rates,
            phase1_contact_rates,
        )

    def _on_step(self):
        if self.num_timesteps % self.eval_freq != 0:
            return True

        (
            score,
            mean_x,
            mean_survival,
            mean_lateral,
            mean_tilt,
            foot_contact_rates,
            phase0_contact_rates,
            phase1_contact_rates,
        ) = self._evaluate()
        self.logger.record("eval_walk/score", score)
        self.logger.record("eval_walk/mean_x", mean_x)
        self.logger.record("eval_walk/mean_survival", mean_survival)
        self.logger.record("eval_walk/mean_abs_y", mean_lateral)
        self.logger.record("eval_walk/mean_tilt", mean_tilt)
        for leg_id in range(4):
            self.logger.record(f"eval_walk/foot_{leg_id}_contact_rate", float(foot_contact_rates[leg_id]))
            self.logger.record(f"eval_walk/phase0_foot_{leg_id}_contact_rate", float(phase0_contact_rates[leg_id]))
            self.logger.record(f"eval_walk/phase1_foot_{leg_id}_contact_rate", float(phase1_contact_rates[leg_id]))

        print(
            f"  Eval @ {self.num_timesteps:>6d} | "
            f"score={score:+.3f} x={mean_x:+.3f} "
            f"survival={mean_survival:.2f} |y|={mean_lateral:.3f} tilt={mean_tilt:.3f} "
            f"contacts={np.round(foot_contact_rates, 2).tolist()} "
            f"phase0={np.round(phase0_contact_rates, 2).tolist()} "
            f"phase1={np.round(phase1_contact_rates, 2).tolist()}",
            flush=True,
        )

        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.save_path))
            print(f"  Saved best walker to {self.save_path}.zip", flush=True)

        return True
