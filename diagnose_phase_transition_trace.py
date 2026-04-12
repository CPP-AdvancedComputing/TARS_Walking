import argparse

import mujoco

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR


def parse_args():
    parser = argparse.ArgumentParser(description="Trace a single gait transition under zero action.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH_STR)
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    return parser.parse_args()


def fmt_list(values):
    return "[" + ", ".join(f"{float(v):+.3f}" for v in values) + "]"


def leg_rotation_metrics(env):
    centered_hips = env._centered_hip_thetas()
    track_x = [env._track_vector_body(i)[0] for i in range(4)]
    return {
        "centered_hips": [float(v) for v in centered_hips],
        "track_x": [float(v) for v in track_x],
        "outer_forward_mean": float(0.5 * (track_x[0] + track_x[3])),
        "middle_forward_mean": float(0.5 * (track_x[1] + track_x[2])),
    }


def edge_contact_proxy(env, leg_id):
    quality = float(env.last_foot_planted_qualities[leg_id]) if env.last_foot_planted_qualities is not None else 0.0
    height = float(env._foot_world_height(leg_id))
    foot_on_ground = int(env._foot_on_ground_flags()[leg_id])
    return {
        "height": height,
        "quality": quality,
        "raw_contact": foot_on_ground,
        "edge_touch": int(foot_on_ground and quality < env.FULL_PLANT_MIN_QUALITY and height <= env.PHASE_SWITCH_GROUND_Z),
    }


def main():
    args = parse_args()
    env = TARSEnv(model_path=args.model_path)

    mujoco.mj_resetData(env.model, env.data)
    env._set_phase_pose(args.phase)
    env._initialize_rod_targets()
    env._begin_phase(args.phase)
    env.data.ctrl[:] = env.control_targets_for_action(env.zero_action(), phase=args.phase)
    mujoco.mj_forward(env.model, env.data)
    env.initial_height = float(env.data.qpos[2])
    env.initial_body_height = float(env.data.body("internals").xpos[2])
    env.prev_body_z = float(env.data.qpos[2])
    env.prev_body_vz = float(env.data.qvel[2])
    env.prev_x = float(env.data.qpos[0])
    env.stagnation_x = float(env.data.qpos[0])
    env.stagnation_timer = 0

    roles = env._phase_transition_roles(args.phase)
    print(f"phase={args.phase} next_phase={env._next_phase(args.phase)}")
    print(f"support_ids={roles['support_ids']} touchdown_ids={roles['touchdown_ids']} liftoff_ids={roles['liftoff_ids']}")

    for step in range(args.steps + 1):
        targets = env._desired_leg_role_targets(env.zero_action(), phase=args.phase)
        contacts_raw = env._foot_on_ground_flags()
        contacts = [int(v) for v in contacts_raw]
        planted_flags, planted_qualities = env._foot_fully_planted_flags(foot_on_ground=contacts_raw)
        planted = [int(v) for v in planted_flags]
        heights = [env._foot_world_height(i) for i in range(4)]
        x_positions = [env._foot_track_world(i)[0] for i in range(4)]
        target_z = [targets[i]["foot_target_world"][2] for i in range(4)]
        target_x = [targets[i]["foot_target_world"][0] for i in range(4)]
        motion_roles = [targets[i]["motion_role"] for i in range(4)]
        rotation = leg_rotation_metrics(env)
        middle_edge = {leg_id: edge_contact_proxy(env, leg_id) for leg_id in (1, 2)}
        ready, next_phase, _, _ = env._phase_switch_ready(args.phase)
        print(
            f"step={step:03d} timer={env.phase_timer:03d} "
            f"contacts={contacts} planted={planted} qualities={fmt_list(planted_qualities)} heights={fmt_list(heights)} "
            f"x={fmt_list(x_positions)} target_x={fmt_list(target_x)} "
            f"target_z={fmt_list(target_z)} roles={motion_roles} "
            f"hips={fmt_list(rotation['centered_hips'])} track_body_x={fmt_list(rotation['track_x'])} "
            f"outer_fwd={rotation['outer_forward_mean']:+.3f} middle_fwd={rotation['middle_forward_mean']:+.3f} "
            f"mid_edge={[middle_edge[1]['edge_touch'], middle_edge[2]['edge_touch']]} "
            f"ready={int(ready)} next={next_phase} "
            f"next_contact={env.last_next_plant_contact_fraction:.2f} "
            f"next_low={env.last_next_plant_low_fraction:.2f} "
            f"swing_clear={env.last_switch_swing_clear_fraction:.2f} "
            f"hist_contact={env.last_history_next_plant_contact_fraction:.2f} "
            f"hist_low={env.last_history_next_plant_low_fraction:.2f} "
            f"hist_clear={env.last_history_swing_clear_fraction:.2f} "
            f"block={int(env.last_phase_switch_block_code)} "
            f"force={int(env.last_phase_switch_force_armed)}"
        )
        if step >= args.steps:
            break
        env.step(env.zero_action())


if __name__ == "__main__":
    main()
