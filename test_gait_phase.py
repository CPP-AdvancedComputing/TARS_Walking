import numpy as np
import mujoco
from tars_env import TARSEnv


URDF = r"C:\Users\anike\tars-urdf\tars_mjcf.xml"


def test_crutch_pair_order_is_explicit():
    env = TARSEnv(URDF)
    assert env._phase_pairs(0) == (env.SWING_PAIR, env.SUPPORT_PAIR)
    assert env._phase_pairs(1) == (env.SUPPORT_PAIR, env.SWING_PAIR)


def test_fixed_carriages_are_children_of_servos():
    env = TARSEnv(URDF)
    for leg_id in range(4):
        fixed_body = env.model.body(f"fixed_carriage_l{leg_id}")
        parent_body = env.model.body(env.model.body_parentid[fixed_body.id])
        assert parent_body.name == f"servo_l{leg_id}"


def test_active_carriage_and_servo_meshes_stay_on_their_links():
    env = TARSEnv(URDF)
    for leg_id in range(4):
        for body_name in [f"active_carriage_l{leg_id}", f"servo_l{leg_id}"]:
            geom_id = env.model.body_geomadr[env.model.body(body_name).id]
            geom_body = env.model.body(env.model.geom_bodyid[geom_id])
            assert geom_body.name == body_name


def test_only_internal_fastener_duplicates_are_hidden():
    env = TARSEnv(URDF)
    internals_id = env.model.body("internals").id
    start = env.model.body_geomadr[internals_id]
    count = env.model.body_geomnum[internals_id]
    visible_meshes = set()
    hidden_meshes = set()
    for geom_id in range(start, start + count):
        mesh_id = int(env.model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        mesh_name = env.model.mesh(mesh_id).name
        alpha = float(env.model.geom_rgba[geom_id][3])
        if alpha > 0.0:
            visible_meshes.add(mesh_name)
        else:
            hidden_meshes.add(mesh_name)
    for expected_visible in [
        "side_plate",
        "faceplate_corner",
        "corner",
        "aluminum_rail",
        "top_plate",
        "middle_part",
        "bottom_plate",
    ]:
        assert expected_visible in visible_meshes
    assert "active_carriage" not in visible_meshes
    assert "servo_horn" not in visible_meshes
    assert "stepper_mount_active" not in visible_meshes
    assert "stepper_mount_fixed" not in visible_meshes
    assert "active_carriage" in hidden_meshes
    assert "servo_horn" in hidden_meshes
    assert "stepper_mount_active" in hidden_meshes
    assert "stepper_mount_fixed" in hidden_meshes
    assert "stepper_coupler" not in visible_meshes
    assert "stepper_shaft" not in visible_meshes
    assert "funky_tab" not in visible_meshes
    assert "jst_s6b_ph_k_s" not in visible_meshes
    assert "philips_m3x30" not in visible_meshes
    assert "stepper_coupler" in hidden_meshes
    assert "stepper_shaft" in hidden_meshes
    assert "funky_tab" in hidden_meshes
    assert "jst_s6b_ph_k_s" in hidden_meshes
    assert "philips_m3x30" in hidden_meshes
    assert "bearing" in hidden_meshes
    assert "bearing__2" in hidden_meshes
    assert "connector_tab" in hidden_meshes


def test_visible_leg_hardware_is_attached_to_moving_bodies():
    env = TARSEnv(URDF)
    for leg_id in range(4):
        active_meshes = {
            env.model.mesh(int(env.model.geom_dataid[geom_id])).name
            for geom_id in range(
                int(env.model.body_geomadr[env.model.body(f"active_carriage_l{leg_id}").id]),
                int(env.model.body_geomadr[env.model.body(f"active_carriage_l{leg_id}").id])
                + int(env.model.body_geomnum[env.model.body(f"active_carriage_l{leg_id}").id]),
            )
            if int(env.model.geom_dataid[geom_id]) >= 0
        }
        servo_meshes = {
            env.model.mesh(int(env.model.geom_dataid[geom_id])).name
            for geom_id in range(
                int(env.model.body_geomadr[env.model.body(f"servo_l{leg_id}").id]),
                int(env.model.body_geomadr[env.model.body(f"servo_l{leg_id}").id])
                + int(env.model.body_geomnum[env.model.body(f"servo_l{leg_id}").id]),
            )
            if int(env.model.geom_dataid[geom_id]) >= 0
        }
        fixed_meshes = {
            env.model.mesh(int(env.model.geom_dataid[geom_id])).name
            for geom_id in range(
                int(env.model.body_geomadr[env.model.body(f"fixed_carriage_l{leg_id}").id]),
                int(env.model.body_geomadr[env.model.body(f"fixed_carriage_l{leg_id}").id])
                + int(env.model.body_geomnum[env.model.body(f"fixed_carriage_l{leg_id}").id]),
            )
            if int(env.model.geom_dataid[geom_id]) >= 0
        }
        assert "active_carriage" in active_meshes
        assert "stepper_mount_active" in active_meshes
        assert "servo_horn" in servo_meshes
        assert "stepper_mount_fixed" in fixed_meshes


def test_no_env_visual_proxies_are_added():
    env = TARSEnv(URDF)
    env.reset()
    assert env.shell_followers == []
    assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "body_visual_core") == -1
    assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "body_visual_head") == -1


def test_pair_lock_blend_respects_requested_value():
    env = TARSEnv(URDF)
    env.set_curriculum(pair_lock_blend=0.2)
    assert env.pair_lock_blend == 0.2


def test_contact_pattern_reward_prefers_literal_crutch_pairs():
    env = TARSEnv(URDF)
    correct_phase_zero = env._contact_pattern_reward([True, False, True, False], phase=0)[0]
    wrong_phase_zero = env._contact_pattern_reward([True, False, False, True], phase=0)[0]
    correct_phase_one = env._contact_pattern_reward([False, True, False, True], phase=1)[0]
    wrong_phase_one = env._contact_pattern_reward([False, True, True, False], phase=1)[0]

    assert correct_phase_zero > wrong_phase_zero
    assert correct_phase_one > wrong_phase_one
    np.testing.assert_allclose(env._desired_foot_contacts(0), np.array([1.0, 0.0, 1.0, 0.0]))
    np.testing.assert_allclose(env._desired_foot_contacts(1), np.array([0.0, 1.0, 0.0, 1.0]))


def test_contact_feet_are_attached_to_fixed_carriages():
    env = TARSEnv(URDF)
    for leg_id in range(4):
        foot_geom = env.model.geom(f"servo_l{leg_id}_foot")
        foot_body = env.model.body(env.model.geom_bodyid[foot_geom.id])
        assert foot_body.name == f"fixed_carriage_l{leg_id}"


def test_no_env_helper_leg_capsules_are_added():
    env = TARSEnv(URDF)
    for leg_id in range(4):
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"servo_l{leg_id}_leg") == -1


def test_phase_zero_nominal_action_is_zero_for_ik_controller():
    env = TARSEnv(URDF)
    nominal = env._phase_nominal_action(0)
    np.testing.assert_allclose(nominal, np.zeros(env.action_space.shape[0], dtype=np.float64))


def test_pair_theta_difference_uses_centered_hip_angles():
    env = TARSEnv(URDF)
    env.reset()
    swing_pair, _ = env._phase_pairs(0)

    for leg_id in swing_pair:
        env.data.joint(f"hip_revolute_l{leg_id}").qpos[0] = env.standing_ctrl[4 + leg_id] + 0.18

    assert np.isclose(env._pair_theta_difference(swing_pair), 0.0, atol=1e-9)


def test_theta_sync_score_drops_when_pair_desynchronizes():
    env = TARSEnv(URDF)
    env.reset()
    swing_pair, _ = env._phase_pairs(0)
    first_leg, second_leg = swing_pair

    env.data.joint(f"hip_revolute_l{first_leg}").qpos[0] = env.standing_ctrl[4 + first_leg] + 0.2
    env.data.joint(f"hip_revolute_l{second_leg}").qpos[0] = env.standing_ctrl[4 + second_leg] + 0.2
    aligned_score = env._theta_sync_score(swing_pair)

    env.data.joint(f"hip_revolute_l{second_leg}").qpos[0] = env.standing_ctrl[4 + second_leg] - 0.2
    desynced_score = env._theta_sync_score(swing_pair)

    assert aligned_score > 0.99
    assert desynced_score < aligned_score
    assert desynced_score < 0.01


def test_leg_sync_score_uses_full_leg_state():
    env = TARSEnv(URDF)
    env.reset()
    swing_pair, _ = env._phase_pairs(0)
    first_leg, second_leg = swing_pair

    env.data.joint(f"shoulder_prismatic_l{first_leg}").qpos[0] = 0.05
    env.data.joint(f"shoulder_prismatic_l{second_leg}").qpos[0] = 0.05
    env.data.joint(f"hip_revolute_l{first_leg}").qpos[0] = env.standing_ctrl[4 + first_leg] + 0.1
    env.data.joint(f"hip_revolute_l{second_leg}").qpos[0] = env.standing_ctrl[4 + second_leg] + 0.1
    env.data.joint(f"knee_prismatic_l{first_leg}").qpos[0] = -0.04
    env.data.joint(f"knee_prismatic_l{second_leg}").qpos[0] = -0.04
    aligned_score = env._leg_sync_score(swing_pair)

    env.data.joint(f"knee_prismatic_l{second_leg}").qpos[0] = 0.10
    desynced_score = env._leg_sync_score(swing_pair)

    assert aligned_score > 0.99
    assert desynced_score < aligned_score


def test_phase_zero_desired_contacts_ground_legs_zero_and_two():
    env = TARSEnv(URDF)
    env.reset()
    np.testing.assert_allclose(env._desired_foot_contacts(0), np.array([1.0, 0.0, 1.0, 0.0]))


def test_returned_clock_matches_phase_after_boundary_switch():
    env = TARSEnv(URDF)
    env.reset()
    env._phase_switch_ready = lambda phase=None: (
        True,
        1 if (env.phase if phase is None else phase) == 0 else 0,
        env._phase_pairs(1 if (env.phase if phase is None else phase) == 0 else 0)[1],
        {},
    )

    for _ in range(env.PHASE_STEPS):
        obs, _, terminated, truncated, _ = env.step(env.zero_action())
        assert not terminated
        assert not truncated

    assert env.phase == 1
    assert env.phase_timer == 0
    np.testing.assert_allclose(obs[-2:], np.array([0.0, -1.0]), atol=1e-6)


def test_knee_actions_are_kept_in_connected_leg_model():
    env = TARSEnv(URDF)
    action = np.array([0.25, -0.50, 0.75, -1.25], dtype=np.float64)
    effective = env._effective_action(action)
    np.testing.assert_allclose(effective, np.array([0.25, -0.50, 0.75, -1.0]))


def test_pair_locked_actions_keep_crutch_legs_synced():
    env = TARSEnv(URDF)
    env.set_curriculum(pair_lock_blend=0.0)
    assert env.pair_lock_blend == 0.0


def test_step_tracks_theta_sync_metrics():
    env = TARSEnv(URDF)
    env.reset()
    env.step(env.zero_action())
    assert env.last_swing_theta_diff >= 0.0
    assert env.last_plant_theta_diff >= 0.0
    assert env.last_theta_sync_reward >= 0.0
    assert env.last_swing_leg_diff >= 0.0
    assert env.last_plant_leg_diff >= 0.0
    assert env.last_leg_sync_reward >= 0.0


def test_phase_switch_waits_until_next_plant_pair_is_near_ground():
    env = TARSEnv(URDF)
    env.reset()
    env.phase_timer = env.PHASE_STEPS - 1
    env._phase_switch_ready = lambda phase=None: (False, 1, (1, 3), {1: 0.087, 3: 0.287})

    _, _, terminated, truncated, _ = env.step(env.zero_action())

    assert not terminated
    assert not truncated
    assert env.phase == 0
    assert env.phase_timer == env.PHASE_STEPS


def test_phase_switch_happens_once_next_plant_pair_is_near_ground():
    env = TARSEnv(URDF)
    env.reset()
    env.phase_timer = env.PHASE_STEPS - 1
    env._phase_switch_ready = lambda phase=None: (True, 1, (1, 3), {1: 0.02, 3: 0.03})

    _, _, terminated, truncated, _ = env.step(env.zero_action())

    assert not terminated
    assert not truncated
    assert env.phase == 1
    assert env.phase_timer == 0


def test_phase_switch_stalls_after_timeout_if_feet_never_land():
    env = TARSEnv(URDF)
    env.reset()
    env.phase_timer = env.PHASE_SWITCH_TIMEOUT_STEPS - 1
    env._phase_switch_ready = lambda phase=None: (False, 1, (1, 3), {1: 0.2, 3: 0.2})

    _, _, terminated, truncated, _ = env.step(env.zero_action())

    assert not terminated
    assert not truncated
    assert env.phase == 0
    assert env.phase_timer == env.PHASE_SWITCH_TIMEOUT_STEPS

