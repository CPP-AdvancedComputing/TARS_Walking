import os
from collections import deque

import gymnasium
import mujoco
import numpy as np

from tars_model import load_tars_spec, resolve_model_path


def quat_conjugate(quat):
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_rotate_vector(quat, vector):
    pure = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float64)
    rotated = quat_multiply(quat_multiply(quat, pure), quat_conjugate(quat))
    return rotated[1:]


def vector_from_env(var_name, default):
    raw = os.environ.get(var_name)
    if not raw:
        return tuple(default)
    try:
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip() != "")
    except ValueError:
        return tuple(default)
    if len(values) != len(default):
        return tuple(default)
    return values


DEFAULT_RESET_PLANT_POSE = vector_from_env("TARS_RESET_PLANT_POSE", (-0.15, 0.0, -0.10))
DEFAULT_RESET_SWING_POSE = vector_from_env("TARS_RESET_SWING_POSE", (-0.10, 0.0, -0.02))


class TARSEnv(gymnasium.Env):

    FOOT_CONTACT_GEOMETRY = {
        "fixed_carriage_l0": {
            "pos": [0.0, -0.0165366, -0.6286930],
            "size": [0.040, 0.040, 0.0100],
        },
        "fixed_carriage_l1": {
            "pos": [0.0, -0.0165370, -0.5208440],
            "size": [0.040, 0.040, 0.0100],
        },
        "fixed_carriage_l2": {
            "pos": [0.0226945, -0.0165370, -0.8158440],
            "size": [0.040, 0.040, 0.0100],
        },
        "fixed_carriage_l3": {
            "pos": [0.0214305, -0.0145360, -0.5836930],
            "size": [0.040, 0.040, 0.0100],
        },
    }
    UPPER_ROD_ANCHORS = {
        0: [-0.0476250, 0.0637134, 0.6809500],
        1: [-0.0476250, 0.3660920, 0.5538920],
        2: [-0.0812508, 0.2672990, 0.6585880],
        3: [-0.0798627, 0.1645060, 0.6565950],
    }
    LOWER_ROD_ANCHORS = {
        "fixed_carriage_l0": [-0.00834185, 0.01834358, -0.15974110],
        "fixed_carriage_l1": [-0.02888102, 0.01834373, -0.06607028],
        "fixed_carriage_l2": [-0.02996766, 0.01834358, -0.19503245],
        "fixed_carriage_l3": [0.05385041, -0.01834390, -0.15235845],
    }
    INTERNALS_HIDDEN_MESH_PREFIXES = (
        "bearing",
        "connector_tab",
    )
    INTERNALS_STATIC_LEG_MESHES = frozenset({
        "active_carriage",
    })
    INTERNALS_REATTACH_MESH_TARGETS = {
        "stepper_mount_active": "active_carriage_l",
        "servo_horn": "servo_l",
        "stepper_mount_fixed": "fixed_carriage_l",
    }
    INTERNALS_NEAREST_LEG_HARDWARE_MESHES = frozenset({
        "stepper_coupler",
        "stepper_shaft",
        "funky_tab",
        "jst_s6b_ph_k_s",
        "philips_m3x30",
    })

    STANDING_HIP_ANGLE = 0.2
    FRAME_SKIP = 5         # 5 steps x 0.002s = 10ms per action
    PHASE_STEPS = 30       # 0.3s per phase at 10ms/action, 0.9s full cycle
    MAX_EPISODE_STEPS = max(int(os.environ.get("TARS_MAX_EPISODE_STEPS", "2000")), 1)
    REWARD_PROFILE = os.environ.get("TARS_REWARD_PROFILE", "foundation").strip().lower()
    PHASE_COUNT = 3
    PHASE_CONTACT_MASKS = {
        0: np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64),
        1: np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        2: np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
    }
    HIP_STANDING_SIGNS = (-1, 1, -1, 1)
    ACTION_DIM = 8
    ACTIONS_PER_LEG = 2
    ROD_ANGLE_ACTION_SCALE = np.deg2rad(5.0)
    ROD_LENGTH_ACTION_SCALE = 0.025
    ROD_LENGTH_MARGIN = 0.02
    ROD_LENGTH_MIN = 0.44
    ROD_LENGTH_MAX = 0.78
    FOOT_HALF_HEIGHT = 0.01
    FOOT_SPAWN_CLEARANCE = 0.001
    PLANT_TARGET_PRELOAD_DEPTH = 0.006
    PLANT_TARGET_EXTRA_PRELOAD_BY_LEG = {
        0: 0.004,
        1: 0.0,
        2: 0.0,
        3: 0.0,
    }
    PHASE_CTRL_BIAS = {
        0: np.array([
            -0.10, 0.05, 0.0, -0.10,
            -0.50, -0.40, -0.75, -0.50,
            -0.10, 0.10, 0.0, 0.05,
        ], dtype=np.float64),
        1: np.array([
            -0.05, -0.10, 0.0, 0.10,
            -0.50, -0.50, 0.25, -0.50,
            0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float64),
        2: np.array([
            0.0, -0.35, 0.20, 0.0,
            0.0, -0.75, 0.0, 0.0,
            0.0, -0.10, -0.30, 0.0,
        ], dtype=np.float64),
    }
    PHASE_PAIR_CTRL_OFFSETS = {
        0: {
            (0, 3): np.array([0.0, 0.25, -0.05], dtype=np.float64),
            (1, 2): np.array([-0.10, -0.25, -0.10], dtype=np.float64),
        },
        2: {
            # Phase 2 should present the outer pair as the perpendicular return
            # receivers while the middle pair stays rotated outward before the
            # lean-forward handoff back into phase 0.
            (0, 3): np.array([0.0, 0.15, 0.0], dtype=np.float64),
            (1, 2): np.array([0.05, -0.10, 0.0], dtype=np.float64),
        },
    }
    IK_DAMPING = 5e-4
    IK_STEP_SCALE = 0.55
    IK_MAX_SHOULDER_DELTA = 0.008
    IK_MAX_HIP_DELTA = 0.08
    IK_MAX_KNEE_DELTA = 0.008
    IK_SWING_STEP_SCALE = 0.8
    IK_SWING_SHOULDER_SCALE = 1.0
    IK_SWING_HIP_SCALE = 1.75
    IK_SWING_KNEE_SCALE = 1.0
    IK_TOUCHDOWN_STEP_SCALE = 1.6
    IK_TOUCHDOWN_SHOULDER_SCALE = 2.0
    IK_TOUCHDOWN_HIP_SCALE = 2.5
    IK_TOUCHDOWN_KNEE_SCALE = 2.0
    SUPPORT_REFERENCE_CTRL_BLEND = 1.0
    PRETOUCHDOWN_REFERENCE_CTRL_BLEND = 1.0
    PRETOUCHDOWN_REFERENCE_CTRL_BLEND_BY_PHASE = {
        2: 0.0,
    }
    PRELIFTOFF_REFERENCE_CTRL_BLEND = 1.0
    PRELIFTOFF_REFERENCE_CTRL_BLEND_BY_PHASE = {
        2: 0.0,
    }
    ENFORCE_TRIPEDAL_PAIR_LOCK = True
    TRIPEDAL_PAIR_STATE_FEEDBACK_GAIN = 0.75
    TRIPEDAL_PAIR_STATE_FEEDBACK_GAIN_BY_PAIR = {
        (0, 3): 0.75,
        (1, 2): 1.50,
    }
    THETA_SYNC_SIGMA = 0.10
    THETA_TARGET_SIGMA = 0.20
    LEG_SYNC_SIGMA = 0.18
    ROD_SYNC_SIGMA = 0.20
    ROD_TARGET_SIGMA = 0.24
    ROD_LENGTH_SIGMA = 0.04
    SWING_FOOT_SYNC_SIGMA = 0.08
    SWING_FOOT_CLEARANCE = 0.03
    SWING_LIFT_REWARD_SCALE = 8.0
    SWING_LIFT_TARGET_HEIGHT = 0.04
    SWING_PIVOT_ALIGNMENT_SIGMA = 0.05
    SWING_LATERAL_ALIGNMENT_SIGMA = 0.02
    SWING_EARLY_LIFT_CLEARANCE = 0.055
    SWING_LATE_LANDING_RATIO = 0.65
    SWING_FORWARD_PROGRESS_RATIO = 0.60
    TOUCHDOWN_START_PROGRESS_RATIO = 0.80
    PHASE_LIFTOFF_START_PROGRESS = {
        0: 0.60,
        1: 0.25,
        2: 0.50,
    }
    PHASE_TOUCHDOWN_START_PROGRESS = {
        0: 0.45,
        1: 0.80,
        2: 0.35,
    }
    PHASE_TOUCHDOWN_START_PROGRESS_BY_LEG = {
        0: {
            1: 0.65,
            2: 0.00,
        },
        2: {
            0: 0.35,
            3: 0.35,
        },
    }
    PHASE_TOUCHDOWN_CTRL_OFFSETS = {
        0: {
            2: np.array([-0.20, -0.25, -0.05], dtype=np.float64),
        },
        2: {
            0: np.array([0.0, 0.20, 0.05], dtype=np.float64),
            3: np.array([0.0, 0.20, 0.05], dtype=np.float64),
        },
    }
    SWING_TRACK_LIFT_HEIGHT = 0.003
    TOUCHDOWN_TARGET_MAX_X_STEP = 0.0200
    TOUCHDOWN_TARGET_MIN_DESCENT_STEP = 0.0100
    SWING_TARGET_MAX_Z_STEP = 0.0020
    TOUCHDOWN_TARGET_MAX_Z_STEP = 0.0200
    EARLY_SWING_TOUCHDOWN_WINDOW = 0.40
    EARLY_SWING_MIN_FOOT_CLEARANCE = 0.015
    EARLY_SWING_TOUCHDOWN_PENALTY_SCALE = 8.0
    PHASE_SWITCH_GROUND_Z = 0.03
    PHASE_SWITCH_TIMEOUT_STEPS = 60
    PHASE_TRANSITION_REWARD_SCALE = 6.0
    PHASE_SWITCH_MIN_PROGRESS = 0.75
    PHASE_SWITCH_MIN_PROGRESS_BY_PHASE = {
        0: 0.10,
    }
    PHASE_SWITCH_MIN_NEXT_PLANT_CONTACT_FRACTION = 0.5
    PHASE_SWITCH_MIN_NEXT_PLANT_LOW_FRACTION = 1.0
    PHASE_SWITCH_MIN_SWING_CLEAR_FRACTION = 0.25
    PHASE_SWITCH_MIN_SWING_CLEAR_FRACTION_BY_PHASE = {
        1: 1.0,
        2: 1.0,
    }
    PHASE_SWITCH_ALLOW_INSTANT_FULL_READY_BY_PHASE = {
        1: True,
        2: True,
    }
    PHASE_SWITCH_HYSTERESIS_STEPS = 3
    PHASE_SWITCH_HISTORY_STEPS = 4
    PHASE_SWITCH_FORCE_ADVANCE_DELAY_STEPS = 8
    PHASE_TARGET_CAPTURE_DELAY_STEPS = 8
    PLANT_CONTACT_STABILITY_STEPS = 3
    FULL_PLANT_MIN_QUALITY = 0.70
    FULL_PLANT_MIN_CONTACTS = 1
    FULL_PLANT_MIN_FLATNESS = 0.50
    FULL_PLANT_MAX_HEIGHT = 0.022
    RESET_PLANT_POSE = DEFAULT_RESET_PLANT_POSE
    RESET_SWING_POSE = DEFAULT_RESET_SWING_POSE
    REFERENCE_SHOULDER_AMPLITUDE = 0.0
    REFERENCE_HIP_AMPLITUDE = 0.45
    REFERENCE_KNEE_AMPLITUDE = 0.0
    SWING_THETA_SYNC_WEIGHT = 0.8
    PLANT_THETA_SYNC_WEIGHT = 0.4
    SWING_THETA_TARGET_WEIGHT = 0.8
    PLANT_THETA_TARGET_WEIGHT = 0.8
    SWING_LEG_SYNC_WEIGHT = 0.6
    PLANT_LEG_SYNC_WEIGHT = 0.3
    SWING_ROD_SYNC_WEIGHT = 0.45
    PLANT_ROD_SYNC_WEIGHT = 0.55
    SWING_ROD_TARGET_WEIGHT = 0.55
    PLANT_ROD_TARGET_WEIGHT = 0.75
    SWING_FOOT_FORWARD_WEIGHT = 0.7
    SWING_FOOT_PAIR_SYNC_WEIGHT = 0.35
    SWING_VERTICAL_CHECKPOINT_REWARD_SCALE = 1.6
    SWING_LATERAL_ALIGNMENT_REWARD_SCALE = 0.8
    SWING_LATERAL_DRIFT_PENALTY_SCALE = 0.35
    PHASE_CONTACT_PENALTY_SCALE = 1.8
    FORWARD_LEAN_GATE_SIGMA = 0.07
    FORWARD_LEAN_PENALTY_SCALE = 3.0
    RECTANGLE_WINDOW_CENTER = 0.82
    RECTANGLE_WINDOW_SIGMA = 0.18
    RECTANGLE_VERTICAL_REWARD_SCALE = 2.2
    RECTANGLE_CONTACT_REWARD_SCALE = 1.2
    SWING_PAIR_LOCK_START = 0.72
    PHASE_STALL_PENALTY = 12.0
    SWING_REACH_RESET_REWARD_SCALE = 1.4
    REACH_RESET_X_SIGMA = 0.05
    REACH_RESET_Z_SIGMA = 0.08
    PROGRESS_REWARD_SCALE = 20.0
    VELOCITY_BONUS_SCALE = 0.75
    SHAPING_REWARD_SCALE = 0.25
    VERTICAL_OSCILLATION_STEP_PENALTY_SCALE = 4.0
    VERTICAL_OSCILLATION_ACCEL_PENALTY_SCALE = 1.2
    VERTICAL_OSCILLATION_HEIGHT_PENALTY_SCALE = 12.0
    VERTICAL_STABILITY_DISPLACEMENT_SIGMA = 0.080
    VERTICAL_STABILITY_VELOCITY_SIGMA = 1.50
    SHOULDER_PAIR_FEEDBACK_GAIN = 0.45
    HIP_PAIR_FEEDBACK_GAIN = 0.60
    KNEE_PAIR_FEEDBACK_GAIN = 0.45
    CONTACT_MATCH_REWARD_SCALE = 0.65
    CONTACT_PAIR_BONUS = 1.2
    FORCED_PAIR_LOCK_BLEND = 0.35
    GAIT_REFERENCE_HIP_SIGMA = 0.20
    GAIT_REFERENCE_LEG_SIGMA = 0.08
    GAIT_REFERENCE_HIP_REWARD_SCALE = 8.0
    GAIT_REFERENCE_LEG_REWARD_SCALE = 4.0
    OSCILLATOR_OPPOSITION_SIGMA = 0.16
    OSCILLATOR_OPPOSITION_REWARD_SCALE = 1.8
    OSCILLATOR_OPPOSITION_PENALTY_SCALE = 0.6
    LUNGE_PROGRESS_SIGMA = 0.18
    FORWARD_LAUNCH_PENALTY_SCALE = 2.5
    PLANT_UNLOAD_PENALTY_SCALE = 1.0
    

    # Keep swing timing symmetric across paired legs so the learned gait
    # starts from a consistent alternating foundation instead of four
    # different touchdown schedules.
    LEG_SWING_TRACK_LIFT_HEIGHT = {
        0: 0.0030,
        1: 0.0030,
        2: 0.0030,
        3: 0.0060,
    }
    LEG_SWING_LATE_LANDING_RATIO = {
        0: 0.65,
        1: 0.65,
        2: 0.65,
        3: 0.75,
    }
    def __init__(self, model_path=None):
        resolved_model_path = resolve_model_path(model_path)
        spec = load_tars_spec(resolved_model_path)
        is_urdf_model = resolved_model_path.lower().endswith(".urdf")
        joint_names = [
            "shoulder_prismatic_l0", "shoulder_prismatic_l1",
            "shoulder_prismatic_l2", "shoulder_prismatic_l3",
            "hip_revolute_l0", "hip_revolute_l1",
            "hip_revolute_l2", "hip_revolute_l3",
            "knee_prismatic_l0", "knee_prismatic_l1",
            "knee_prismatic_l2", "knee_prismatic_l3",
        ]

        self.joint_names = joint_names
        self.leg_joint_names = {
            leg_id: (
                f"shoulder_prismatic_l{leg_id}",
                f"hip_revolute_l{leg_id}",
                f"knee_prismatic_l{leg_id}",
            )
            for leg_id in range(4)
        }
        self.joint_ranges = []
        if not list(spec.actuators):
            for name in joint_names:
                actuator = spec.add_actuator()
                actuator.name = name + "_act"
                actuator.target = name
                actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
                actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                actuator.forcelimited = True
                if "hip" in name:
                    kp, kd = 150.0, 15.0
                    actuator.forcerange = [-30.0, 30.0]
                else:
                    kp, kd = 140.0, 12.0
                    actuator.forcerange = [-50.0, 50.0]
                actuator.gainprm[0] = kp
                actuator.biasprm[1] = -kp
                actuator.biasprm[2] = -kd

        for joint in spec.joints:
            if joint.name in joint_names:
                if "hip" in joint.name:
                    joint.damping = 20.0
                else:
                    joint.damping = 8.0

        # Keep the raw MJCF solver settings and floor when present; add
        # fallbacks only for URDF-derived specs that don't define them.
        if is_urdf_model:
            spec.option.timestep = 0.002
            spec.option.iterations = 20

        existing_geom_names = {geom.name for geom in spec.geoms if geom.name}
        if "floor" not in existing_geom_names:
            floor = spec.worldbody.add_geom()
            floor.name = "floor"
            floor.type = mujoco.mjtGeom.mjGEOM_PLANE
            floor.size = [10, 10, 0.1]
            floor.rgba = [0.8, 0.8, 0.8, 0.3]
            floor.pos = [0, 0, 0]
            floor.contype = 1
            floor.conaffinity = 1
            floor.friction = [2.0, 0.05, 0.05]
            floor.solref = [-1000, -100]
            floor.solimp = [0.99, 0.99, 0.001, 0.5, 2.0]

        body_by_name = {body.name: body for body in spec.bodies}
        for leg_id in range(4):
            lower_body_name = f"fixed_carriage_l{leg_id}"
            foot_geom_cfg = self.FOOT_CONTACT_GEOMETRY.get(lower_body_name)
            foot_body = body_by_name.get(f"foot_l{leg_id}")
            if foot_body is not None:
                foot_parent = foot_body
                foot_pos = [0.0, 0.0, 0.0]
                track_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            else:
                foot_parent = body_by_name.get(lower_body_name)
                if foot_parent is None or foot_geom_cfg is None:
                    continue
                foot_pos = foot_geom_cfg["pos"]
                track_pos = np.array(self.LOWER_ROD_ANCHORS[lower_body_name], dtype=np.float64)

            foot = foot_parent.add_geom()
            foot.name = f"servo_l{leg_id}_foot"
            foot.type = mujoco.mjtGeom.mjGEOM_BOX
            foot.size = foot_geom_cfg["size"] if foot_geom_cfg is not None else [0.04, 0.04, 0.01]
            foot.pos = foot_pos
            foot.contype = 1
            foot.conaffinity = 1
            foot.friction = [2.0, 0.05, 0.05]
            foot.solref = [-1000, -100]
            foot.solimp = [0.99, 0.99, 0.001, 0.5, 2.0]
            foot.rgba = [0.0, 0.0, 0.0, 0.0]

            track_site = foot_parent.add_site()
            track_site.name = f"servo_l{leg_id}_track"
            track_site.pos = track_pos
            track_site.size = np.array([0.005, 0.005, 0.005], dtype=np.float64)
            track_site.rgba = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Add an invisible collision box for the main body so it cannot clip
        # through the floor when the robot tips over.
        for body in spec.bodies:
            if body.name == "internals":
                hull = body.add_geom()
                hull.name = "body_collision"
                hull.type = mujoco.mjtGeom.mjGEOM_BOX
                hull.size = [0.05, 0.15, 0.12]
                hull.pos = [0, 0.2, 0.42]
                hull.rgba = [0, 0, 0, 0]
                hull.contype = 1
                hull.conaffinity = 1
                hull.mass = 0
                break

        collision_whitelist = {"floor", "body_collision"} | {f"servo_l{i}_foot" for i in range(4)}
        for geom in spec.geoms:
            geom_name = geom.name or ""
            if geom_name in collision_whitelist:
                geom.contype = 1
                geom.conaffinity = 1
            else:
                geom.contype = 0
                geom.conaffinity = 0

        self._reattach_leg_visual_meshes(spec)
        self.model = spec.compile()
        self._hide_duplicate_internals_visuals()
        self.joint_ranges = [
            tuple(float(v) for v in self.model.jnt_range[self.model.joint(name).id])
            for name in self.joint_names
        ]
        self.joint_midpoints = np.array([
            (lo + hi) / 2.0
            for lo, hi in self.joint_ranges
        ], dtype=np.float64)
        self.leg_joint_qpos_indices = {
            leg_id: np.array([
                int(self.model.joint(name).qposadr[0])
                for name in self.leg_joint_names[leg_id]
            ], dtype=np.int32)
            for leg_id in range(4)
        }
        self.leg_joint_dof_indices = {
            leg_id: np.array([
                int(self.model.joint(name).dofadr[0])
                for name in self.leg_joint_names[leg_id]
            ], dtype=np.int32)
            for leg_id in range(4)
        }
        # Moderate angular damping stabilizes the body without locking it.
        # Free joint DOFs: [tx, ty, tz, rx, ry, rz] = indices 0-5.
        self.model.dof_damping[3] = 80.0
        self.model.dof_damping[4] = 80.0
        self.model.dof_damping[5] = 20.0
        self.model.dof_frictionloss[3] = 4.0
        self.model.dof_frictionloss[4] = 4.0
        self.model.dof_frictionloss[5] = 2.0

        # Cache geom IDs for fast contact detection.
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"servo_l{i}_foot")
            for i in range(4)
        ]
        self.foot_track_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"servo_l{i}_track")
            for i in range(4)
        ]
        self.body_collision_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "body_collision")

        self.data = mujoco.MjData(self.model)
        self.outer_torso_mocap_id = self._find_outer_torso_mocap_id()
        self.outer_torso_local_pos = None
        self.outer_torso_local_quat = None

        self.action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(self.ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32
        )
        self.step_count = 0
        self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float64)
        self.phase = 0
        self.phase_timer = 0
        self.shell_followers = []
        self.initial_height = None
        self.last_swing_theta_diff = 0.0
        self.last_plant_theta_diff = 0.0
        self.last_theta_sync_reward = 0.0
        self.last_swing_theta_target_error = 0.0
        self.last_plant_theta_target_error = 0.0
        self.last_theta_target_reward = 0.0
        self.last_swing_leg_diff = 0.0
        self.last_plant_leg_diff = 0.0
        self.last_leg_sync_reward = 0.0
        self.last_swing_rod_diff = 0.0
        self.last_plant_rod_diff = 0.0
        self.last_rod_sync_reward = 0.0
        self.last_swing_rod_target_error = 0.0
        self.last_plant_rod_target_error = 0.0
        self.last_swing_rod_length_error = 0.0
        self.last_plant_rod_length_error = 0.0
        self.last_rod_target_reward = 0.0
        self.last_swing_foot_forward_mean = 0.0
        self.last_swing_foot_forward_diff = 0.0
        self.last_swing_foot_forward_reward = 0.0
        self.last_swing_vertical_checkpoint_mean = 0.0
        self.last_swing_lateral_alignment_mean = 0.0
        self.last_swing_lateral_drift_penalty = 0.0
        self.last_contact_pattern_reward = 0.0
        self.last_contact_match_count = 0.0
        self.last_contact_pair_bonus = 0.0
        self.last_gait_reference_hip_reward = 0.0
        self.last_gait_reference_leg_reward = 0.0
        self.last_foot_contacts = np.zeros(4, dtype=np.float64)
        self.last_desired_contacts = np.zeros(4, dtype=np.float64)
        self.last_scored_phase = 0
        self.last_support_quality = 0.0
        self.last_progress_gate = 0.0
        self.last_body_hull_contact = 0.0
        self.last_phase_contact_quality = 0.0
        self.last_phase_contact_penalty = 0.0
        self.last_forward_lean_penalty = 0.0
        self.last_rectangle_checkpoint_reward = 0.0
        self.last_rectangle_vertical_mean = 0.0
        self.last_rectangle_contact_mean = 0.0
        self.last_swing_reach_reset_reward = 0.0
        self.last_oscillator_opposition_error = 0.0
        self.last_oscillator_opposition_reward = 0.0
        self.last_early_swing_clearance_quality = 0.0
        self.last_early_swing_touchdown_penalty = 0.0
        self.last_vertical_oscillation_penalty = 0.0
        self.last_vertical_stability_gate = 1.0
        self.last_swing_lift_reward = 0.0
        self.last_swing_plant_penalty = 0.0
        self.last_plant_unload_penalty = 0.0
        self.last_locomotion_validity_gate = 0.0
        self.last_phase_transition_reward = 0.0
        self.last_phase_timeout_terminated = 0.0
        self.last_next_plant_contact_fraction = 0.0
        self.last_next_plant_low_fraction = 0.0
        self.last_switch_swing_clear_fraction = 0.0
        self.last_history_next_plant_contact_fraction = 0.0
        self.last_history_next_plant_low_fraction = 0.0
        self.last_history_swing_clear_fraction = 0.0
        self.last_phase_switch_hysteresis = 0.0
        self.last_phase_switch_block_code = -1.0
        self.last_phase_switch_force_armed = 0.0
        self.last_foot_planted_qualities = np.zeros(4, dtype=np.float64)
        self.prev_body_z = None
        self.prev_body_vz = 0.0
        self.rod_role_targets = None
        self.rod_role_lengths = None
        self.phase_leg_targets = None
        self.phase_neutral_targets = None
        self.plant_foot_targets_world = {leg_id: None for leg_id in range(4)}
        self.plant_contact_stability_counts = {leg_id: 0 for leg_id in range(4)}
        self.leg_role_assignments = {}
        self.last_desired_rod_body_vectors = None
        self.last_desired_foot_targets_world = None
        self.phase_zero_action_ctrl_correction = {
            phase: np.zeros(len(self.joint_names), dtype=np.float64)
            for phase in range(self.PHASE_COUNT)
        }
        self.reset_track_vectors_body = None
        self.nominal_action_scale = 1.0
        self.pair_lock_blend = self.FORCED_PAIR_LOCK_BLEND
        self.phase_switch_hysteresis = 0
        self.phase_switch_support_latched = False
        self.phase_switch_contact_history = deque(maxlen=self.PHASE_SWITCH_HISTORY_STEPS)
        self.phase_switch_low_history = deque(maxlen=self.PHASE_SWITCH_HISTORY_STEPS)
        self.phase_switch_swing_clear_history = deque(maxlen=self.PHASE_SWITCH_HISTORY_STEPS)

        self.standing_ctrl = self.joint_midpoints.copy()
        for i, sign in enumerate(self.HIP_STANDING_SIGNS):
            self.standing_ctrl[4 + i] = sign * self.STANDING_HIP_ANGLE
        self.reference_neutral_by_joint = {
            joint_name: float(self.standing_ctrl[index])
            for index, joint_name in enumerate(self.joint_names)
        }
        self.last_ctrl_targets = self.standing_ctrl.copy()
        self.phase_reference_ctrl = self._build_phase_reference_ctrl()

        self._find_spawn_height()
        self.phase_support_offsets_body = self._compute_phase_support_offsets_body()
        self._calibrate_phase_control_corrections()

    def _find_outer_torso_mocap_id(self):
        try:
            body_id = self.model.body("outer_torso_visual").id
        except KeyError:
            return None
        mocap_id = int(self.model.body_mocapid[body_id])
        return mocap_id if mocap_id >= 0 else None

    def _capture_outer_torso_reference(self):
        if self.outer_torso_mocap_id is None:
            return
        internals = self.data.body("internals")
        internals_pos = np.asarray(internals.xpos, dtype=np.float64).copy()
        internals_quat = np.asarray(internals.xquat, dtype=np.float64).copy()
        outer_pos = np.asarray(self.data.mocap_pos[self.outer_torso_mocap_id], dtype=np.float64).copy()
        outer_quat = np.asarray(self.data.mocap_quat[self.outer_torso_mocap_id], dtype=np.float64).copy()
        self.outer_torso_local_pos = quat_rotate_vector(
            quat_conjugate(internals_quat),
            outer_pos - internals_pos,
        )
        self.outer_torso_local_quat = quat_multiply(quat_conjugate(internals_quat), outer_quat)

    def _sync_outer_torso_visual(self):
        if (
            self.outer_torso_mocap_id is None
            or self.outer_torso_local_pos is None
            or self.outer_torso_local_quat is None
        ):
            return
        internals = self.data.body("internals")
        internals_pos = np.asarray(internals.xpos, dtype=np.float64)
        internals_quat = np.asarray(internals.xquat, dtype=np.float64)
        outer_pos = internals_pos + quat_rotate_vector(internals_quat, self.outer_torso_local_pos)
        outer_quat = quat_multiply(internals_quat, self.outer_torso_local_quat)
        self.data.mocap_pos[self.outer_torso_mocap_id] = outer_pos
        self.data.mocap_quat[self.outer_torso_mocap_id] = outer_quat

    def _hide_duplicate_internals_visuals(self):
        internals_id = self.model.body("internals").id
        start = int(self.model.body_geomadr[internals_id])
        count = int(self.model.body_geomnum[internals_id])
        for geom_id in range(start, start + count):
            mesh_id = int(self.model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            mesh_name = self.model.mesh(mesh_id).name
            if any(
                mesh_name == prefix or mesh_name.startswith(prefix + "__")
                for prefix in self.INTERNALS_HIDDEN_MESH_PREFIXES
            ):
                self.model.geom_rgba[geom_id, 3] = 0.0

    def _reattach_leg_visual_meshes(self, spec):
        temp_model = spec.compile()
        temp_data = mujoco.MjData(temp_model)
        mujoco.mj_forward(temp_model, temp_data)

        body_by_name = {body.name: body for body in spec.bodies}
        internals_body = body_by_name["internals"]

        spec_geoms_by_mesh = {}
        for geom in internals_body.geoms:
            mesh_name = getattr(geom, "meshname", "")
            if mesh_name:
                spec_geoms_by_mesh.setdefault(mesh_name, []).append(geom)

        temp_internals_id = temp_model.body("internals").id
        start = int(temp_model.body_geomadr[temp_internals_id])
        count = int(temp_model.body_geomnum[temp_internals_id])
        temp_geom_ids_by_mesh = {}
        for geom_id in range(start, start + count):
            mesh_id = int(temp_model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            mesh_name = temp_model.mesh(mesh_id).name
            temp_geom_ids_by_mesh.setdefault(mesh_name, []).append(geom_id)

        for mesh_name in self.INTERNALS_STATIC_LEG_MESHES:
            for geom in spec_geoms_by_mesh.get(mesh_name, []):
                rgba = np.array(geom.rgba, dtype=np.float64)
                rgba[3] = 0.0
                geom.rgba = rgba

        for mesh_name, target_prefix in self.INTERNALS_REATTACH_MESH_TARGETS.items():
            spec_geoms = spec_geoms_by_mesh.get(mesh_name, [])
            temp_geom_ids = temp_geom_ids_by_mesh.get(mesh_name, [])
            for leg_id, (spec_geom, temp_geom_id) in enumerate(zip(spec_geoms, temp_geom_ids)):
                target_body_name = f"{target_prefix}{leg_id}"
                target_body = body_by_name[target_body_name]
                local_pos, local_quat = self._geom_local_pose_for_body(
                    temp_data,
                    temp_geom_id,
                    target_body_name,
                )
                clone = target_body.add_geom()
                clone.name = f"{target_body_name}_{mesh_name}_visual"
                clone.type = spec_geom.type
                clone.meshname = spec_geom.meshname
                clone.pos = local_pos
                clone.quat = local_quat
                clone.rgba = np.array(spec_geom.rgba, dtype=np.float64)
                clone.group = spec_geom.group
                clone.contype = 0
                clone.conaffinity = 0
                clone.mass = 0.0

                rgba = np.array(spec_geom.rgba, dtype=np.float64)
                rgba[3] = 0.0
                spec_geom.rgba = rgba

        for mesh_name in self.INTERNALS_NEAREST_LEG_HARDWARE_MESHES:
            spec_geoms = spec_geoms_by_mesh.get(mesh_name, [])
            temp_geom_ids = temp_geom_ids_by_mesh.get(mesh_name, [])
            for spec_geom, temp_geom_id in zip(spec_geoms, temp_geom_ids):
                target_body_name = self._nearest_leg_body_name(temp_data, temp_geom_id)
                target_body = body_by_name[target_body_name]
                local_pos, local_quat = self._geom_local_pose_for_body(
                    temp_data,
                    temp_geom_id,
                    target_body_name,
                )
                clone = target_body.add_geom()
                clone.name = f"{target_body_name}_{mesh_name}_visual_{temp_geom_id}"
                clone.type = spec_geom.type
                clone.meshname = spec_geom.meshname
                clone.pos = local_pos
                clone.quat = local_quat
                clone.rgba = np.array(spec_geom.rgba, dtype=np.float64)
                clone.group = spec_geom.group
                clone.contype = 0
                clone.conaffinity = 0
                clone.mass = 0.0

                rgba = np.array(spec_geom.rgba, dtype=np.float64)
                rgba[3] = 0.0
                spec_geom.rgba = rgba

    def _geom_local_pose_for_body(self, data, geom_id, body_name):
        geom_pos = np.array(data.geom_xpos[geom_id], dtype=np.float64)
        geom_mat = np.array(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)

        body = data.body(body_name)
        body_pos = np.array(body.xpos, dtype=np.float64)
        body_mat = np.array(body.xmat, dtype=np.float64).reshape(3, 3)

        local_pos = body_mat.T @ (geom_pos - body_pos)
        local_mat = body_mat.T @ geom_mat
        local_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(local_quat, local_mat.reshape(-1))
        return local_pos, local_quat

    def _nearest_leg_body_name(self, data, geom_id):
        geom_pos = np.array(data.geom_xpos[geom_id], dtype=np.float64)
        candidates = []
        for leg_id in range(4):
            for prefix in ("active_carriage_l", "servo_l", "fixed_carriage_l"):
                body_name = f"{prefix}{leg_id}"
                body_pos = np.array(data.body(body_name).xpos, dtype=np.float64)
                candidates.append((np.linalg.norm(geom_pos - body_pos), body_name))
        return min(candidates, key=lambda item: item[0])[1]

    def set_curriculum(self, nominal_action_scale=None, pair_lock_blend=None):
        recalibrate = False
        if nominal_action_scale is not None:
            updated_scale = float(np.clip(nominal_action_scale, 0.0, 1.5))
            recalibrate = recalibrate or not np.isclose(updated_scale, self.nominal_action_scale)
            self.nominal_action_scale = updated_scale
        if pair_lock_blend is not None:
            updated_blend = float(np.clip(pair_lock_blend, 0.0, 1.0))
            recalibrate = recalibrate or not np.isclose(updated_blend, self.pair_lock_blend)
            self.pair_lock_blend = updated_blend
        if recalibrate and self.phase_reference_ctrl is not None:
            self._calibrate_phase_control_corrections()

    def _set_standing_pose(self):
        for index, joint_name in enumerate(self.joint_names):
            self.data.joint(joint_name).qpos[0] = self.standing_ctrl[index]

    def _phase_pairs(self, phase=None):
        phase = self.phase if phase is None else phase
        phase_index = int(phase) % self.PHASE_COUNT
        plant_ids = tuple(
            leg_id for leg_id, grounded in enumerate(self.PHASE_CONTACT_MASKS[phase_index])
            if grounded > 0.5
        )
        swing_ids = tuple(leg_id for leg_id in range(4) if leg_id not in plant_ids)
        return swing_ids, plant_ids

    def _next_phase(self, phase=None):
        phase = self.phase if phase is None else int(phase)
        return (phase + 1) % self.PHASE_COUNT

    def _phase_transition_roles(self, phase=None):
        phase = self.phase if phase is None else int(phase)
        current_mask = self.PHASE_CONTACT_MASKS[phase % self.PHASE_COUNT] > 0.5
        next_mask = self.PHASE_CONTACT_MASKS[self._next_phase(phase)] > 0.5
        support_ids = tuple(leg_id for leg_id in range(4) if current_mask[leg_id] and next_mask[leg_id])
        touchdown_ids = tuple(leg_id for leg_id in range(4) if (not current_mask[leg_id]) and next_mask[leg_id])
        liftoff_ids = tuple(leg_id for leg_id in range(4) if current_mask[leg_id] and (not next_mask[leg_id]))
        air_ids = tuple(leg_id for leg_id in range(4) if (not current_mask[leg_id]) and (not next_mask[leg_id]))
        motion_ids = tuple(sorted(set(touchdown_ids + liftoff_ids)))
        return {
            "support_ids": support_ids,
            "touchdown_ids": touchdown_ids,
            "liftoff_ids": liftoff_ids,
            "air_ids": air_ids,
            "motion_ids": motion_ids,
            "current_ground_ids": tuple(leg_id for leg_id in range(4) if current_mask[leg_id]),
            "current_air_ids": tuple(leg_id for leg_id in range(4) if not current_mask[leg_id]),
            "next_ground_ids": tuple(leg_id for leg_id in range(4) if next_mask[leg_id]),
            "next_air_ids": tuple(leg_id for leg_id in range(4) if not next_mask[leg_id]),
        }

    def _liftoff_start_progress(self, phase=None):
        phase = self.phase if phase is None else int(phase)
        return float(self.PHASE_LIFTOFF_START_PROGRESS.get(phase % self.PHASE_COUNT, self.SWING_FORWARD_PROGRESS_RATIO))

    def _touchdown_start_progress(self, phase=None, leg_id=None):
        phase = self.phase if phase is None else int(phase)
        if leg_id is not None:
            phase_leg_overrides = self.PHASE_TOUCHDOWN_START_PROGRESS_BY_LEG.get(phase % self.PHASE_COUNT, {})
            if leg_id in phase_leg_overrides:
                return float(phase_leg_overrides[leg_id])
        return float(self.PHASE_TOUCHDOWN_START_PROGRESS.get(phase % self.PHASE_COUNT, self.TOUCHDOWN_START_PROGRESS_RATIO))

    def phase_pairs(self, phase=None):
        return self._phase_pairs(phase)

    def _phase_nominal_action(self, phase=None):
        return np.zeros(self.action_space.shape[0], dtype=np.float64)

    def nominal_action(self, phase=None):
        return self._phase_nominal_action(phase).copy()

    def zero_action(self):
        return np.zeros(self.action_space.shape[0], dtype=np.float64)

    def _oscillator_signal(self, phase=None, phase_timer=None):
        phase = self.phase if phase is None else int(phase)
        phase_timer = self.phase_timer if phase_timer is None else float(phase_timer)
        total_steps = max(self.PHASE_COUNT * self.PHASE_STEPS, 1)
        cycle_step = (phase % self.PHASE_COUNT) * self.PHASE_STEPS + phase_timer
        return float(np.sin(2.0 * np.pi * cycle_step / total_steps))

    def _phase_reset_ctrl(self, phase=None):
        phase = self.phase if phase is None else phase
        roles = self._phase_transition_roles(phase)
        plant_ids = roles["current_ground_ids"]
        swing_ids = roles["motion_ids"] if roles["motion_ids"] else roles["current_air_ids"]
        ctrl = self.standing_ctrl.copy()

        for leg_id in plant_ids:
            ctrl[leg_id] = self.standing_ctrl[leg_id] + self.RESET_PLANT_POSE[0]
            ctrl[4 + leg_id] = self.standing_ctrl[4 + leg_id] + self.RESET_PLANT_POSE[1]
            ctrl[8 + leg_id] = self.standing_ctrl[8 + leg_id] + self.RESET_PLANT_POSE[2]

        for leg_id in swing_ids:
            ctrl[leg_id] = self.standing_ctrl[leg_id] + self.RESET_SWING_POSE[0]
            ctrl[4 + leg_id] = self.standing_ctrl[4 + leg_id] + self.RESET_SWING_POSE[1]
            ctrl[8 + leg_id] = self.standing_ctrl[8 + leg_id] + self.RESET_SWING_POSE[2]

        ctrl = ctrl + self.PHASE_CTRL_BIAS[int(phase) % self.PHASE_COUNT]
        ctrl = self._enforce_tripedal_pair_lock(ctrl)
        ctrl = self._apply_phase_pair_ctrl_offsets(ctrl, phase=phase)
        lo_arr = np.array([lo for lo, _ in self.joint_ranges], dtype=np.float64)
        hi_arr = np.array([hi for _, hi in self.joint_ranges], dtype=np.float64)
        return np.clip(ctrl, lo_arr, hi_arr)

    def _build_phase_reference_ctrl(self):
        return {
            phase: self._phase_reset_ctrl(phase=phase)
            for phase in range(self.PHASE_COUNT)
        }

    def _reference_ctrl_targets(self, phase=None, phase_timer=None):
        phase = self.phase if phase is None else phase
        phase_timer = self.phase_timer if phase_timer is None else phase_timer
        current_ctrl = np.asarray(self.phase_reference_ctrl[int(phase) % self.PHASE_COUNT], dtype=np.float64)
        next_ctrl = np.asarray(self.phase_reference_ctrl[self._next_phase(phase)], dtype=np.float64)
        ctrl = current_ctrl.copy()
        phase_progress = float(np.clip(phase_timer / max(self.PHASE_STEPS, 1), 0.0, 1.0))
        roles = self._phase_transition_roles(phase)

        def smooth_alpha(progress, start):
            alpha = float(np.clip((progress - start) / max(1.0 - start, 1e-6), 0.0, 1.0))
            return alpha * alpha * (3.0 - 2.0 * alpha)

        full_alpha = smooth_alpha(phase_progress, 0.0)
        liftoff_alpha = smooth_alpha(phase_progress, self._liftoff_start_progress(phase))
        for leg_id in roles["support_ids"]:
            indices = (leg_id, 4 + leg_id, 8 + leg_id)
            for index in indices:
                ctrl[index] = (1.0 - full_alpha) * current_ctrl[index] + full_alpha * next_ctrl[index]

        for leg_id in roles["air_ids"]:
            indices = (leg_id, 4 + leg_id, 8 + leg_id)
            for index in indices:
                ctrl[index] = (1.0 - full_alpha) * current_ctrl[index] + full_alpha * next_ctrl[index]

        for leg_id in roles["liftoff_ids"]:
            indices = (leg_id, 4 + leg_id, 8 + leg_id)
            for index in indices:
                ctrl[index] = (1.0 - liftoff_alpha) * current_ctrl[index] + liftoff_alpha * next_ctrl[index]

        for leg_id in roles["touchdown_ids"]:
            touchdown_alpha = smooth_alpha(phase_progress, self._touchdown_start_progress(phase, leg_id=leg_id))
            indices = (leg_id, 4 + leg_id, 8 + leg_id)
            for index in indices:
                ctrl[index] = (1.0 - touchdown_alpha) * current_ctrl[index] + touchdown_alpha * next_ctrl[index]
            touchdown_offset = self.PHASE_TOUCHDOWN_CTRL_OFFSETS.get(phase % self.PHASE_COUNT, {}).get(leg_id)
            if touchdown_offset is not None:
                ctrl[list(indices)] += touchdown_alpha * touchdown_offset

        ctrl = self._enforce_tripedal_pair_lock(ctrl)
        lo_arr = np.array([lo for lo, _ in self.joint_ranges], dtype=np.float64)
        hi_arr = np.array([hi for _, hi in self.joint_ranges], dtype=np.float64)
        return np.clip(ctrl, lo_arr, hi_arr)

    def _effective_action(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != self.action_space.shape[0]:
            raise ValueError(
                f"Expected action with shape ({self.action_space.shape[0]},), got {action.shape}"
            )
        return np.clip(action, -1.0, 1.0)

    def _leg_action_offsets(self, action, leg_id):
        base = leg_id * self.ACTIONS_PER_LEG
        return {
            "angle": float(action[base]) * self.ROD_ANGLE_ACTION_SCALE,
            "length": float(action[base + 1]) * self.ROD_LENGTH_ACTION_SCALE,
        }

    def effective_action(self, action):
        return self._effective_action(action)

    def _centered_hip_thetas(self):
        hip_angles = np.array([
            self.data.joint(f"hip_revolute_l{i}").qpos[0]
            for i in range(4)
        ], dtype=np.float64)
        return hip_angles - self.standing_ctrl[4:8]

    def _centered_leg_state(self, leg_id):
        return np.array([
            self.data.joint(f"shoulder_prismatic_l{leg_id}").qpos[0] - self.standing_ctrl[leg_id],
            self.data.joint(f"hip_revolute_l{leg_id}").qpos[0] - self.standing_ctrl[4 + leg_id],
            self.data.joint(f"knee_prismatic_l{leg_id}").qpos[0] - self.standing_ctrl[8 + leg_id],
        ], dtype=np.float64)

    def _enforce_tripedal_pair_lock(self, ctrl):
        ctrl = np.asarray(ctrl, dtype=np.float64).copy()
        if not self.ENFORCE_TRIPEDAL_PAIR_LOCK:
            return ctrl
        for pair in ((0, 3), (1, 2)):
            a, b = pair
            for offset in (0, 4, 8):
                mean_value = 0.5 * (ctrl[offset + a] + ctrl[offset + b])
                ctrl[offset + a] = mean_value
                ctrl[offset + b] = mean_value
        return ctrl

    def _apply_phase_pair_ctrl_offsets(self, ctrl, phase=None):
        phase = self.phase if phase is None else int(phase)
        ctrl = np.asarray(ctrl, dtype=np.float64).copy()
        for pair, offset in self.PHASE_PAIR_CTRL_OFFSETS.get(phase % self.PHASE_COUNT, {}).items():
            offset = np.asarray(offset, dtype=np.float64)
            for leg_id in pair:
                ctrl[leg_id] += offset[0]
                ctrl[4 + leg_id] += offset[1]
                ctrl[8 + leg_id] += offset[2]
        return ctrl

    def _pair_theta_difference(self, pair):
        if len(pair) < 2:
            return 0.0
        centered = self._centered_hip_thetas()
        values = np.asarray([centered[leg_id] for leg_id in pair], dtype=np.float64)
        return float(np.max(values) - np.min(values))

    def _pair_mean_theta(self, pair):
        if not pair:
            return 0.0
        centered = self._centered_hip_thetas()
        return float(np.mean([centered[leg_id] for leg_id in pair]))

    def _pair_target_theta(self, pair, phase=None):
        if not pair:
            return 0.0
        reference_ctrl = self._reference_ctrl_targets(phase=phase)
        return float(np.mean([
            reference_ctrl[4 + leg_id] - self.standing_ctrl[4 + leg_id]
            for leg_id in pair
        ]))

    def _pair_theta_target_error(self, pair, phase=None):
        return abs(self._pair_mean_theta(pair) - self._pair_target_theta(pair, phase))

    def _theta_sync_score(self, pair):
        theta_diff = self._pair_theta_difference(pair)
        return float(np.exp(-np.square(theta_diff / self.THETA_SYNC_SIGMA)))

    def _theta_target_score(self, pair, phase=None):
        theta_error = self._pair_theta_target_error(pair, phase)
        return float(np.exp(-np.square(theta_error / self.THETA_TARGET_SIGMA)))

    def _pair_leg_difference(self, pair):
        if len(pair) < 2:
            return 0.0
        states = np.asarray([self._centered_leg_state(leg_id) for leg_id in pair], dtype=np.float64)
        centered_states = states - np.mean(states, axis=0, keepdims=True)
        return float(np.sqrt(np.mean(np.sum(np.square(centered_states), axis=1))))

    def _leg_sync_score(self, pair):
        leg_diff = self._pair_leg_difference(pair)
        return float(np.exp(-np.square(leg_diff / self.LEG_SYNC_SIGMA)))

    def _find_spawn_height(self):
        current_qpos = self.data.qpos.copy()
        current_qvel = self.data.qvel.copy()
        max_required_height = 0.0
        candidate_ctrls = [self.standing_ctrl]
        candidate_ctrls.extend(self.phase_reference_ctrl[phase] for phase in range(self.PHASE_COUNT))
        for ctrl in candidate_ctrls:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:7] = self.model.qpos0[:7]
            self.data.qpos[3:7] = [1, 0, 0, 0]
            for index, joint_name in enumerate(self.joint_names):
                self.data.joint(joint_name).qpos[0] = ctrl[index]
            mujoco.mj_forward(self.model, self.data)
            min_foot_z = float("inf")
            for i in range(4):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"servo_l{i}_foot")
                min_foot_z = min(min_foot_z, self.data.geom(gid).xpos[2])
            max_required_height = max(
                max_required_height,
                -min_foot_z + self.FOOT_HALF_HEIGHT + self.FOOT_SPAWN_CLEARANCE,
            )
        self.data.qpos[:] = current_qpos
        self.data.qvel[:] = current_qvel
        mujoco.mj_forward(self.model, self.data)
        self.spawn_height = max_required_height

    def _settle_with_high_gains(self, n_steps=300):
        """Settle the reset pose using high-gain actuators, then restore normal gains."""
        original_gainprm = self.model.actuator_gainprm.copy()
        original_biasprm = self.model.actuator_biasprm.copy()
        original_dof_damping = self.model.dof_damping.copy()

        # Boost actuator gains 7x
        for i in range(self.model.nu):
            kp = self.model.actuator_gainprm[i, 0]
            new_kp = kp * 7.0
            self.model.actuator_gainprm[i, 0] = new_kp
            self.model.actuator_biasprm[i, 1] = -new_kp

        # Boost joint damping 5x to prevent oscillation during settle
        self.model.dof_damping[:] = original_dof_damping * 5.0

        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        # Clamp any joints that drifted outside range during settle
        for i, name in enumerate(self.joint_names):
            lo, hi = self.joint_ranges[i]
            current = float(self.data.joint(name).qpos[0])
            if current < lo or current > hi:
                self.data.joint(name).qpos[0] = float(np.clip(current, lo, hi))

        # Restore everything
        self.model.actuator_gainprm[:] = original_gainprm
        self.model.actuator_biasprm[:] = original_biasprm
        self.model.dof_damping[:] = original_dof_damping
        mujoco.mj_forward(self.model, self.data)

    def _capture_grounded_phase_snapshot(self, phase, settle_steps=300):
        current_qpos = self.data.qpos.copy()
        current_qvel = self.data.qvel.copy()
        current_ctrl = self.data.ctrl.copy()

        def snapshot_state():
            internals_pos = np.asarray(self.data.body("internals").xpos, dtype=np.float64).copy()
            rod_vectors_body = {
                leg_id: self._rod_vector_body(leg_id).copy()
                for leg_id in range(4)
            }
            support_offsets_body = {}
            _, plant_ids = self._phase_pairs(phase)
            for leg_id in plant_ids:
                foot_pos = np.asarray(self.data.geom(f"servo_l{leg_id}_foot").xpos, dtype=np.float64).copy()
                support_offsets_body[leg_id] = self._world_vector_to_body(foot_pos - internals_pos)
            contacts = [int(v) for v in self._foot_on_ground_flags()]
            planted_flags, _ = self._foot_fully_planted_flags(self._foot_on_ground_flags())
            planted = [int(v) for v in planted_flags]
            desired = self._desired_foot_contacts_for_time(phase=phase, phase_timer=0).astype(int).tolist()
            score = 0.0
            for leg_id in range(4):
                if desired[leg_id]:
                    score += 2.0 * planted[leg_id] + contacts[leg_id]
                else:
                    score -= 2.0 * planted[leg_id] + contacts[leg_id]
            return {
                "rod_vectors_body": rod_vectors_body,
                "support_offsets_body": support_offsets_body,
                "score": score,
                "contacts": contacts,
                "planted": planted,
            }

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.model.qpos0[:7]
        self.data.qpos[2] = self.spawn_height
        self.data.qpos[3:7] = [1, 0, 0, 0]
        phase_ctrl = self.phase_reference_ctrl[phase]
        for index, joint_name in enumerate(self.joint_names):
            self.data.joint(joint_name).qpos[0] = phase_ctrl[index]
        self.data.ctrl[:] = phase_ctrl
        mujoco.mj_forward(self.model, self.data)
        self._ground_support_feet(phase)
        mujoco.mj_forward(self.model, self.data)
        grounded_snapshot = snapshot_state()
        self._settle_with_high_gains(n_steps=settle_steps)
        mujoco.mj_forward(self.model, self.data)
        settled_snapshot = snapshot_state()
        snapshot = settled_snapshot if settled_snapshot["score"] >= grounded_snapshot["score"] else grounded_snapshot

        self.data.qpos[:] = current_qpos
        self.data.qvel[:] = current_qvel
        self.data.ctrl[:] = current_ctrl
        mujoco.mj_forward(self.model, self.data)
        return {
            "rod_vectors_body": snapshot["rod_vectors_body"],
            "support_offsets_body": snapshot["support_offsets_body"],
        }

    def _body_rotation(self):
        return np.asarray(self.data.body("internals").xmat, dtype=np.float64).reshape(3, 3)

    def _world_vector_to_body(self, vector):
        return self._body_rotation().T @ np.asarray(vector, dtype=np.float64)

    def _body_vector_to_world(self, vector):
        return self._body_rotation() @ np.asarray(vector, dtype=np.float64)

    def _rod_vector_body(self, leg_id):
        return self._world_vector_to_body(self._rod_vector(leg_id))

    def _rod_angle_from_body_vector(self, rod_vector_body):
        rod_vector_body = np.asarray(rod_vector_body, dtype=np.float64)
        return float(np.arctan2(rod_vector_body[0], -rod_vector_body[2]))

    def _rod_length_from_body_vector(self, rod_vector_body):
        return float(np.linalg.norm(np.asarray(rod_vector_body, dtype=np.float64)))

    def _compose_body_rod_vector(self, leg_id, angle, length):
        if self.phase_leg_targets is None:
            self._initialize_rod_targets()
        lateral = float(self.phase_leg_targets[self.phase][leg_id]["lateral"])
        min_length = max(self.ROD_LENGTH_MIN, abs(lateral) + self.ROD_LENGTH_MARGIN)
        length = float(np.clip(length, min_length, self.ROD_LENGTH_MAX))
        sagittal = np.sqrt(max(length * length - lateral * lateral, 1e-9))
        return np.array([
            sagittal * np.sin(angle),
            lateral,
            -sagittal * np.cos(angle),
        ], dtype=np.float64)

    def _current_centered_joint_state(self):
        return np.array([
            self.data.joint(joint_name).qpos[0] - self.standing_ctrl[index]
            for index, joint_name in enumerate(self.joint_names)
        ], dtype=np.float64)

    def _current_joint_state(self):
        return np.array([
            self.data.joint(joint_name).qpos[0]
            for joint_name in self.joint_names
        ], dtype=np.float64)

    def _swing_pair_feedback_scale(self, phase=None, phase_timer=None, foot_on_ground=None):
        phase = self.phase if phase is None else phase
        phase_timer = self.phase_timer if phase_timer is None else phase_timer
        foot_on_ground = self._foot_on_ground_flags() if foot_on_ground is None else foot_on_ground
        phase_progress = float(np.clip(phase_timer / max(self.PHASE_STEPS, 1), 0.0, 1.0))
        if phase_progress < self.SWING_PAIR_LOCK_START:
            return 0.0
        swing_ids = self._phase_transition_roles(phase)["motion_ids"]
        if any(foot_on_ground[leg_id] for leg_id in swing_ids):
            return 0.0
        return float(np.clip(
            (phase_progress - self.SWING_PAIR_LOCK_START)
            / max(1.0 - self.SWING_PAIR_LOCK_START, 1e-6),
            0.0,
            1.0,
        ))

    def _apply_pair_state_feedback(self, ctrl_targets, phase=None, phase_timer=None, foot_on_ground=None):
        centered_targets = np.asarray(ctrl_targets, dtype=np.float64) - self.standing_ctrl
        centered_state = self._current_centered_joint_state()
        phase = self.phase if phase is None else phase
        phase_timer = self.phase_timer if phase_timer is None else phase_timer
        foot_on_ground = self._foot_on_ground_flags() if foot_on_ground is None else foot_on_ground
        swing_ids = self._phase_transition_roles(phase)["motion_ids"]
        swing_pair_scale = self._swing_pair_feedback_scale(
            phase=phase,
            phase_timer=phase_timer,
            foot_on_ground=foot_on_ground,
        )

        for pair, pair_scale in (
            (swing_ids, swing_pair_scale),
        ):
            for offset, gain in (
                (0, self.SHOULDER_PAIR_FEEDBACK_GAIN),
                (4, self.HIP_PAIR_FEEDBACK_GAIN),
                (8, self.KNEE_PAIR_FEEDBACK_GAIN),
            ):
                pair_indices = [offset + leg_id for leg_id in pair]
                mean_state = float(np.mean(centered_state[pair_indices]))
                effective_gain = self.pair_lock_blend * gain * pair_scale
                for index in pair_indices:
                    centered_targets[index] -= effective_gain * (centered_state[index] - mean_state)

        corrected = self.standing_ctrl + centered_targets
        lo_arr = np.array([lo for lo, hi in self.joint_ranges], dtype=np.float64)
        hi_arr = np.array([hi for lo, hi in self.joint_ranges], dtype=np.float64)
        return np.clip(corrected, lo_arr, hi_arr)

    def _apply_tripedal_pair_lock_feedback(self, ctrl_targets):
        centered_targets = np.asarray(ctrl_targets, dtype=np.float64) - self.standing_ctrl
        centered_state = self._current_centered_joint_state()
        if not self.ENFORCE_TRIPEDAL_PAIR_LOCK:
            return np.asarray(ctrl_targets, dtype=np.float64)

        for pair in ((0, 3), (1, 2)):
            pair_gain = float(self.TRIPEDAL_PAIR_STATE_FEEDBACK_GAIN_BY_PAIR.get(pair, self.TRIPEDAL_PAIR_STATE_FEEDBACK_GAIN))
            for offset in (0, 4, 8):
                pair_indices = [offset + leg_id for leg_id in pair]
                mean_state = float(np.mean(centered_state[pair_indices]))
                for index in pair_indices:
                    centered_targets[index] -= pair_gain * (
                        centered_state[index] - mean_state
                    )

        corrected = self.standing_ctrl + centered_targets
        lo_arr = np.array([lo for lo, hi in self.joint_ranges], dtype=np.float64)
        hi_arr = np.array([hi for lo, hi in self.joint_ranges], dtype=np.float64)
        return np.clip(corrected, lo_arr, hi_arr)

    def _foot_on_ground_flags(self):
        foot_on_ground = [False, False, False, False]
        for contact_id in range(self.data.ncon):
            g1 = self.data.contact[contact_id].geom1
            g2 = self.data.contact[contact_id].geom2
            for leg_id, foot_geom_id in enumerate(self.foot_geom_ids):
                if (g1 == self.floor_geom_id and g2 == foot_geom_id) or (g1 == foot_geom_id and g2 == self.floor_geom_id):
                    foot_on_ground[leg_id] = True
        return foot_on_ground

    def _foot_floor_contact_counts(self):
        counts = [0, 0, 0, 0]
        for contact_id in range(self.data.ncon):
            g1 = self.data.contact[contact_id].geom1
            g2 = self.data.contact[contact_id].geom2
            for leg_id, foot_geom_id in enumerate(self.foot_geom_ids):
                if (g1 == self.floor_geom_id and g2 == foot_geom_id) or (g1 == foot_geom_id and g2 == self.floor_geom_id):
                    counts[leg_id] += 1
        return counts

    def _foot_flatness_score(self, leg_id):
        xmat = np.asarray(self.data.geom(f"servo_l{leg_id}_foot").xmat, dtype=np.float64).reshape(3, 3)
        sole_normal = xmat[:, 2]
        flatness = abs(float(sole_normal[2]))
        score = (flatness - self.FULL_PLANT_MIN_FLATNESS) / max(1.0 - self.FULL_PLANT_MIN_FLATNESS, 1e-6)
        return float(np.clip(score, 0.0, 1.0))

    def _foot_planted_quality(self, leg_id, foot_on_ground=None, contact_counts=None):
        if foot_on_ground is None:
            foot_on_ground = self._foot_on_ground_flags()
        if contact_counts is None:
            contact_counts = self._foot_floor_contact_counts()
        if not foot_on_ground[leg_id]:
            return 0.0
        height = self._foot_world_height(leg_id)
        height_score = 1.0 - np.clip(
            max(height - self.FOOT_HALF_HEIGHT, 0.0) / max(self.FULL_PLANT_MAX_HEIGHT - self.FOOT_HALF_HEIGHT, 1e-6),
            0.0,
            1.0,
        )
        contact_score = np.clip(contact_counts[leg_id] / max(self.FULL_PLANT_MIN_CONTACTS, 1), 0.0, 1.0)
        flatness_score = self._foot_flatness_score(leg_id)
        return float(
            0.50 * height_score
            + 0.30 * contact_score
            + 0.20 * flatness_score
        )

    def _foot_fully_planted_flags(self, foot_on_ground=None, contact_counts=None):
        if foot_on_ground is None:
            foot_on_ground = self._foot_on_ground_flags()
        if contact_counts is None:
            contact_counts = self._foot_floor_contact_counts()
        qualities = np.array([
            self._foot_planted_quality(leg_id, foot_on_ground=foot_on_ground, contact_counts=contact_counts)
            for leg_id in range(4)
        ], dtype=np.float64)
        self.last_foot_planted_qualities = qualities.copy()
        return list(qualities >= self.FULL_PLANT_MIN_QUALITY), qualities

    def _desired_foot_contacts(self, phase=None):
        desired = self._desired_foot_contacts_for_time(phase=phase, phase_timer=self.phase_timer)
        return desired

    def _desired_foot_contacts_for_time(self, phase=None, phase_timer=None):
        phase = self.phase if phase is None else int(phase)
        return self.PHASE_CONTACT_MASKS[phase % self.PHASE_COUNT].copy()

    def _body_hull_on_ground(self):
        for contact_id in range(self.data.ncon):
            g1 = self.data.contact[contact_id].geom1
            g2 = self.data.contact[contact_id].geom2
            if (
                (g1 == self.floor_geom_id and g2 == self.body_collision_geom_id)
                or (g1 == self.body_collision_geom_id and g2 == self.floor_geom_id)
            ):
                return True
        return False

    def _contact_pattern_reward(self, foot_on_ground, phase=None, phase_timer=None):
        fully_planted, _ = self._foot_fully_planted_flags(foot_on_ground=foot_on_ground)
        actual = np.asarray(fully_planted, dtype=bool)
        desired = self._desired_foot_contacts_for_time(phase=phase, phase_timer=phase_timer).astype(bool)
        match_count = float(np.sum(actual == desired))
        foot_match_reward = self.CONTACT_MATCH_REWARD_SCALE * (2.0 * match_count - 4.0)

        swing_ids, plant_ids = self._phase_pairs(phase)
        if bool(np.all(desired)):
            pair_bonus = self.CONTACT_PAIR_BONUS * float(np.all(actual))
        else:
            plant_pair_grounded = float(all(actual[list(plant_ids)]))
            swing_pair_lifted = float(all(~np.asarray(foot_on_ground, dtype=bool)[list(swing_ids)]))
            pair_bonus = self.CONTACT_PAIR_BONUS * (plant_pair_grounded + swing_pair_lifted)
        return float(foot_match_reward + pair_bonus), match_count, pair_bonus

    def _update_leg_role_assignments(self, phase, foot_on_ground):
        roles = self._phase_transition_roles(phase)
        plant_ids = roles["support_ids"]
        motion_ids = roles["motion_ids"]
        current_assignments = {
            leg_id: ("plant" if leg_id in plant_ids else "swing")
            for leg_id in range(4)
        }
        for leg_id, role in current_assignments.items():
            previous_role = self.leg_role_assignments.get(leg_id)
            if leg_id in motion_ids or role == "swing":
                self.plant_foot_targets_world[leg_id] = None
                self.plant_contact_stability_counts[leg_id] = 0
            elif previous_role != "plant":
                self.plant_foot_targets_world[leg_id] = None
                self.plant_contact_stability_counts[leg_id] = 0

        self.leg_role_assignments = current_assignments

    def _desired_leg_role_targets(self, action, phase=None):
        phase = self.phase if phase is None else phase
        if self.phase_leg_targets is None or self.phase_neutral_targets is None:
            self._initialize_rod_targets()
        foot_on_ground = self._foot_on_ground_flags()
        contact_counts = self._foot_floor_contact_counts()
        foot_fully_planted, foot_planted_qualities = self._foot_fully_planted_flags(
            foot_on_ground=foot_on_ground,
            contact_counts=contact_counts,
        )
        phase_progress = float(np.clip(self.phase_timer / max(self.PHASE_STEPS, 1), 0.0, 1.0))
        transition_roles = self._phase_transition_roles(phase)
        current_ground_ids = transition_roles["current_ground_ids"]
        touchdown_ids = transition_roles["touchdown_ids"]
        liftoff_ids = transition_roles["liftoff_ids"]
        next_phase = self._next_phase(phase)
        phase_targets = self.phase_leg_targets[int(phase) % self.PHASE_COUNT]
        next_phase_targets = self.phase_leg_targets[next_phase]
        liftoff_started = phase_progress >= self._liftoff_start_progress(phase)
        active_plant_ids = tuple(
            leg_id for leg_id in current_ground_ids
            if leg_id not in liftoff_ids or not liftoff_started
        )
        active_swing_ids = tuple(
            leg_id for leg_id in range(4)
            if leg_id not in active_plant_ids
        )
        current_assignments = {
            leg_id: ("plant" if leg_id in active_plant_ids else "swing")
            for leg_id in range(4)
        }
        for leg_id, role in current_assignments.items():
            previous_role = self.leg_role_assignments.get(leg_id)
            if role == "swing":
                self.plant_foot_targets_world[leg_id] = None
                self.plant_contact_stability_counts[leg_id] = 0
            elif previous_role != "plant":
                self.plant_foot_targets_world[leg_id] = None
                self.plant_contact_stability_counts[leg_id] = 0
        self.leg_role_assignments = current_assignments
        capture_contacts_allowed = self.phase_timer >= self.PHASE_TARGET_CAPTURE_DELAY_STEPS
        for leg_id in range(4):
            if leg_id in active_plant_ids and capture_contacts_allowed and foot_fully_planted[leg_id]:
                self.plant_contact_stability_counts[leg_id] += 1
            else:
                self.plant_contact_stability_counts[leg_id] = 0
        action = self._effective_action(action)

        targets = {}
        for leg_id in range(4):
            touchdown_started = False
            if leg_id in active_plant_ids:
                motion_role = "support"
                phase_target = phase_targets[leg_id]
            elif leg_id in liftoff_ids:
                motion_role = "liftoff"
                phase_target = next_phase_targets[leg_id]
            elif leg_id in touchdown_ids:
                motion_role = "touchdown"
                touchdown_started = phase_progress >= self._touchdown_start_progress(phase, leg_id=leg_id)
                phase_target = next_phase_targets[leg_id] if touchdown_started else phase_targets[leg_id]
            else:
                motion_role = "air"
                phase_target = phase_targets[leg_id]
            leg_action = self._leg_action_offsets(action, leg_id)
            angle = phase_target["angle"] + self.nominal_action_scale * leg_action["angle"]
            length = phase_target["length"] + self.nominal_action_scale * leg_action["length"]
            vector_body = self._compose_body_rod_vector(leg_id, angle, length)
            body_frame_target = self.body_point_world(
                "internals",
                np.asarray(self.UPPER_ROD_ANCHORS[leg_id], dtype=np.float64) + vector_body,
            )
            stable_plant_contact = (
                motion_role == "support"
                and capture_contacts_allowed
                and foot_fully_planted[leg_id]
                and self.plant_contact_stability_counts[leg_id] >= self.PLANT_CONTACT_STABILITY_STEPS
            )
            if motion_role == "support" and (
                self.plant_foot_targets_world[leg_id] is None
                or (capture_contacts_allowed and not foot_fully_planted[leg_id])
            ):
                if stable_plant_contact:
                    self.plant_foot_targets_world[leg_id] = np.asarray(
                        self._foot_track_world(leg_id),
                        dtype=np.float64,
                    ).copy()
                else:
                    self.plant_foot_targets_world[leg_id] = self._grounded_track_target_world(leg_id, phase=phase)
            elif stable_plant_contact:
                self.plant_foot_targets_world[leg_id] = np.asarray(
                    self._foot_track_world(leg_id),
                    dtype=np.float64,
                ).copy()
            if motion_role == "support" and self.plant_foot_targets_world[leg_id] is not None:
                world_target = self.plant_foot_targets_world[leg_id].copy()
            else:
                target_world_hint = body_frame_target
                if motion_role == "touchdown" and touchdown_started:
                    target_world_hint = self._grounded_track_target_world(leg_id, phase=next_phase)
                world_target = self._swing_target_world(
                    leg_id,
                    target_world_hint,
                    phase=phase,
                    motion_role=motion_role,
                )
                if motion_role == "touchdown" and not touchdown_started:
                    world_target = np.asarray(self._foot_track_world(leg_id), dtype=np.float64).copy()
                    world_target[2] = max(
                        world_target[2],
                        self.FOOT_HALF_HEIGHT + self.EARLY_SWING_MIN_FOOT_CLEARANCE,
                    )
                elif (
                    motion_role in ("touchdown", "air")
                    and phase_progress < self.EARLY_SWING_TOUCHDOWN_WINDOW
                    and foot_on_ground[leg_id]
                ):
                    world_target = self._swing_vertical_hold_target(leg_id)
            if motion_role == "touchdown" and touchdown_started:
                vector_body = self._world_vector_to_body(
                    world_target - self._rod_upper_anchor_world(leg_id)
                )
            elif (
                motion_role in ("touchdown", "air")
                and phase_progress < self.EARLY_SWING_TOUCHDOWN_WINDOW
                and foot_on_ground[leg_id]
            ):
                vector_body = self._world_vector_to_body(
                    world_target - self._rod_upper_anchor_world(leg_id)
                )
            targets[leg_id] = {
                "role": "plant" if motion_role == "support" else "swing",
                "motion_role": motion_role,
                "current_grounded": leg_id in current_ground_ids,
                "planted_quality": float(foot_planted_qualities[leg_id]),
                "angle": angle,
                "length": length,
                "vector_body": vector_body,
                "foot_target_world": world_target,
            }
        return targets

    def _leg_joint_qpos(self, leg_id):
        indices = self.leg_joint_qpos_indices[leg_id]
        return np.asarray([self.data.qpos[index] for index in indices], dtype=np.float64)

    def _leg_joint_target_from_foot_target(self, leg_id, foot_target_world, role=None):
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.foot_track_site_ids[leg_id])
        leg_jac = jacp[:, self.leg_joint_dof_indices[leg_id]]
        foot_pos = self._foot_track_world(leg_id)
        error = np.asarray(foot_target_world, dtype=np.float64) - foot_pos
        solve_matrix = leg_jac @ leg_jac.T + self.IK_DAMPING * np.eye(3, dtype=np.float64)
        delta = leg_jac.T @ np.linalg.solve(solve_matrix, error)
        step_scale = self.IK_STEP_SCALE
        shoulder_scale = 1.0
        hip_scale = 1.0
        knee_scale = 1.0
        if role in ("swing", "air", "liftoff"):
            step_scale *= self.IK_SWING_STEP_SCALE
            shoulder_scale = self.IK_SWING_SHOULDER_SCALE
            hip_scale = self.IK_SWING_HIP_SCALE
            knee_scale = self.IK_SWING_KNEE_SCALE
        elif role == "touchdown":
            step_scale *= self.IK_TOUCHDOWN_STEP_SCALE
            shoulder_scale = self.IK_TOUCHDOWN_SHOULDER_SCALE
            hip_scale = self.IK_TOUCHDOWN_HIP_SCALE
            knee_scale = self.IK_TOUCHDOWN_KNEE_SCALE

        delta *= step_scale
        delta = np.clip(
            delta,
            np.array([
                -self.IK_MAX_SHOULDER_DELTA * shoulder_scale,
                -self.IK_MAX_HIP_DELTA * hip_scale,
                -self.IK_MAX_KNEE_DELTA * knee_scale,
            ], dtype=np.float64),
            np.array([
                self.IK_MAX_SHOULDER_DELTA * shoulder_scale,
                self.IK_MAX_HIP_DELTA * hip_scale,
                self.IK_MAX_KNEE_DELTA * knee_scale,
            ], dtype=np.float64),
        )

        current_qpos = self._leg_joint_qpos(leg_id)
        target_qpos = current_qpos + delta
        range_indices = (leg_id, 4 + leg_id, 8 + leg_id)
        lower = np.array([self.joint_ranges[index][0] for index in range_indices], dtype=np.float64)
        upper = np.array([self.joint_ranges[index][1] for index in range_indices], dtype=np.float64)
        return np.clip(target_qpos, lower, upper)

    def _ctrl_targets_from_action(self, action, phase=None):
        phase = self.phase if phase is None else phase
        targets = self._desired_leg_role_targets(action, phase)
        ctrl_targets = self.standing_ctrl.copy()
        reference_ctrl = self._reference_ctrl_targets(phase=phase, phase_timer=self.phase_timer)
        phase_progress = float(np.clip(self.phase_timer / max(self.PHASE_STEPS, 1), 0.0, 1.0))
        for leg_id in range(4):
            target_qpos = self._leg_joint_target_from_foot_target(
                leg_id,
                targets[leg_id]["foot_target_world"],
                role=targets[leg_id]["motion_role"],
            )
            motion_role = targets[leg_id]["motion_role"]
            blend = 0.0
            if motion_role == "support":
                blend = self.SUPPORT_REFERENCE_CTRL_BLEND
            elif motion_role == "touchdown" and phase_progress < self._touchdown_start_progress(phase):
                blend = float(
                    self.PRETOUCHDOWN_REFERENCE_CTRL_BLEND_BY_PHASE.get(
                        phase % self.PHASE_COUNT,
                        self.PRETOUCHDOWN_REFERENCE_CTRL_BLEND,
                    )
                )
            elif motion_role == "liftoff" and phase_progress < self._liftoff_start_progress(phase):
                blend = float(
                    self.PRELIFTOFF_REFERENCE_CTRL_BLEND_BY_PHASE.get(
                        phase % self.PHASE_COUNT,
                        self.PRELIFTOFF_REFERENCE_CTRL_BLEND,
                    )
                )

            if blend > 0.0:
                ref_qpos = np.array([
                    reference_ctrl[leg_id],
                    reference_ctrl[4 + leg_id],
                    reference_ctrl[8 + leg_id],
                ], dtype=np.float64)
                target_qpos = (1.0 - blend) * target_qpos + blend * ref_qpos

            ctrl_targets[leg_id] = target_qpos[0]
            ctrl_targets[4 + leg_id] = target_qpos[1]
            ctrl_targets[8 + leg_id] = target_qpos[2]

        ctrl_targets = self._enforce_tripedal_pair_lock(ctrl_targets)
        ctrl_targets = self._apply_pair_state_feedback(ctrl_targets, phase=phase)
        ctrl_targets = self._apply_tripedal_pair_lock_feedback(ctrl_targets)
        return ctrl_targets, targets

    def _calibrate_phase_control_corrections(self):
        current_qpos = self.data.qpos.copy()
        current_qvel = self.data.qvel.copy()
        current_ctrl = self.data.ctrl.copy()
        current_phase = self.phase
        current_phase_timer = self.phase_timer
        current_assignments = dict(self.leg_role_assignments)
        current_plant_targets = {
            leg_id: None if target is None else np.asarray(target, dtype=np.float64).copy()
            for leg_id, target in self.plant_foot_targets_world.items()
        }
        current_nominal_action_scale = self.nominal_action_scale
        current_pair_lock_blend = self.pair_lock_blend
        try:
            for phase in range(self.PHASE_COUNT):
                mujoco.mj_resetData(self.model, self.data)
                self._set_phase_pose(phase)
                self._initialize_rod_targets()
                self._begin_phase(phase)
                raw_ctrl, _ = self._ctrl_targets_from_action(self.zero_action(), phase=phase)
                reference_ctrl = np.asarray(self.phase_reference_ctrl[phase], dtype=np.float64)
                self.phase_zero_action_ctrl_correction[phase] = reference_ctrl - raw_ctrl
        finally:
            self.data.qpos[:] = current_qpos
            self.data.qvel[:] = current_qvel
            self.data.ctrl[:] = current_ctrl
            self.phase = current_phase
            self.phase_timer = current_phase_timer
            self.leg_role_assignments = current_assignments
            self.plant_foot_targets_world = current_plant_targets
            self.nominal_action_scale = current_nominal_action_scale
            self.pair_lock_blend = current_pair_lock_blend
            mujoco.mj_forward(self.model, self.data)

    def _scale_action(self, action, phase=None):
        phase = self.phase if phase is None else phase
        ctrl_targets, targets = self._ctrl_targets_from_action(action, phase=phase)
        ctrl_targets = ctrl_targets + self.phase_zero_action_ctrl_correction.get(
            phase,
            np.zeros_like(ctrl_targets),
        )
        lo_arr = np.array([lo for lo, _ in self.joint_ranges], dtype=np.float64)
        hi_arr = np.array([hi for _, hi in self.joint_ranges], dtype=np.float64)
        ctrl_targets = np.clip(ctrl_targets, lo_arr, hi_arr)
        self.last_ctrl_targets = ctrl_targets.copy()
        self.last_desired_rod_body_vectors = {
            leg_id: targets[leg_id]["vector_body"].copy()
            for leg_id in range(4)
        }
        self.last_desired_foot_targets_world = {
            leg_id: targets[leg_id]["foot_target_world"].copy()
            for leg_id in range(4)
        }
        return ctrl_targets

    def control_targets_for_action(self, action, phase=None):
        phase = self.phase if phase is None else phase
        return self._scale_action(self._effective_action(action), phase=phase)

    def body_point_world(self, body_name, local_point=None):
        local_point = np.zeros(3, dtype=np.float64) if local_point is None else np.asarray(local_point, dtype=np.float64)
        body = self.data.body(body_name)
        rotation = np.asarray(body.xmat, dtype=np.float64).reshape(3, 3)
        return np.asarray(body.xpos, dtype=np.float64) + rotation @ local_point

    def _rod_upper_anchor_world(self, leg_id):
        return self.body_point_world("internals", self.UPPER_ROD_ANCHORS[leg_id])

    def _foot_track_world(self, leg_id):
        return np.asarray(
            self.data.site(f"servo_l{leg_id}_track").xpos,
            dtype=np.float64,
        ).copy()

    def _panel_anchor_world(self, leg_id):
        fixed_body_name = f"fixed_carriage_l{leg_id}"
        return self.body_point_world(
            fixed_body_name,
            self.LOWER_ROD_ANCHORS[fixed_body_name],
        )

    def _rod_lower_anchor_world(self, leg_id):
        return self._foot_track_world(leg_id)

    def _rod_vector(self, leg_id):
        return self._rod_lower_anchor_world(leg_id) - self._rod_upper_anchor_world(leg_id)

    def _rod_length(self, leg_id):
        return float(np.linalg.norm(self._rod_vector(leg_id)))

    def _unit_rod_vector(self, leg_id):
        rod_vector = self._rod_vector_body(leg_id)
        rod_length = np.linalg.norm(rod_vector)
        if rod_length < 1e-9:
            return np.zeros(3, dtype=np.float64)
        return rod_vector / rod_length

    def _pair_mean_rod_vector(self, pair):
        if not pair:
            return np.zeros(3, dtype=np.float64)
        vectors = np.array([self._unit_rod_vector(leg_id) for leg_id in pair], dtype=np.float64)
        mean_vector = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm < 1e-9:
            return np.zeros(3, dtype=np.float64)
        return mean_vector / norm

    def _mean_unit_vector(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.clip(norms, 1e-9, None)
        mean_vector = np.mean(normalized, axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm < 1e-9:
            return np.zeros(3, dtype=np.float64)
        return mean_vector / norm

    def _pair_rod_difference(self, pair):
        if len(pair) < 2:
            return 0.0
        vectors = np.asarray([self._unit_rod_vector(leg_id) for leg_id in pair], dtype=np.float64)
        centered_vectors = vectors - np.mean(vectors, axis=0, keepdims=True)
        return float(np.sqrt(np.mean(np.sum(np.square(centered_vectors), axis=1))))

    def _pair_mean_rod_length(self, pair):
        if not pair:
            return 0.0
        return float(np.mean([self._rod_length(leg_id) for leg_id in pair]))

    def _pair_mean_desired_rod_vector(self, pair):
        if self.last_desired_rod_body_vectors is None:
            return None
        if not pair:
            return np.zeros(3, dtype=np.float64)
        vectors = np.array([
            self.last_desired_rod_body_vectors[leg_id]
            for leg_id in pair
        ], dtype=np.float64)
        mean_vector = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm < 1e-9:
            return np.zeros(3, dtype=np.float64)
        return mean_vector / norm

    def _pair_mean_desired_rod_length(self, pair):
        if self.last_desired_rod_body_vectors is None:
            return None
        if not pair:
            return 0.0
        return float(np.mean([
            np.linalg.norm(self.last_desired_rod_body_vectors[leg_id])
            for leg_id in pair
        ]))

    def _pair_rod_target_error(self, pair, role, phase=None):
        target_vector = self._pair_mean_desired_rod_vector(pair)
        if target_vector is None:
            return 0.0
        return float(np.linalg.norm(self._pair_mean_rod_vector(pair) - target_vector))

    def _pair_rod_length_error(self, pair, role, phase=None):
        target_length = self._pair_mean_desired_rod_length(pair)
        if target_length is None:
            return 0.0
        return abs(self._pair_mean_rod_length(pair) - target_length)

    def _rod_sync_score(self, pair):
        rod_diff = self._pair_rod_difference(pair)
        return float(np.exp(-np.square(rod_diff / self.ROD_SYNC_SIGMA)))

    def _rod_target_score(self, pair, role, phase=None):
        rod_error = self._pair_rod_target_error(pair, role, phase)
        length_error = self._pair_rod_length_error(pair, role, phase)
        return float(
            np.exp(-np.square(rod_error / self.ROD_TARGET_SIGMA))
            * np.exp(-np.square(length_error / self.ROD_LENGTH_SIGMA))
        )

    def _foot_forward_offset(self, leg_id):
        lower = self._foot_track_world(leg_id)
        upper = self._rod_upper_anchor_world(leg_id)
        return float(lower[0] - upper[0])

    def _foot_world_height(self, leg_id):
        return float(self.data.geom(f"servo_l{leg_id}_foot").xpos[2])

    def _track_vector_body(self, leg_id):
        return self._world_vector_to_body(
            self._foot_track_world(leg_id) - self._rod_upper_anchor_world(leg_id)
        )

    def _swing_vertical_checkpoint_score(self, leg_id):
        vector_body = self._track_vector_body(leg_id)
        return float(np.exp(-np.square(vector_body[0] / self.SWING_PIVOT_ALIGNMENT_SIGMA)))

    def _swing_lateral_alignment_score(self, leg_id):
        vector_body = self._track_vector_body(leg_id)
        if self.phase_leg_targets is None:
            self._initialize_rod_targets()
        target_lateral = float(self.phase_leg_targets[self.phase][leg_id]["lateral"])
        lateral_error = vector_body[1] - target_lateral
        return float(np.exp(-np.square(lateral_error / self.SWING_LATERAL_ALIGNMENT_SIGMA)))

    def _reach_reset_track_score(self, leg_id):
        if self.reset_track_vectors_body is None:
            return 0.0
        current = self._track_vector_body(leg_id)
        target = np.asarray(self.reset_track_vectors_body[leg_id], dtype=np.float64)
        x_score = np.exp(-np.square((current[0] - target[0]) / self.REACH_RESET_X_SIGMA))
        z_score = np.exp(-np.square((current[2] - target[2]) / self.REACH_RESET_Z_SIGMA))
        return float(x_score * z_score)

    def _phase_switch_ready(self, phase=None):
        phase = self.phase if phase is None else phase
        next_phase = self._next_phase(phase)
        foot_on_ground = self._foot_on_ground_flags()
        contact_counts = self._foot_floor_contact_counts()
        foot_fully_planted, _ = self._foot_fully_planted_flags(
            foot_on_ground=foot_on_ground,
            contact_counts=contact_counts,
        )
        next_desired = self.PHASE_CONTACT_MASKS[next_phase % self.PHASE_COUNT] > 0.5
        next_plant_ids = tuple(np.flatnonzero(next_desired))
        next_air_ids = tuple(np.flatnonzero(~next_desired))
        phase_progress = float(np.clip(self.phase_timer / max(self.PHASE_STEPS, 1), 0.0, 1.0))
        foot_heights = {
            leg_id: self._foot_world_height(leg_id)
            for leg_id in next_plant_ids
        }
        next_plant_contact_fraction = float(np.mean([foot_fully_planted[leg_id] for leg_id in next_plant_ids])) if next_plant_ids else 1.0
        next_plant_low_fraction = float(np.mean([
            foot_heights[leg_id] <= self.PHASE_SWITCH_GROUND_Z
            for leg_id in next_plant_ids
        ])) if next_plant_ids else 1.0
        swing_clear_fraction = float(np.mean([not foot_fully_planted[leg_id] for leg_id in next_air_ids])) if next_air_ids else 1.0
        self.phase_switch_contact_history.append(next_plant_contact_fraction)
        self.phase_switch_low_history.append(next_plant_low_fraction)
        self.phase_switch_swing_clear_history.append(swing_clear_fraction)
        history_next_plant_contact_fraction = float(np.mean(self.phase_switch_contact_history))
        history_next_plant_low_fraction = float(np.mean(self.phase_switch_low_history))
        history_swing_clear_fraction = float(np.mean(self.phase_switch_swing_clear_history))
        min_next_down_contact_fraction = 1.0 if next_plant_ids else 1.0
        min_next_up_clear_fraction = float(
            self.PHASE_SWITCH_MIN_SWING_CLEAR_FRACTION_BY_PHASE.get(
                phase % self.PHASE_COUNT,
                1.0 if not next_air_ids else 0.5,
            )
        )
        min_progress = float(self.PHASE_SWITCH_MIN_PROGRESS_BY_PHASE.get(phase % self.PHASE_COUNT, self.PHASE_SWITCH_MIN_PROGRESS))
        support_ready = (
            phase_progress >= min_progress
            and history_next_plant_contact_fraction >= min_next_down_contact_fraction
            and history_next_plant_low_fraction >= self.PHASE_SWITCH_MIN_NEXT_PLANT_LOW_FRACTION
        )
        if support_ready:
            self.phase_switch_support_latched = True
        swing_ready = history_swing_clear_fraction >= min_next_up_clear_fraction
        instant_full_ready = (
            phase_progress >= min_progress
            and next_plant_contact_fraction >= min_next_down_contact_fraction
            and next_plant_low_fraction >= self.PHASE_SWITCH_MIN_NEXT_PLANT_LOW_FRACTION
            and swing_clear_fraction >= min_next_up_clear_fraction
        )
        force_advance_ready = (
            self.phase_switch_support_latched
            and self.phase_timer >= self.PHASE_STEPS + self.PHASE_SWITCH_FORCE_ADVANCE_DELAY_STEPS
        )
        base_ready = self.phase_switch_support_latched and (swing_ready or force_advance_ready)
        if base_ready:
            self.phase_switch_hysteresis += 1
        else:
            self.phase_switch_hysteresis = 0

        if phase_progress < min_progress:
            block_code = 0
        elif history_next_plant_contact_fraction < min_next_down_contact_fraction:
            block_code = 1
        elif history_next_plant_low_fraction < self.PHASE_SWITCH_MIN_NEXT_PLANT_LOW_FRACTION:
            block_code = 2
        elif not swing_ready and not force_advance_ready:
            block_code = 3
        elif force_advance_ready and not swing_ready:
            block_code = 5
        else:
            block_code = 4

        self.last_next_plant_contact_fraction = float(next_plant_contact_fraction)
        self.last_next_plant_low_fraction = float(next_plant_low_fraction)
        self.last_switch_swing_clear_fraction = float(swing_clear_fraction)
        self.last_history_next_plant_contact_fraction = float(history_next_plant_contact_fraction)
        self.last_history_next_plant_low_fraction = float(history_next_plant_low_fraction)
        self.last_history_swing_clear_fraction = float(history_swing_clear_fraction)
        self.last_phase_switch_hysteresis = float(self.phase_switch_hysteresis)
        self.last_phase_switch_block_code = float(block_code)
        self.last_phase_switch_force_armed = float(force_advance_ready)

        instant_ready_allowed = bool(
            self.PHASE_SWITCH_ALLOW_INSTANT_FULL_READY_BY_PHASE.get(phase % self.PHASE_COUNT, False)
        )
        ready = (
            (instant_ready_allowed and instant_full_ready)
            or self.phase_switch_hysteresis >= self.PHASE_SWITCH_HYSTERESIS_STEPS
        )
        return ready, next_phase, next_plant_ids, foot_heights

    def _swing_foot_forward_score(self, leg_id):
        forward_offset = self._foot_forward_offset(leg_id)
        return float(np.exp(-np.square(forward_offset / self.SWING_PIVOT_ALIGNMENT_SIGMA)))

    def _pair_foot_forward_difference(self, pair):
        if len(pair) < 2:
            return 0.0
        offsets = np.asarray([abs(self._foot_forward_offset(leg_id)) for leg_id in pair], dtype=np.float64)
        return float(np.max(offsets) - np.min(offsets))

    def _pair_foot_forward_sync_score(self, pair):
        foot_diff = self._pair_foot_forward_difference(pair)
        return float(np.exp(-np.square(foot_diff / self.SWING_FOOT_SYNC_SIGMA)))

    def _oscillator_opposition_error(self, reference_ctrl):
        centered_hips = self._centered_hip_thetas()
        target_centered_hips = np.asarray(reference_ctrl[4:8], dtype=np.float64) - self.standing_ctrl[4:8]
        roles = self._phase_transition_roles(self.phase)
        plant_ids = roles["support_ids"]
        swing_ids = roles["motion_ids"]
        if not plant_ids or not swing_ids:
            return 0.0
        outer_current = float(np.mean([centered_hips[leg_id] for leg_id in plant_ids]))
        middle_current = float(np.mean([centered_hips[leg_id] for leg_id in swing_ids]))
        outer_target = float(np.mean([target_centered_hips[leg_id] for leg_id in plant_ids]))
        middle_target = float(np.mean([target_centered_hips[leg_id] for leg_id in swing_ids]))
        return abs((middle_current - outer_current) - (middle_target - outer_target))

    def _oscillator_opposition_score(self, reference_ctrl):
        opposition_error = self._oscillator_opposition_error(reference_ctrl)
        return float(np.exp(-np.square(opposition_error / self.OSCILLATOR_OPPOSITION_SIGMA)))

    def _shaping_terms(
        self,
        gait_reward,
        theta_sync_reward,
        theta_target_reward,
        leg_sync_reward,
        rod_sync_reward,
        rod_target_reward,
        swing_foot_forward_reward,
        swing_vertical_checkpoint_reward,
        swing_lateral_alignment_reward,
        rectangle_checkpoint_reward,
        swing_reach_reset_reward,
        oscillator_opposition_reward,
        phase_contact_quality,
        early_swing_clearance_quality,
        forward_lean_gate,
        vertical_checkpoint_gate,
    ):
        if self.REWARD_PROFILE == "foundation":
            return (
                gait_reward
                + theta_target_reward
                + 0.5 * theta_sync_reward
                + 0.4 * leg_sync_reward
                + 0.2 * rod_sync_reward
                + swing_vertical_checkpoint_reward
                + swing_lateral_alignment_reward
                + phase_contact_quality * 1.5
                + early_swing_clearance_quality
            )

        return (
            gait_reward
            + theta_sync_reward
            + theta_target_reward
            + leg_sync_reward
            + rod_sync_reward
            + rod_target_reward
            + swing_foot_forward_reward
            + swing_vertical_checkpoint_reward
            + swing_lateral_alignment_reward
            + rectangle_checkpoint_reward
            + swing_reach_reset_reward
            + oscillator_opposition_reward
            + phase_contact_quality * 1.2
            + early_swing_clearance_quality * 0.8
            + forward_lean_gate * 0.5
            + vertical_checkpoint_gate
        )

    def leg_visual_state(self, leg_id, phase=None):
        phase = self.phase if phase is None else phase
        roles = self._phase_transition_roles(phase)
        plant_ids = roles["support_ids"]
        return {
            "upper_anchor": self._rod_upper_anchor_world(leg_id),
            "lower_anchor": self._rod_lower_anchor_world(leg_id),
            "desired_lower_anchor": None if self.last_desired_foot_targets_world is None else self.last_desired_foot_targets_world[leg_id].copy(),
            "panel_anchor": self._panel_anchor_world(leg_id),
            "foot_center": np.asarray(
                self.data.geom(f"servo_l{leg_id}_foot").xpos,
                dtype=np.float64,
            ).copy(),
            "role": "plant" if leg_id in plant_ids else "swing",
            "hip_angle": float(self.data.joint(f"hip_revolute_l{leg_id}").qpos[0]),
            "shoulder": float(self.data.joint(f"shoulder_prismatic_l{leg_id}").qpos[0]),
            "knee": float(self.data.joint(f"knee_prismatic_l{leg_id}").qpos[0]),
            "rod_length": self._rod_length(leg_id),
            "rod_vector": self._rod_vector(leg_id),
        }

    def _initialize_rod_targets(self):
        self.phase_leg_targets = {}
        self.phase_neutral_targets = {}
        grounded_vectors = []
        airborne_vectors = []
        grounded_lengths = []
        airborne_lengths = []
        for phase in range(self.PHASE_COUNT):
            snapshot = self._capture_grounded_phase_snapshot(phase)
            phase_targets = {}
            current_mask = self.PHASE_CONTACT_MASKS[phase % self.PHASE_COUNT] > 0.5
            for leg_id in range(4):
                vector_body = np.asarray(snapshot["rod_vectors_body"][leg_id], dtype=np.float64)
                angle = self._rod_angle_from_body_vector(vector_body)
                length = self._rod_length_from_body_vector(vector_body)
                lateral = float(vector_body[1])
                min_length = max(self.ROD_LENGTH_MIN, abs(lateral) + self.ROD_LENGTH_MARGIN)
                length = float(np.clip(length, min_length, self.ROD_LENGTH_MAX))
                phase_targets[leg_id] = {
                    "angle": self._rod_angle_from_body_vector(vector_body),
                    "length": length,
                    "vector_body": vector_body.copy(),
                    "lateral": lateral,
                    "grounded": bool(current_mask[leg_id]),
                }
                if current_mask[leg_id]:
                    grounded_vectors.append(vector_body.copy())
                    grounded_lengths.append(length)
                else:
                    airborne_vectors.append(vector_body.copy())
                    airborne_lengths.append(length)
            self.phase_leg_targets[phase] = phase_targets

        for phase in range(self.PHASE_COUNT):
            next_phase = self._next_phase(phase)
            neutral_targets = {}
            for leg_id in range(4):
                current_target = self.phase_leg_targets[phase][leg_id]
                next_target = self.phase_leg_targets[next_phase][leg_id]
                neutral_targets[leg_id] = {
                    "angle": 0.5 * (current_target["angle"] + next_target["angle"]),
                    "length": 0.5 * (current_target["length"] + next_target["length"]),
                }
            self.phase_neutral_targets[phase] = neutral_targets

        self.rod_role_targets = {
            "grounded": self._mean_unit_vector(grounded_vectors) if grounded_vectors else np.zeros(3, dtype=np.float64),
            "airborne": self._mean_unit_vector(airborne_vectors) if airborne_vectors else np.zeros(3, dtype=np.float64),
        }
        self.rod_role_lengths = {
            "grounded": float(np.mean(grounded_lengths)) if grounded_lengths else 0.0,
            "airborne": float(np.mean(airborne_lengths)) if airborne_lengths else 0.0,
        }

    def _set_phase_pose(self, phase):
        self.data.qpos[:7] = self.model.qpos0[:7]
        phase_ctrl = self.phase_reference_ctrl[phase]
        for index, joint_name in enumerate(self.joint_names):
            self.data.joint(joint_name).qpos[0] = phase_ctrl[index]
        mujoco.mj_forward(self.model, self.data)
        self._ground_support_feet(phase)

    def _set_phase_zero_pose(self):
        self._set_phase_pose(0)

    def _apply_reset_stable_pose(self, phase=None):
        self.data.qpos[:7] = self.model.qpos0[:7]
        phase_ctrl = self._phase_reset_ctrl(phase=phase)
        for index, joint_name in enumerate(self.joint_names):
            self.data.joint(joint_name).qpos[0] = phase_ctrl[index]
        mujoco.mj_forward(self.model, self.data)

    def _ground_support_feet(self, phase):
        _, plant_ids = self._phase_pairs(phase)
        target_z = self.FOOT_HALF_HEIGHT + self.FOOT_SPAWN_CLEARANCE
        support_heights = [float(self.data.geom(f"servo_l{leg_id}_foot").xpos[2]) for leg_id in plant_ids]
        if len(plant_ids) >= 3:
            support_z = min(support_heights)
        else:
            support_z = max(support_heights)
        self.data.qpos[2] -= support_z - target_z
        mujoco.mj_forward(self.model, self.data)

    def _compute_phase_support_offsets_body(self):
        offsets = {}
        for phase in range(self.PHASE_COUNT):
            snapshot = self._capture_grounded_phase_snapshot(phase)
            offsets[phase] = snapshot["support_offsets_body"]
        return offsets

    def _grounded_track_target_world(self, leg_id, phase=None):
        phase = self.phase if phase is None else phase
        support_offset = self.phase_support_offsets_body.get(phase, {}).get(leg_id)
        if support_offset is None:
            return self._foot_track_world(leg_id)
        foot_target = self.body_point_world("internals", support_offset)
        foot_target[2] = self._plant_target_floor_z(leg_id)
        return self._track_target_from_foot_target(leg_id, foot_target)

    def _grounded_foot_target_world(self, leg_id, phase=None):
        phase = self.phase if phase is None else phase
        support_offset = self.phase_support_offsets_body.get(phase, {}).get(leg_id)
        if support_offset is None:
            target = np.asarray(self.data.geom(f"servo_l{leg_id}_foot").xpos, dtype=np.float64).copy()
        else:
            target = self.body_point_world("internals", support_offset)
        target[2] = self._plant_target_floor_z(leg_id)
        return target

    def _plant_target_floor_z(self, leg_id):
        extra_preload = self.PLANT_TARGET_EXTRA_PRELOAD_BY_LEG.get(leg_id, 0.0)
        return self.FOOT_HALF_HEIGHT - self.PLANT_TARGET_PRELOAD_DEPTH - extra_preload

    def _track_target_from_foot_target(self, leg_id, foot_target_world):
        body_name = f"fixed_carriage_l{leg_id}"
        body = self.data.body(body_name)
        rotation = np.asarray(body.xmat, dtype=np.float64).reshape(3, 3)
        local_track_offset = (
            np.asarray(self.LOWER_ROD_ANCHORS[body_name], dtype=np.float64)
            - np.asarray(self.FOOT_CONTACT_GEOMETRY[body_name]["pos"], dtype=np.float64)
        )
        return np.asarray(foot_target_world, dtype=np.float64) + rotation @ local_track_offset

    def _swing_vertical_hold_target(self, leg_id):
        current = self._foot_track_world(leg_id)
        top_anchor = self._rod_upper_anchor_world(leg_id)
        leg_lift_height = self.LEG_SWING_TRACK_LIFT_HEIGHT.get(leg_id, self.SWING_TRACK_LIFT_HEIGHT)
        lift_boost = max(leg_lift_height, 0.5 * self.SWING_EARLY_LIFT_CLEARANCE)
        target = current.copy()
        target[0] = top_anchor[0]
        target[2] = max(
            current[2] + lift_boost,
            self.FOOT_HALF_HEIGHT + self.EARLY_SWING_MIN_FOOT_CLEARANCE,
            self.FOOT_HALF_HEIGHT + leg_lift_height,
        )
        return target

    def _swing_target_world(self, leg_id, body_frame_target, phase=None, motion_role=None):
        phase = self.phase if phase is None else phase
        desired = np.asarray(body_frame_target, dtype=np.float64).copy()
        current = self._foot_track_world(leg_id)
        top_anchor = self._rod_upper_anchor_world(leg_id)
        leg_lift_height = self.LEG_SWING_TRACK_LIFT_HEIGHT.get(leg_id, self.SWING_TRACK_LIFT_HEIGHT)
        leg_late_landing_ratio = self.LEG_SWING_LATE_LANDING_RATIO.get(leg_id, self.SWING_LATE_LANDING_RATIO)
        phase_progress = 0.0
        if phase == self.phase:
            phase_progress = float(np.clip(self.phase_timer / self.PHASE_STEPS, 0.0, 1.0))
        touchdown_start = self._touchdown_start_progress(phase, leg_id=leg_id)
        touchdown_started = motion_role == "touchdown" and phase_progress >= touchdown_start

        target = desired.copy()
        if touchdown_started:
            target[2] = desired[2]
            if not self._foot_on_ground_flags()[leg_id]:
                target[2] = min(target[2], current[2] - self.TOUCHDOWN_TARGET_MIN_DESCENT_STEP)
        else:
            settle_progress = float(np.clip(
                (phase_progress - leg_late_landing_ratio)
                / max(1.0 - leg_late_landing_ratio, 1e-6),
                0.0,
                1.0,
            ))
            peak_track_z = max(current[2], desired[2]) + leg_lift_height
            if phase_progress < leg_late_landing_ratio:
                lift_progress = float(np.clip(
                    phase_progress / max(self.EARLY_SWING_TOUCHDOWN_WINDOW, 1e-6),
                    0.0,
                    1.0,
                ))
                target[2] = current[2] + (peak_track_z - current[2]) * np.sin(0.5 * np.pi * lift_progress)
            else:
                target[2] = peak_track_z + (desired[2] - peak_track_z) * settle_progress
        max_z_step = self.SWING_TARGET_MAX_Z_STEP
        if motion_role == "touchdown":
            max_z_step = self.TOUCHDOWN_TARGET_MAX_Z_STEP
        target[2] = current[2] + np.clip(
            target[2] - current[2],
            -max_z_step,
            max_z_step,
        )

        if motion_role == "touchdown":
            touchdown_progress = float(np.clip(
                (phase_progress - touchdown_start)
                / max(1.0 - touchdown_start, 1e-6),
                0.0,
                1.0,
            ))
            target[0] = (1.0 - touchdown_progress) * top_anchor[0] + touchdown_progress * desired[0]
            target[0] = current[0] + np.clip(
                target[0] - current[0],
                -self.TOUCHDOWN_TARGET_MAX_X_STEP,
                self.TOUCHDOWN_TARGET_MAX_X_STEP,
            )
        elif phase_progress < self.SWING_FORWARD_PROGRESS_RATIO:
            target[0] = top_anchor[0]
        else:
            settle_x_progress = float(np.clip(
                (phase_progress - self.SWING_FORWARD_PROGRESS_RATIO)
                / max(1.0 - self.SWING_FORWARD_PROGRESS_RATIO, 1e-6),
                0.0,
                1.0,
            ))
            target[0] = (1.0 - settle_x_progress) * top_anchor[0] + settle_x_progress * desired[0]
        return target

    def _prime_phase_targets(self, phase, allow_contact_capture=False):
        plant_ids = self._phase_transition_roles(phase)["current_ground_ids"]
        foot_on_ground = self._foot_on_ground_flags()
        foot_fully_planted, _ = self._foot_fully_planted_flags(foot_on_ground=foot_on_ground)
        self.leg_role_assignments = {
            leg_id: ("plant" if leg_id in plant_ids else "swing")
            for leg_id in range(4)
        }
        for leg_id in range(4):
            if leg_id in plant_ids:
                if self.plant_foot_targets_world[leg_id] is None:
                    if allow_contact_capture and foot_fully_planted[leg_id]:
                        self.plant_foot_targets_world[leg_id] = np.asarray(
                            self._foot_track_world(leg_id),
                            dtype=np.float64,
                        ).copy()
                    else:
                        self.plant_foot_targets_world[leg_id] = self._grounded_track_target_world(leg_id, phase=phase)
            else:
                self.plant_foot_targets_world[leg_id] = None

    def _begin_phase(self, phase, preserve_ground_contacts=False):
        self.phase = int(phase)
        self.phase_timer = 0
        self.phase_switch_hysteresis = 0
        self.phase_switch_support_latched = False
        self.phase_switch_contact_history.clear()
        self.phase_switch_low_history.clear()
        self.phase_switch_swing_clear_history.clear()
        self.plant_contact_stability_counts = {leg_id: 0 for leg_id in range(4)}
        self.leg_role_assignments = {}
        self.plant_foot_targets_world = {leg_id: None for leg_id in range(4)}
        if preserve_ground_contacts:
            foot_on_ground = self._foot_on_ground_flags()
            foot_fully_planted, _ = self._foot_fully_planted_flags(foot_on_ground=foot_on_ground)
            plant_ids = self._phase_transition_roles(self.phase)["current_ground_ids"]
            for leg_id in plant_ids:
                if foot_fully_planted[leg_id]:
                    self.plant_foot_targets_world[leg_id] = np.asarray(
                        self._foot_track_world(leg_id),
                        dtype=np.float64,
                    ).copy()
        self._prime_phase_targets(self.phase, allow_contact_capture=False)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.prev_action = self.zero_action()
        requested_start_phase = kwargs.get("start_phase")
        if requested_start_phase is None:
            start_phase = int(self.np_random.integers(0, self.PHASE_COUNT))
        else:
            start_phase = int(requested_start_phase) % self.PHASE_COUNT
        self.phase = start_phase
        self.phase_timer = 0
        self.plant_foot_targets_world = {leg_id: None for leg_id in range(4)}
        self.leg_role_assignments = {}
        self.last_swing_theta_diff = 0.0
        self.last_plant_theta_diff = 0.0
        self.last_theta_sync_reward = 0.0
        self.last_swing_theta_target_error = 0.0
        self.last_plant_theta_target_error = 0.0
        self.last_theta_target_reward = 0.0
        self.last_swing_leg_diff = 0.0
        self.last_plant_leg_diff = 0.0
        self.last_leg_sync_reward = 0.0
        self.last_swing_rod_diff = 0.0
        self.last_plant_rod_diff = 0.0
        self.last_rod_sync_reward = 0.0
        self.last_swing_rod_target_error = 0.0
        self.last_plant_rod_target_error = 0.0
        self.last_swing_rod_length_error = 0.0
        self.last_plant_rod_length_error = 0.0
        self.last_rod_target_reward = 0.0
        self.last_swing_foot_forward_mean = 0.0
        self.last_swing_foot_forward_diff = 0.0
        self.last_swing_foot_forward_reward = 0.0
        self.last_swing_vertical_checkpoint_mean = 0.0
        self.last_swing_lateral_alignment_mean = 0.0
        self.last_swing_lateral_drift_penalty = 0.0
        self.last_contact_pattern_reward = 0.0
        self.last_contact_match_count = 0.0
        self.last_contact_pair_bonus = 0.0
        self.last_gait_reference_hip_reward = 0.0
        self.last_gait_reference_leg_reward = 0.0
        self.last_foot_contacts[:] = 0.0
        self.last_desired_contacts[:] = 0.0
        self.last_scored_phase = 0
        self.last_support_quality = 0.0
        self.last_progress_gate = 0.0
        self.last_body_hull_contact = 0.0
        self.last_phase_contact_quality = 0.0
        self.last_phase_contact_penalty = 0.0
        self.last_forward_lean_penalty = 0.0
        self.last_rectangle_checkpoint_reward = 0.0
        self.last_rectangle_vertical_mean = 0.0
        self.last_rectangle_contact_mean = 0.0
        self.last_swing_reach_reset_reward = 0.0
        self.last_oscillator_opposition_error = 0.0
        self.last_oscillator_opposition_reward = 0.0
        self.last_early_swing_clearance_quality = 0.0
        self.last_early_swing_touchdown_penalty = 0.0
        self.last_vertical_oscillation_penalty = 0.0
        self.last_vertical_stability_gate = 1.0
        self.last_swing_lift_reward = 0.0
        self.last_swing_plant_penalty = 0.0
        self.last_plant_unload_penalty = 0.0
        self.last_locomotion_validity_gate = 0.0
        self.last_phase_transition_reward = 0.0
        self.last_phase_timeout_terminated = 0.0
        self.last_next_plant_contact_fraction = 0.0
        self.last_next_plant_low_fraction = 0.0
        self.last_switch_swing_clear_fraction = 0.0
        self.last_history_next_plant_contact_fraction = 0.0
        self.last_history_next_plant_low_fraction = 0.0
        self.last_history_swing_clear_fraction = 0.0
        self.last_phase_switch_hysteresis = 0.0
        self.last_phase_switch_block_code = -1.0
        self.last_phase_switch_force_armed = 0.0
        self.prev_body_z = None
        self.prev_body_vz = 0.0
        self.last_desired_rod_body_vectors = None
        self.last_desired_foot_targets_world = None
        self.phase_switch_hysteresis = 0
        self.phase_switch_contact_history.clear()
        self.phase_switch_low_history.clear()
        self.phase_switch_swing_clear_history.clear()
        self.plant_contact_stability_counts = {leg_id: 0 for leg_id in range(4)}

        # Start from a clean slate at the correct spawn height
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.spawn_height
        self.data.qpos[3:7] = [1, 0, 0, 0]  # upright orientation

        # Start directly from the phase reference pose so reset does not
        # latch targets from an intermediate standing-to-phase transient.
        phase_ctrl = self.phase_reference_ctrl[start_phase]
        for index, joint_name in enumerate(self.joint_names):
            self.data.joint(joint_name).qpos[0] = phase_ctrl[index]
        self.data.ctrl[:] = phase_ctrl
        mujoco.mj_forward(self.model, self.data)

        # Ground the planted feet and let the pose settle before any plant
        # targets or phase logic trust contacts.
        self._ground_support_feet(start_phase)
        mujoco.mj_forward(self.model, self.data)

        # Short settle under strong damping to remove bounce from the spawn.
        self._settle_with_high_gains(n_steps=300)
        mujoco.mj_forward(self.model, self.data)
        if self.outer_torso_mocap_id is not None and self.outer_torso_local_pos is None:
            self._capture_outer_torso_reference()
        self._sync_outer_torso_visual()
        mujoco.mj_forward(self.model, self.data)

        # Rebuild targets and phase bookkeeping from the settled pose, then
        # allow a short production-gain settle before the episode starts.
        self.phase = start_phase
        self.phase_timer = 0
        self._initialize_rod_targets()
        self._begin_phase(start_phase)
        ik_ctrl = self._scale_action(self.zero_action(), phase=start_phase)
        self.data.ctrl[:] = ik_ctrl
        for i, name in enumerate(self.joint_names):
            lo, hi = self.joint_ranges[i]
            self.data.joint(name).qpos[0] = float(np.clip(ik_ctrl[i], lo, hi))
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.reset_track_vectors_body = {
            leg_id: self._track_vector_body(leg_id).copy()
            for leg_id in range(4)
        }

        self.initial_height = self.data.qpos[2]
        self.initial_body_height = self.data.body("internals").xpos[2]
        self.prev_body_z = float(self.data.qpos[2])
        self.prev_body_vz = float(self.data.qvel[2])
        self.prev_x = self.data.qpos[0]
        self.stagnation_x = self.data.qpos[0]
        self.stagnation_timer = 0
        return self._get_obs(), {}

    def _get_obs(self):
        full_phase = (self.phase * self.PHASE_STEPS + self.phase_timer) / (self.PHASE_COUNT * self.PHASE_STEPS)
        clock = np.array([
            np.sin(2 * np.pi * full_phase),
            np.cos(2 * np.pi * full_phase),
        ])
        return np.concatenate([
            self.data.qpos[2:],
            self.data.qvel,
            np.array([self.data.body("internals").xpos[2]], dtype=np.float64),
            self.data.body("internals").xquat.copy(),
            clock,
        ])

    def step(self, action):
        pre_ready_to_switch, pre_next_phase, _, _ = self._phase_switch_ready(self.phase)
        if pre_ready_to_switch:
            self._begin_phase(pre_next_phase, preserve_ground_contacts=True)

        effective_action = self._effective_action(action)
        step_phase = self.phase
        self.last_scored_phase = int(step_phase)
        self.data.ctrl[:] = self._scale_action(effective_action, phase=step_phase)
        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self._sync_outer_torso_visual()
        mujoco.mj_forward(self.model, self.data)
        self.step_count += 1

        free_joint_z = self.data.qpos[2]
        quat = self.data.body("internals").xquat
        tilt = 2 * np.arccos(np.clip(abs(quat[0]), 0, 1))
        rotation = self._body_rotation()
        forward_axis = rotation[:, 0]
        up_axis = rotation[:, 2]
        height_change = abs(free_joint_z - self.initial_height)
        vertical_velocity = float(self.data.qvel[2])
        vertical_displacement = 0.0 if self.prev_body_z is None else float(free_joint_z - self.prev_body_z)
        vertical_velocity_delta = float(vertical_velocity - self.prev_body_vz)
        body_hull_contact = self._body_hull_on_ground()
        fallen = bool(
            tilt > 0.45
            or height_change > 0.25
            or body_hull_contact
        )

        # 1. Forward progress is the main goal.
        current_x = self.data.qpos[0]
        forward_displacement = current_x - self.prev_x
        self.prev_x = current_x
        progress_reward = self.PROGRESS_REWARD_SCALE * forward_displacement

        # 2. Small velocity bonus to guide early exploration.
        velocity_bonus = self.VELOCITY_BONUS_SCALE * max(self.data.qvel[0], 0.0)

        # 3. Gait shaping for the current crutch phase.
        transition_roles = self._phase_transition_roles(step_phase)
        current_swing_ids = transition_roles["current_air_ids"]
        current_plant_ids = transition_roles["current_ground_ids"]
        support_ids = transition_roles["support_ids"]
        motion_ids = transition_roles["motion_ids"] if transition_roles["motion_ids"] else current_swing_ids
        liftoff_ids = transition_roles["liftoff_ids"]
        phase_progress = float(np.clip(self.phase_timer / max(self.PHASE_STEPS, 1), 0.0, 1.0))
        reference_ctrl = self._reference_ctrl_targets(phase=step_phase, phase_timer=self.phase_timer)
        current_joint_state = self._current_joint_state()
        hip_indices = np.arange(4, 8, dtype=np.int32)
        hip_error = current_joint_state[hip_indices] - reference_ctrl[hip_indices]
        full_joint_error = current_joint_state - reference_ctrl
        gait_reference_hip_reward = self.GAIT_REFERENCE_HIP_REWARD_SCALE * float(np.mean(
            np.exp(-np.square(hip_error / self.GAIT_REFERENCE_HIP_SIGMA))
        ))
        gait_reference_leg_reward = self.GAIT_REFERENCE_LEG_REWARD_SCALE * float(np.mean(
            np.exp(-np.square(full_joint_error / self.GAIT_REFERENCE_LEG_SIGMA))
        ))

        foot_on_ground = self._foot_on_ground_flags()
        contact_counts = self._foot_floor_contact_counts()
        foot_fully_planted, foot_planted_qualities = self._foot_fully_planted_flags(
            foot_on_ground=foot_on_ground,
            contact_counts=contact_counts,
        )
        desired_contacts = self._desired_foot_contacts_for_time(
            phase=step_phase,
            phase_timer=self.phase_timer,
        )
        desired_contact_mask = desired_contacts > 0.5
        actual_contact_mask = np.asarray(foot_fully_planted, dtype=bool)
        gait_reward, contact_match_count, contact_pair_bonus = self._contact_pattern_reward(
            foot_on_ground,
            phase=step_phase,
            phase_timer=self.phase_timer,
        )
        plant_contact_fraction = float(np.mean([foot_fully_planted[i] for i in current_plant_ids])) if current_plant_ids else 1.0
        swing_clear_fraction = float(np.mean([not foot_on_ground[i] for i in current_swing_ids])) if current_swing_ids else 1.0
        plant_loss_fraction = float(np.mean([not foot_fully_planted[i] for i in current_plant_ids])) if current_plant_ids else 0.0
        swing_touch_fraction = float(np.mean([foot_on_ground[i] for i in current_swing_ids])) if current_swing_ids else 0.0
        desired_down_ids = np.flatnonzero(desired_contact_mask)
        desired_up_ids = np.flatnonzero(~desired_contact_mask)
        desired_down_fraction = (
            float(np.mean(actual_contact_mask[desired_down_ids]))
            if desired_down_ids.size
            else 1.0
        )
        desired_up_fraction = (
            float(np.mean(~actual_contact_mask[desired_up_ids]))
            if desired_up_ids.size
            else 1.0
        )
        swing_foot_heights = [self._foot_world_height(i) for i in current_swing_ids]
        early_window = max(
            0.0,
            1.0 - (phase_progress / max(self.EARLY_SWING_TOUCHDOWN_WINDOW, 1e-6)),
        )
        early_height_violation = (
            float(np.mean([
                max(self.EARLY_SWING_MIN_FOOT_CLEARANCE - foot_height, 0.0)
                / self.EARLY_SWING_MIN_FOOT_CLEARANCE
                for foot_height in swing_foot_heights
            ]))
            if swing_foot_heights
            else 0.0
        )
        early_swing_clearance_quality = max(
            0.0,
            1.0 - early_window * (0.9 * swing_touch_fraction + 0.5 * early_height_violation),
        )
        early_swing_touchdown_penalty = -self.EARLY_SWING_TOUCHDOWN_PENALTY_SCALE * early_window * (
            1.5 * swing_touch_fraction + early_height_violation
        )
        support_quality = 0.7 * desired_down_fraction + 0.3 * desired_up_fraction
        upright_gate = float(np.exp(-np.square(tilt / 0.18)))
        spin_gate = float(np.exp(-np.square(self.data.qvel[5] / 0.8)))
        upward_velocity = max(vertical_velocity, 0.0)
        forward_velocity = max(float(self.data.qvel[0]), 0.0)
        forward_lean = max(-float(forward_axis[2]), 0.0)
        forward_lean_gate = float(np.exp(-np.square(forward_lean / self.FORWARD_LEAN_GATE_SIGMA)))
        vertical_stability_gate = float(np.exp(
            -np.square(abs(vertical_displacement) / self.VERTICAL_STABILITY_DISPLACEMENT_SIGMA)
            -np.square(abs(vertical_velocity) / self.VERTICAL_STABILITY_VELOCITY_SIGMA)
        ))
        contact_error = np.abs(actual_contact_mask.astype(np.float64) - desired_contacts)
        phase_contact_quality = max(0.0, 1.0 - float(np.mean(contact_error)))
        phase_contact_penalty = -self.PHASE_CONTACT_PENALTY_SCALE * float(np.mean(contact_error))
        launch_gate = float(np.exp(-np.square((upward_velocity + 0.75 * plant_loss_fraction) / self.LUNGE_PROGRESS_SIGMA)))
        oscillator_opposition_reward = (
            self.OSCILLATOR_OPPOSITION_REWARD_SCALE
            * self._oscillator_opposition_score(reference_ctrl)
        )
        oscillator_opposition_error = self._oscillator_opposition_error(reference_ctrl)
        swing_vertical_scores = [self._swing_vertical_checkpoint_score(i) for i in motion_ids]
        swing_lateral_scores = [self._swing_lateral_alignment_score(i) for i in motion_ids]
        swing_vertical_checkpoint_reward = (
            self.SWING_VERTICAL_CHECKPOINT_REWARD_SCALE * float(np.mean(swing_vertical_scores))
            if swing_vertical_scores
            else 0.0
        )
        swing_lateral_alignment_reward = (
            self.SWING_LATERAL_ALIGNMENT_REWARD_SCALE * float(np.mean(swing_lateral_scores))
            if swing_lateral_scores
            else 0.0
        )
        swing_lateral_drift_penalty = -self.SWING_LATERAL_DRIFT_PENALTY_SCALE * (
            1.0 - (float(np.mean(swing_lateral_scores)) if swing_lateral_scores else 1.0)
        )
        swing_vertical_checkpoint_reward *= vertical_stability_gate
        swing_lateral_alignment_reward *= vertical_stability_gate
        vertical_checkpoint_gate = float(np.mean(swing_vertical_scores)) if swing_vertical_scores else 1.0
        all_leg_vertical_scores = [self._swing_vertical_checkpoint_score(i) for i in range(4)]
        rectangle_vertical_mean = float(np.mean(all_leg_vertical_scores))
        rectangle_contact_mean = float(np.mean([float(c) for c in foot_fully_planted]))
        rectangle_window = float(np.exp(-np.square((phase_progress - self.RECTANGLE_WINDOW_CENTER) / self.RECTANGLE_WINDOW_SIGMA)))
        rectangle_checkpoint_reward = (
            rectangle_window
            * (
                self.RECTANGLE_VERTICAL_REWARD_SCALE * rectangle_vertical_mean
                + self.RECTANGLE_CONTACT_REWARD_SCALE * rectangle_contact_mean
            )
        )
        rectangle_checkpoint_reward *= vertical_stability_gate
        vertical_height_error = abs(free_joint_z - self.initial_height)
        vertical_oscillation_penalty = -(
            1.0 + rectangle_contact_mean
        ) * (
            self.VERTICAL_OSCILLATION_STEP_PENALTY_SCALE * abs(vertical_displacement)
            + self.VERTICAL_OSCILLATION_ACCEL_PENALTY_SCALE * abs(vertical_velocity_delta)
            + self.VERTICAL_OSCILLATION_HEIGHT_PENALTY_SCALE * max(vertical_height_error - 0.004, 0.0)
        )
        swing_reach_reset_reward = 0.0
        if step_phase == 2:
            reach_scores = [self._reach_reset_track_score(i) for i in motion_ids]
            swing_reach_reset_reward = (
                self.SWING_REACH_RESET_REWARD_SCALE * float(np.mean(reach_scores))
                if reach_scores
                else 0.0
            )
        swing_reach_reset_reward *= vertical_stability_gate

        progress_gate = (
            support_quality
            * phase_contact_quality
            * early_swing_clearance_quality
            * upright_gate
            * spin_gate
            * forward_lean_gate
            * launch_gate
            * min(
                oscillator_opposition_reward / max(self.OSCILLATOR_OPPOSITION_REWARD_SCALE, 1e-9),
                1.0,
            )
            * max(vertical_checkpoint_gate, 0.05)
            * max(rectangle_vertical_mean, 0.05)
            * vertical_stability_gate
        )
        swing_foot_forward_offsets = [self._foot_forward_offset(i) for i in motion_ids]
        swing_foot_forward_reward = (
            self.SWING_FOOT_FORWARD_WEIGHT * (
                float(np.mean([self._swing_foot_forward_score(i) for i in motion_ids]))
                if motion_ids
                else 0.0
            )
            + self.SWING_FOOT_PAIR_SYNC_WEIGHT * self._pair_foot_forward_sync_score(motion_ids)
        )
        swing_foot_forward_reward *= vertical_stability_gate

        swing_theta_diff = self._pair_theta_difference(motion_ids)
        plant_theta_diff = self._pair_theta_difference(support_ids)
        theta_sync_reward = (
            self.SWING_THETA_SYNC_WEIGHT * self._theta_sync_score(motion_ids)
            + self.PLANT_THETA_SYNC_WEIGHT * self._theta_sync_score(support_ids)
        )
        swing_theta_target_error = self._pair_theta_target_error(motion_ids, step_phase)
        plant_theta_target_error = self._pair_theta_target_error(support_ids, step_phase)
        theta_target_reward = (
            self.SWING_THETA_TARGET_WEIGHT * self._theta_target_score(motion_ids, step_phase)
            + self.PLANT_THETA_TARGET_WEIGHT * self._theta_target_score(support_ids, step_phase)
        )
        swing_leg_diff = self._pair_leg_difference(motion_ids)
        plant_leg_diff = self._pair_leg_difference(support_ids)
        leg_sync_reward = (
            self.SWING_LEG_SYNC_WEIGHT * self._leg_sync_score(motion_ids)
            + self.PLANT_LEG_SYNC_WEIGHT * self._leg_sync_score(support_ids)
        )
        swing_rod_diff = self._pair_rod_difference(motion_ids)
        plant_rod_diff = self._pair_rod_difference(support_ids)
        rod_sync_reward = (
            self.SWING_ROD_SYNC_WEIGHT * self._rod_sync_score(motion_ids)
            + self.PLANT_ROD_SYNC_WEIGHT * self._rod_sync_score(support_ids)
        )
        swing_rod_target_error = self._pair_rod_target_error(motion_ids, "swing", step_phase)
        plant_rod_target_error = self._pair_rod_target_error(support_ids, "plant", step_phase)
        swing_rod_length_error = self._pair_rod_length_error(motion_ids, "swing", step_phase)
        plant_rod_length_error = self._pair_rod_length_error(support_ids, "plant", step_phase)
        rod_target_reward = (
            self.SWING_ROD_TARGET_WEIGHT * self._rod_target_score(motion_ids, "swing", step_phase)
            + self.PLANT_ROD_TARGET_WEIGHT * self._rod_target_score(support_ids, "plant", step_phase)
        )
        self.last_swing_theta_diff = swing_theta_diff
        self.last_plant_theta_diff = plant_theta_diff
        self.last_theta_sync_reward = theta_sync_reward
        self.last_swing_theta_target_error = swing_theta_target_error
        self.last_plant_theta_target_error = plant_theta_target_error
        self.last_theta_target_reward = theta_target_reward
        self.last_swing_leg_diff = swing_leg_diff
        self.last_plant_leg_diff = plant_leg_diff
        self.last_leg_sync_reward = leg_sync_reward
        self.last_swing_rod_diff = swing_rod_diff
        self.last_plant_rod_diff = plant_rod_diff
        self.last_rod_sync_reward = rod_sync_reward
        self.last_swing_rod_target_error = swing_rod_target_error
        self.last_plant_rod_target_error = plant_rod_target_error
        self.last_swing_rod_length_error = swing_rod_length_error
        self.last_plant_rod_length_error = plant_rod_length_error
        self.last_rod_target_reward = rod_target_reward
        self.last_swing_foot_forward_mean = float(np.mean(swing_foot_forward_offsets)) if swing_foot_forward_offsets else 0.0
        self.last_swing_foot_forward_diff = self._pair_foot_forward_difference(motion_ids)
        self.last_swing_foot_forward_reward = swing_foot_forward_reward
        self.last_swing_vertical_checkpoint_mean = float(np.mean(swing_vertical_scores)) if swing_vertical_scores else 1.0
        self.last_swing_lateral_alignment_mean = float(np.mean(swing_lateral_scores)) if swing_lateral_scores else 1.0
        self.last_swing_lateral_drift_penalty = float(swing_lateral_drift_penalty)
        self.last_rectangle_checkpoint_reward = float(rectangle_checkpoint_reward)
        self.last_rectangle_vertical_mean = float(rectangle_vertical_mean)
        self.last_rectangle_contact_mean = float(rectangle_contact_mean)
        self.last_swing_reach_reset_reward = float(swing_reach_reset_reward)
        self.last_contact_pattern_reward = gait_reward
        self.last_contact_match_count = contact_match_count
        self.last_contact_pair_bonus = contact_pair_bonus
        self.last_gait_reference_hip_reward = gait_reference_hip_reward
        self.last_gait_reference_leg_reward = gait_reference_leg_reward
        self.last_foot_contacts = np.asarray(foot_fully_planted, dtype=np.float64)
        self.last_desired_contacts = desired_contacts
        self.last_support_quality = support_quality
        self.last_progress_gate = progress_gate
        self.last_body_hull_contact = float(body_hull_contact)
        self.last_phase_contact_quality = float(phase_contact_quality)
        self.last_phase_contact_penalty = float(phase_contact_penalty)
        self.last_oscillator_opposition_error = oscillator_opposition_error
        self.last_oscillator_opposition_reward = oscillator_opposition_reward
        self.last_early_swing_clearance_quality = float(early_swing_clearance_quality)
        self.last_early_swing_touchdown_penalty = float(early_swing_touchdown_penalty)
        self.last_vertical_stability_gate = float(vertical_stability_gate)
        locomotion_validity_gate = (
            support_quality
            * phase_contact_quality
            * early_swing_clearance_quality
            * upright_gate
            * forward_lean_gate
            * vertical_stability_gate
        )
        self.last_locomotion_validity_gate = float(locomotion_validity_gate)
        # 4. Soft guardrails for body stability.
        tilt_penalty = -2.5 * max(tilt - 0.10, 0.0)
        height_penalty = -3.0 * max(height_change - 0.015, 0.0)
        lateral_penalty = -0.45 * abs(self.data.qvel[1]) - 0.15 * abs(self.data.qpos[1])
        yaw_penalty = -1.2 * abs(self.data.qvel[5])
        pitch_rate_penalty = -0.9 * abs(self.data.qvel[4])
        roll_rate_penalty = -0.7 * abs(self.data.qvel[3])
        vertical_velocity_penalty = -0.45 * abs(vertical_velocity) - 1.1 * upward_velocity
        forward_lean_penalty = -self.FORWARD_LEAN_PENALTY_SCALE * max(forward_lean - 0.02, 0.0)
        nose_down_penalty = -3.2 * max(-forward_axis[2] - 0.02, 0.0)
        upright_axis_penalty = -1.5 * max(0.90 - up_axis[2], 0.0)
        forward_launch_penalty = -self.FORWARD_LAUNCH_PENALTY_SCALE * forward_velocity * (upward_velocity + plant_loss_fraction)
        plant_unload_penalty = -self.PLANT_UNLOAD_PENALTY_SCALE * plant_loss_fraction
        self.last_forward_lean_penalty = float(forward_lean_penalty)
        self.last_vertical_oscillation_penalty = float(vertical_oscillation_penalty)
        fall_penalty = -8.0 if fallen else 0.0

        # Direct reward for swing feet being off the ground
        swing_lift_reward = 0.0
        for leg_id in motion_ids:
            foot_height = self._foot_world_height(leg_id)
            if foot_height > 0.005:  # foot is actually lifted
                lift_quality = min(foot_height / self.SWING_LIFT_TARGET_HEIGHT, 1.0)
                swing_lift_reward += lift_quality
        swing_lift_reward = self.SWING_LIFT_REWARD_SCALE * swing_lift_reward / max(len(motion_ids), 1)

        # Penalty for swing feet staying planted when they should lift
        swing_plant_penalty = 0.0
        for leg_id in current_swing_ids + liftoff_ids:
            if foot_on_ground[leg_id]:
                swing_plant_penalty -= 2.0
        self.last_swing_lift_reward = float(swing_lift_reward)
        self.last_swing_plant_penalty = float(swing_plant_penalty)
        self.last_plant_unload_penalty = float(plant_unload_penalty)

        smooth_penalty = -0.001 * np.sum(np.square(effective_action - self.prev_action))
        energy_penalty = -0.0003 * np.sum(np.square(effective_action))

        oscillator_opposition_penalty = -self.OSCILLATOR_OPPOSITION_PENALTY_SCALE * oscillator_opposition_error

        shaping_reward = self.SHAPING_REWARD_SCALE * self._shaping_terms(
            gait_reward=gait_reward,
            theta_sync_reward=theta_sync_reward,
            theta_target_reward=theta_target_reward,
            leg_sync_reward=leg_sync_reward,
            rod_sync_reward=rod_sync_reward,
            rod_target_reward=rod_target_reward,
            swing_foot_forward_reward=swing_foot_forward_reward,
            swing_vertical_checkpoint_reward=swing_vertical_checkpoint_reward,
            swing_lateral_alignment_reward=swing_lateral_alignment_reward,
            rectangle_checkpoint_reward=rectangle_checkpoint_reward,
            swing_reach_reset_reward=swing_reach_reset_reward,
            oscillator_opposition_reward=oscillator_opposition_reward,
            phase_contact_quality=phase_contact_quality,
            early_swing_clearance_quality=early_swing_clearance_quality,
            forward_lean_gate=forward_lean_gate,
            vertical_checkpoint_gate=vertical_checkpoint_gate,
        )
        shaping_reward *= locomotion_validity_gate
        gait_reference_hip_reward *= locomotion_validity_gate
        gait_reference_leg_reward *= locomotion_validity_gate
        swing_lift_reward *= locomotion_validity_gate

        phase_transition_reward = 0.0
        phase_timeout_terminated = False

        reward = (
            progress_reward * progress_gate
            + velocity_bonus * progress_gate
            + shaping_reward
            + gait_reference_hip_reward
            + gait_reference_leg_reward
            + swing_lift_reward
            + swing_plant_penalty
            + oscillator_opposition_penalty
            + phase_contact_penalty
            + early_swing_touchdown_penalty
            + swing_lateral_drift_penalty
            + tilt_penalty
            + height_penalty
            + lateral_penalty
            + yaw_penalty
            + pitch_rate_penalty
            + roll_rate_penalty
            + vertical_oscillation_penalty
            + vertical_velocity_penalty
            + forward_lean_penalty
            + nose_down_penalty
            + upright_axis_penalty
            + forward_launch_penalty
            + plant_unload_penalty
            + smooth_penalty
            + energy_penalty
            + fall_penalty
        )

        self.prev_action = effective_action

        # Advance the phase after scoring this step so the next observation
        # matches the phase the next action should follow.
        self.phase_timer += 1
        ready_to_switch, next_phase, _, _ = self._phase_switch_ready(step_phase)
        phase_stalled = self.phase_timer >= self.PHASE_SWITCH_TIMEOUT_STEPS and not ready_to_switch
        # Switch as soon as the phase-specific readiness gate is met. The
        # readiness gate already enforces per-phase minimum progress and
        # hysteresis, so waiting for the full nominal phase duration can let a
        # short-lived valid contact window decay before the runtime is allowed
        # to advance.
        if ready_to_switch:
            phase_transition_reward = self.PHASE_TRANSITION_REWARD_SCALE * locomotion_validity_gate
            reward += phase_transition_reward
            self._begin_phase(next_phase, preserve_ground_contacts=ready_to_switch)
        if phase_stalled:
            reward -= self.PHASE_STALL_PENALTY
            phase_timeout_terminated = True
        obs = self._get_obs()
        self.prev_body_z = float(free_joint_z)
        self.prev_body_vz = float(vertical_velocity)
        self.last_phase_transition_reward = float(phase_transition_reward)
        self.last_phase_timeout_terminated = float(phase_timeout_terminated)

        # Stagnation termination: if no forward progress in 200 steps, end episode.
        self.stagnation_timer += 1
        if self.stagnation_timer >= 200:
            progress_since_checkpoint = self.data.qpos[0] - self.stagnation_x
            if progress_since_checkpoint < 0.01:
                fallen = True
            self.stagnation_x = self.data.qpos[0]
            self.stagnation_timer = 0

        if phase_timeout_terminated:
            fallen = True

        terminated = bool(fallen)
        truncated = bool(self.step_count >= self.MAX_EPISODE_STEPS)
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass
