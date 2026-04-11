from pathlib import Path

import mujoco

from mujoco_loader import load_spec_with_free_root


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_URDF_PATH = REPO_ROOT / "robot_mujoco.urdf"
DEFAULT_MODEL_PATH = REPO_ROOT / "tars_mjcf.xml"
DEFAULT_MODEL_PATH_STR = str(DEFAULT_MODEL_PATH)
DEFAULT_URDF_PATH_STR = str(DEFAULT_URDF_PATH)


def resolve_model_path(model_path=None):
    if model_path is None:
        return DEFAULT_MODEL_PATH_STR if DEFAULT_MODEL_PATH.exists() else DEFAULT_URDF_PATH_STR
    return str(model_path)


def load_tars_spec(model_path=None):
    model_path = resolve_model_path(model_path)
    if model_path.lower().endswith(".urdf"):
        return load_spec_with_free_root(model_path)
    return mujoco.MjSpec.from_file(model_path)

