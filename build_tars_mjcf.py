from tars_model import DEFAULT_MODEL_PATH, DEFAULT_URDF_PATH_STR
from mujoco_loader import load_spec_with_free_root


def main():
    spec = load_spec_with_free_root(DEFAULT_URDF_PATH_STR)
    spec.compile()
    DEFAULT_MODEL_PATH.write_text(spec.to_xml(), encoding="utf-8")
    print(f"Wrote {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
