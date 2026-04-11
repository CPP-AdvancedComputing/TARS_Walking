import mujoco


ROOT_BODY_NAME = "internals"


def ensure_free_root(spec, root_body_name=ROOT_BODY_NAME):
    """Give a root-body URDF a floating base when loaded into MuJoCo."""
    if any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in spec.joints):
        return spec

    root_body = next((body for body in spec.bodies if body.name == root_body_name), None)
    if root_body is None:
        root_body = next(
            (
                body for body in spec.bodies
                if body.name != "world" and getattr(body.parent, "name", None) == "world"
            ),
            None,
        )
    if root_body is None:
        raise ValueError("Unable to find a root body to attach a MuJoCo free joint.")

    root_body.add_freejoint()
    return spec


def load_spec_with_free_root(path, root_body_name=ROOT_BODY_NAME):
    spec = mujoco.MjSpec.from_file(path)
    return ensure_free_root(spec, root_body_name=root_body_name)
