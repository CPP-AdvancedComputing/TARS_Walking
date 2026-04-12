import argparse
import importlib
import os
import posixpath
import shlex
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
VENDOR_PATH = REPO_ROOT / "vendor_py" / "usr" / "lib" / "python3" / "dist-packages"
if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tide import TIDEClient, gpu_info, run_code


REMOTE_ROOT = "tars_remote"
REMOTE_LIVE_ROLLOUT_DIR = posixpath.join(REMOTE_ROOT, "live_rollouts")
LOCAL_LIVE_ROLLOUT_DIR = REPO_ROOT / "live_rollouts"
SYNC_FILES = [
    "train.py",
    "quick_train.py",
    "training_helpers.py",
    "tars_env.py",
    "tars_model.py",
    "tars_gait_reference.py",
    "mujoco_loader.py",
    "requirements.txt",
    "requirements-optional.txt",
    "tars_mjcf.xml",
    "robot_mujoco.urdf",
    "config.json",
    "diagnose_phase_support.py",
    "diagnose_phase_control_consistency.py",
    "audit_leg1_geometry.py",
    "sweep_phase_pose_offsets.py",
    "sweep_phase1_l1_bias.py",
    "sweep_phase1_l1_joint_bias.py",
    "diagnose_single_swing_support.py",
    "diagnose_reset_settle_transient.py",
    "diagnose_phase1_contacts.py",
    "diagnose_phase0_leg_jacobians.py",
    "diagnose_phase0_foot_alignment.py",
    "diagnose_phase_chain_trace.py",
    "diagnose_phase_snapshot_geometry.py",
    "diagnose_phase_transition_trace.py",
    "search_phase1_hold_action.py",
    "search_phase0_l0_l3_direct_pose.py",
    "search_phase0_l2_direct_transition_pose.py",
    "search_phase0_l2_large_vertical_alignment.py",
    "search_phase1_outer_support_pose.py",
    "search_phase0_offlegs_direct_pose.py",
    "search_phase0_l2_tracksite_offsets.py",
    "search_phase0_pairlock_pose.py",
    "search_phase0_transition_touchdown_offsets.py",
    "search_phase0_transition_touchdown_target_offsets.py",
    "search_phase1_l1_direct_pose.py",
    "search_phase1_l0_l1_direct_pose.py",
    "search_phase2_l1_l2_direct_pose.py",
    "search_phase2_l2_support_pose.py",
    "search_phase1_chain_support_offsets.py",
    "search_phase2_chain_return_offsets.py",
    "search_phase2_chain_timing_offsets.py",
    "search_phase2_return_mechanics.py",
    "search_phase2_leg1_return_offsets.py",
    "sweep_l1_contact_geometry_z.py",
    "sweep_l1_actuator_strength.py",
    "test_l1_geometry_candidates.py",
    "test_l1_axis_candidates.py",
    "test_l1_mjcf_pose_candidates.py",
    "test_l1_pure_phase_hold_candidates.py",
    "verify_tripedal_pair_gait.py",
]

ENV_PASSTHROUGH_KEYS = [
    "TARS_REWARD_PROFILE",
    "TARS_MAX_EPISODE_STEPS",
    "TARS_SUPPORT_PAIR",
]


def iter_sync_paths():
    for rel in SYNC_FILES:
        path = REPO_ROOT / rel
        if path.exists():
            yield path
    for path in sorted(REPO_ROOT.glob("*.stl")):
        yield path
    assets_dir = REPO_ROOT / "assets"
    for path in sorted(assets_dir.rglob("*.stl")):
        yield path


def remote_path_for(local_path: Path) -> str:
    rel = local_path.relative_to(REPO_ROOT).as_posix()
    return posixpath.join(REMOTE_ROOT, rel)


def sync_project(client: TIDEClient) -> list[str]:
    uploaded = []
    for local_path in iter_sync_paths():
        remote_path = remote_path_for(local_path)
        print(f"upload {local_path.relative_to(REPO_ROOT)} -> {remote_path}", flush=True)
        client.upload_file(str(local_path), remote_path)
        uploaded.append(remote_path)
    return uploaded


def remote_smoke_code(timesteps: int) -> str:
    env_passthrough = {
        key: os.environ[key]
        for key in ENV_PASSTHROUGH_KEYS
        if key in os.environ and os.environ[key] != ""
    }
    return f"""
import importlib
import os
import runpy
import subprocess
import sys
from pathlib import Path

root = Path("{REMOTE_ROOT}")
os.chdir(root)
print("remote cwd:", Path.cwd(), flush=True)

required = {{
    "gymnasium": "gymnasium==1.2.3",
    "mujoco": "mujoco==3.6.0",
    "stable_baselines3": "stable-baselines3==2.7.1",
    "PIL": "Pillow==12.1.1",
    "tensorboard": "tensorboard",
}}
missing = []
for module_name, package_name in required.items():
    try:
        importlib.import_module(module_name)
        print("have", module_name, flush=True)
    except Exception:
        missing.append(package_name)

if missing:
    print("installing:", " ".join(missing), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)

os.environ["LIVE_VIEWER"] = "0"
os.environ["TOTAL_TIMESTEPS"] = "{timesteps}"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["TARS_VIDEO_EVERY"] = "20000"
os.environ["TARS_VIDEO_STEPS"] = "180"
os.environ["TARS_VIDEO_DIR"] = "live_rollouts"
for key, value in {env_passthrough!r}.items():
    os.environ[key] = value
runpy.run_path("train.py", run_name="__main__")
print("smoke run finished", flush=True)
"""


def remote_script_code(script: str, script_args: list[str]) -> str:
    env_passthrough = {
        key: os.environ[key]
        for key in ENV_PASSTHROUGH_KEYS
        if key in os.environ and os.environ[key] != ""
    }
    return f"""
import importlib
import os
import runpy
import subprocess
import sys
from pathlib import Path

root = Path("{REMOTE_ROOT}")
os.chdir(root)
print("remote cwd:", Path.cwd(), flush=True)

required = {{
    "gymnasium": "gymnasium==1.2.3",
    "mujoco": "mujoco==3.6.0",
}}
missing = []
for module_name, package_name in required.items():
    try:
        importlib.import_module(module_name)
        print("have", module_name, flush=True)
    except Exception:
        missing.append(package_name)

if missing:
    print("installing:", " ".join(missing), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)

os.environ["LIVE_VIEWER"] = "0"
os.environ.setdefault("MUJOCO_GL", "egl")
for key, value in {env_passthrough!r}.items():
    os.environ[key] = value
sys.argv = [{script!r}, *{script_args!r}]
runpy.run_path({script!r}, run_name="__main__")
"""


def remote_command_code(command: str) -> str:
    return f"""
import os
import subprocess
from pathlib import Path

root = Path("{REMOTE_ROOT}")
os.chdir(root)
print("remote cwd:", Path.cwd(), flush=True)
os.environ.setdefault("MUJOCO_GL", "egl")
subprocess.run({command!r}, shell=True, check=True)
"""


def download_artifact(client: TIDEClient, remote_name: str, local_name: str) -> None:
    remote_path = posixpath.join(REMOTE_ROOT, remote_name)
    local_path = REPO_ROOT / local_name
    client.download_file(remote_path, str(local_path))
    print(f"downloaded {remote_path} -> {local_path}", flush=True)


def mirror_live_rollouts(client: TIDEClient, stop_event: threading.Event, interval_sec: float = 20.0) -> None:
    downloaded = set()
    LOCAL_LIVE_ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    while not stop_event.is_set():
        try:
            entries = client.list_files(REMOTE_LIVE_ROLLOUT_DIR)
            for entry in entries:
                name = entry["name"]
                if not name.endswith(".gif") or name in downloaded:
                    continue
                remote_path = posixpath.join(REMOTE_LIVE_ROLLOUT_DIR, name)
                local_path = LOCAL_LIVE_ROLLOUT_DIR / name
                client.download_file(remote_path, str(local_path))
                downloaded.add(name)
                print(f"mirrored {remote_path} -> {local_path}", flush=True)
        except Exception:
            pass
        stop_event.wait(interval_sec)

    try:
        entries = client.list_files(REMOTE_LIVE_ROLLOUT_DIR)
        for entry in entries:
            name = entry["name"]
            if not name.endswith(".gif") or name in downloaded:
                continue
            remote_path = posixpath.join(REMOTE_LIVE_ROLLOUT_DIR, name)
            local_path = LOCAL_LIVE_ROLLOUT_DIR / name
            client.download_file(remote_path, str(local_path))
            print(f"mirrored {remote_path} -> {local_path}", flush=True)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["verify", "gpuinfo", "sync", "smoke-train", "remote-cmd", "remote-script"])
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--remote-cmd", help="Shell command to run on the remote TIDE workspace after syncing.")
    parser.add_argument("--script", help="Script path under the repo to run on the remote TIDE workspace.")
    parser.add_argument("--script-args", default="", help="Arguments passed to the remote script.")
    args = parser.parse_args()

    client = TIDEClient()

    if args.command == "verify":
        print(client.verify_connection())
        return

    if args.command == "gpuinfo":
        print(gpu_info(client))
        return

    if args.command == "sync":
        sync_project(client)
        return

    if args.command == "smoke-train":
        sync_project(client)
        stop_event = threading.Event()
        mirror_thread = threading.Thread(
            target=mirror_live_rollouts,
            args=(client, stop_event),
            daemon=True,
        )
        mirror_thread.start()
        result = run_code(
            client,
            remote_smoke_code(args.timesteps),
            timeout=max(3600, args.timesteps),
            on_output=lambda text: print(text, end="", flush=True),
        )
        stop_event.set()
        mirror_thread.join(timeout=5.0)
        print(f"\nstatus={result.status} elapsed={result.elapsed_seconds}s", flush=True)
        if result.error:
            print(result.error, file=sys.stderr)
            raise SystemExit(1)
        download_artifact(client, "tars_policy.zip", "tide_tars_policy_smoke.zip")
        return

    if args.command == "remote-cmd":
        if not args.remote_cmd:
            raise SystemExit("--remote-cmd is required")
        sync_project(client)
        result = run_code(
            client,
            remote_command_code(args.remote_cmd),
            timeout=3600,
            on_output=lambda text: print(text, end="", flush=True),
        )
        print(f"\nstatus={result.status} elapsed={result.elapsed_seconds}s", flush=True)
        if result.error:
            print(result.error, file=sys.stderr)
            raise SystemExit(1)
        return

    if args.command == "remote-script":
        if not args.script:
            raise SystemExit("--script is required")
        sync_project(client)
        result = run_code(
            client,
            remote_script_code(args.script, shlex.split(args.script_args)),
            timeout=3600,
            on_output=lambda text: print(text, end="", flush=True),
        )
        print(f"\nstatus={result.status} elapsed={result.elapsed_seconds}s", flush=True)
        if result.error:
            print(result.error, file=sys.stderr)
            raise SystemExit(1)
        return


if __name__ == "__main__":
    main()
