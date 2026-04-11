import argparse
import os
import posixpath
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
VENDOR_PATH = REPO_ROOT / "vendor_py" / "usr" / "lib" / "python3" / "dist-packages"
if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tide import TIDEClient, gpu_info, run_code


REMOTE_ROOT = "tars_remote"
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
runpy.run_path("train.py", run_name="__main__")
print("smoke run finished", flush=True)
"""


def download_artifact(client: TIDEClient, remote_name: str, local_name: str) -> None:
    remote_path = posixpath.join(REMOTE_ROOT, remote_name)
    local_path = REPO_ROOT / local_name
    client.download_file(remote_path, str(local_path))
    print(f"downloaded {remote_path} -> {local_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["verify", "gpuinfo", "sync", "smoke-train"])
    parser.add_argument("--timesteps", type=int, default=4096)
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
        result = run_code(
            client,
            remote_smoke_code(args.timesteps),
            timeout=max(3600, args.timesteps),
            on_output=lambda text: print(text, end="", flush=True),
        )
        print(f"\nstatus={result.status} elapsed={result.elapsed_seconds}s", flush=True)
        if result.error:
            print(result.error, file=sys.stderr)
            raise SystemExit(1)
        download_artifact(client, "tars_policy.zip", "tide_tars_policy_smoke.zip")
        return


if __name__ == "__main__":
    main()
