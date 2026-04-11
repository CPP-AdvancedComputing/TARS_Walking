import os
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[1]))
load_dotenv(REPO_ROOT / ".env")

HUB_BASE = "https://csu-tide-jupyterhub.nrp-nautilus.io"


class TIDEClient:
    def __init__(
        self,
        token: str | None = None,
        username: str | None = None,
        hub_base: str = HUB_BASE,
    ):
        self.token = token or os.getenv("TIDE_API_KEY") or _required("TIDE_API_KEY")
        self.username = username or os.getenv("TIDE_USERNAME") or _required("TIDE_USERNAME")
        self.hub_base = hub_base.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"token {self.token}"})

    @property
    def hub_api(self) -> str:
        return f"{self.hub_base}/hub/api"

    @property
    def _username_encoded(self) -> str:
        return quote(self.username, safe="")

    @property
    def server_api(self) -> str:
        return f"{self.hub_base}/user/{self.username}/api"

    @property
    def server_ws(self) -> str:
        base = self.hub_base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{base}/user/{self.username}/api"

    def server_status(self) -> dict:
        response = self._session.get(f"{self.hub_api}/users/{self._username_encoded}")
        response.raise_for_status()
        data = response.json()
        default = data.get("servers", {}).get("", {})
        return {
            "ready": default.get("ready", False),
            "stopped": default.get("stopped", True),
            "pending": default.get("pending"),
            "started": default.get("started"),
            "last_activity": default.get("last_activity"),
            "profile": default.get("user_options", {}),
            "url": default.get("url"),
        }

    def start_server(self, wait: bool = True, timeout: int = 120) -> dict:
        import time

        status = self.server_status()
        if status["ready"]:
            return status

        response = self._session.post(f"{self.hub_api}/users/{self._username_encoded}/server")
        if response.status_code not in (200, 201, 202):
            response.raise_for_status()

        if not wait:
            return self.server_status()

        elapsed = 0
        while elapsed < timeout:
            time.sleep(3)
            elapsed += 3
            status = self.server_status()
            if status["ready"]:
                return status
        raise TimeoutError(f"Server did not start within {timeout}s.")

    def stop_server(self) -> None:
        response = self._session.delete(f"{self.hub_api}/users/{self._username_encoded}/server")
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()

    def create_kernel(self, kernel_name: str = "python3") -> str:
        response = self._session.post(f"{self.server_api}/kernels", json={"name": kernel_name})
        response.raise_for_status()
        return response.json()["id"]

    def list_kernels(self) -> list[dict]:
        response = self._session.get(f"{self.server_api}/kernels")
        response.raise_for_status()
        return response.json()

    def delete_kernel(self, kernel_id: str) -> None:
        response = self._session.delete(f"{self.server_api}/kernels/{kernel_id}")
        if response.status_code not in (200, 204):
            response.raise_for_status()

    def _ensure_remote_directory(self, remote_dir: str) -> None:
        if not remote_dir or remote_dir == ".":
            return
        payload = {"type": "directory", "format": "json", "content": []}
        self._session.put(f"{self.server_api}/contents/{remote_dir.lstrip('/')}", json=payload)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        import base64

        remote_dir = str(Path(remote_path).parent)
        self._ensure_remote_directory(remote_dir)
        content = Path(local_path).read_bytes()
        payload = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(content).decode(),
        }
        response = self._session.put(
            f"{self.server_api}/contents/{remote_path.lstrip('/')}",
            json=payload,
        )
        response.raise_for_status()

    def download_file(self, remote_path: str, local_path: str) -> None:
        import base64

        response = self._session.get(
            f"{self.server_api}/contents/{remote_path.lstrip('/')}",
            params={"format": "base64"},
        )
        response.raise_for_status()
        data = response.json()
        content = base64.b64decode(data["content"])
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(content)

    def list_files(self, remote_path: str = "") -> list[dict]:
        response = self._session.get(f"{self.server_api}/contents/{remote_path.lstrip('/')}")
        response.raise_for_status()
        data = response.json()
        if data.get("type") == "directory":
            return [
                {"name": item["name"], "type": item["type"], "size": item.get("size")}
                for item in data.get("content", [])
            ]
        return [{"name": data["name"], "type": data["type"], "size": data.get("size")}]

    def delete_file(self, remote_path: str) -> None:
        response = self._session.delete(f"{self.server_api}/contents/{remote_path.lstrip('/')}")
        if response.status_code not in (200, 204):
            response.raise_for_status()

    def verify_connection(self) -> dict:
        return self.server_status()


def _required(var: str) -> str:
    raise EnvironmentError(f"Required env var '{var}' is not set.")
