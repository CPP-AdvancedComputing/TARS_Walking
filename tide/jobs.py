import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .client import TIDEClient
from .execute import execute


@dataclass
class JobResult:
    job_id: str
    status: str
    output: str
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    remote_script_path: Optional[str] = None


def run_script(
    client: TIDEClient,
    local_script: str,
    timeout: int = 3600,
    remote_dir: str = "tars_jobs",
    on_output=None,
    cleanup: bool = True,
) -> JobResult:
    script_path = Path(local_script)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {local_script}")

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    remote_path = f"{remote_dir}/{job_id}_{script_path.name}"

    client.start_server(wait=True)
    client.upload_file(str(script_path), remote_path)

    code = f"import runpy; runpy.run_path('{remote_path}', run_name='__main__')"
    kernel_id = client.create_kernel()
    try:
        exec_result = execute(
            client.server_ws,
            kernel_id,
            client.token,
            code,
            timeout=timeout,
            on_output=on_output,
        )
    finally:
        client.delete_kernel(kernel_id)
        if cleanup:
            try:
                client.delete_file(remote_path)
            except Exception:
                pass

    return JobResult(
        job_id=job_id,
        status="failed" if exec_result.error else "complete",
        output=exec_result.output,
        error=exec_result.error,
        elapsed_seconds=exec_result.elapsed_seconds,
        remote_script_path=None if cleanup else remote_path,
    )


def run_code(
    client: TIDEClient,
    code: str,
    timeout: int = 3600,
    on_output=None,
) -> JobResult:
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    client.start_server(wait=True)
    kernel_id = client.create_kernel()
    try:
        exec_result = execute(
            client.server_ws,
            kernel_id,
            client.token,
            code,
            timeout=timeout,
            on_output=on_output,
        )
    finally:
        client.delete_kernel(kernel_id)

    return JobResult(
        job_id=job_id,
        status="failed" if exec_result.error else "complete",
        output=exec_result.output,
        error=exec_result.error,
        elapsed_seconds=exec_result.elapsed_seconds,
    )


def gpu_info(client: TIDEClient) -> str:
    result = run_code(
        client,
        "import subprocess; print(subprocess.check_output(['nvidia-smi'], text=True))",
        timeout=30,
    )
    return result.output or result.error or "No GPU info available."
