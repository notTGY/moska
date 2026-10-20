import asyncio
import os
import signal
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx


@asynccontextmanager
async def agent_server(
    config_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 2024,
    startup_timeout: float = 60,
    n_jobs_per_worker: int = 16,
) -> AsyncIterator[str]:
    """Run a local Agent Server and stop it when the context exits."""
    config_path = Path(config_path).resolve()
    client_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{client_host}:{port}"
    with TemporaryDirectory(prefix="agent-server-") as workdir:
        for path in config_path.parent.iterdir():
            if path.name != ".langgraph_api":
                (Path(workdir) / path.name).symlink_to(path)
        command = [
            sys.executable, "-m", "langgraph_cli", "dev",
            "--config", str(config_path),
            "--host", host, "--port", str(port),
            "--no-browser", "--no-reload",
            "--n-jobs-per-worker", str(n_jobs_per_worker),
        ]
        with socket.socket() as listener:
            listener.bind((host, port))
        process = await asyncio.create_subprocess_exec(
            *command, cwd=workdir, start_new_session=True,
        )
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                async with asyncio.timeout(startup_timeout):
                    while True:
                        if process.returncode is not None:
                            raise RuntimeError(f"Server exited during startup: {process.returncode}")
                        try:
                            response = await client.get(f"{url}/ok", timeout=2)
                            if response.status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        await asyncio.sleep(0.2)
            yield url
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
