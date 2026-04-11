import sys

from rich.console import Console
from rich.table import Table

from .client import TIDEClient
from .jobs import gpu_info, run_code


console = Console()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        console.print("usage: python -m tide.cli <verify|gpuinfo|exec> [...]")
        sys.exit(1)

    client = TIDEClient()
    command = args[0]

    if command == "verify":
        status = client.verify_connection()
        console.print(status)
        return

    if command == "gpuinfo":
        console.print(gpu_info(client))
        return

    if command == "exec":
        if len(args) < 2:
            console.print("usage: python -m tide.cli exec \"print('hi')\"")
            sys.exit(1)
        result = run_code(client, args[1], on_output=lambda text: console.print(text, end=""))
        if result.error:
            console.print(result.error)
            sys.exit(1)
        return

    if command == "ls":
        entries = client.list_files(args[1] if len(args) > 1 else "")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Type", width=6)
        table.add_column("Name")
        table.add_column("Size", justify="right")
        for entry in entries:
            size = f"{entry['size']:,}" if entry.get("size") else "-"
            table.add_row(entry["type"], entry["name"], size)
        console.print(table)
        return

    console.print(f"unknown command: {command}")
    sys.exit(1)
