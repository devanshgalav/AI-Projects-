"""Execute notebook code cells one by one in a single Python process."""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import nbformat


def load_env(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def run_shell_command(command: str, cwd: Path) -> tuple[int, str]:
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output


def execute_source(source: str, globals_dict: dict[str, object], cwd: Path) -> tuple[bool, str]:
    lines: list[str] = []
    output_parts: list[str] = []

    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("!"):
            if lines:
                buffer = io.StringIO()
                try:
                    with redirect_stdout(buffer):
                        exec("\n".join(lines), globals_dict, globals_dict)
                    printed = buffer.getvalue().strip()
                    if printed:
                        output_parts.append(printed)
                except Exception:
                    output_parts.append(traceback.format_exc())
                    return False, "\n".join(output_parts)
                lines = []
            code, shell_out = run_shell_command(stripped[1:].strip(), cwd)
            if shell_out.strip():
                output_parts.append(shell_out.rstrip())
            if code != 0:
                output_parts.append(f"Shell command failed with exit code {code}")
                return False, "\n".join(output_parts)
        else:
            lines.append(raw_line)

    if lines:
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                exec("\n".join(lines), globals_dict, globals_dict)
            printed = buffer.getvalue().strip()
            if printed:
                output_parts.append(printed)
        except Exception:
            output_parts.append(traceback.format_exc())
            return False, "\n".join(output_parts)

    return True, "\n".join(output_parts)


def run_notebook(notebook_path: Path, start_cell: int = 0, stop_on_error: bool = True) -> int:
    notebook_path = notebook_path.resolve()
    repo_root = notebook_path.parents[3]
    notebook_dir = notebook_path.parent
    load_env(repo_root)
    os.chdir(notebook_dir)

    nb = nbformat.read(notebook_path, as_version=4)
    globals_dict: dict[str, object] = {"__name__": "__main__"}

    code_cells = [(idx, cell) for idx, cell in enumerate(nb.cells) if cell.cell_type == "code"]

    print(f"Notebook: {notebook_path.name}")
    print(f"Working directory: {notebook_dir}")
    print(f"Code cells: {len(code_cells)}")
    print("=" * 80)

    for idx, cell in code_cells:
        if idx < start_cell:
            continue

        source_preview = cell.source.strip().splitlines()
        preview = source_preview[0] if source_preview else "<empty>"
        print(f"\n[Cell {idx}] {preview[:100]}")
        print("-" * 80)

        ok, output = execute_source(cell.source, globals_dict, notebook_dir)
        if output.strip():
            safe_print(output.rstrip())

        if ok:
            print(f"OK Cell {idx} completed")
        else:
            print(f"FAILED Cell {idx}")
            if stop_on_error:
                return 1

    print("\n" + "=" * 80)
    print("All requested cells completed.")
    return 0


def safe_print(text: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.buffer.write((text + "\n").encode(encoding, errors="replace"))
    sys.stdout.buffer.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--start-cell", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    return run_notebook(
        args.notebook,
        start_cell=args.start_cell,
        stop_on_error=not args.continue_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
