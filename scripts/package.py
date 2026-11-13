"""Package source only, excluding datasets/checkpoints/caches, with SHA256 manifest."""

import argparse
import hashlib
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}")
    ignored = {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "data",
        "runs",
        "logs",
        ".git",
        ".venv",
    }
    files = [
        p
        for p in sorted(project.rglob("*"))
        if p.is_file()
        and not any(
            part in ignored or part.endswith(".egg-info") for part in p.relative_to(project).parts
        )
        and p.suffix not in (".pt", ".pyc", ".npz", ".zip")
        and p.resolve() != output
    ]
    manifest = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(project)
            archive.write(path, f"{project.name}/{relative.as_posix()}")
            manifest.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative.as_posix()}"
            )
        archive.writestr(f"{project.name}/MANIFEST.sha256", "\n".join(manifest) + "\n")
        for directory in ("data/raw", "runs", "logs", "reports"):
            archive.writestr(f"{project.name}/{directory}/", "")
    print(f"Packaged {len(files)} source/config/documentation files: {output}")


if __name__ == "__main__":
    main()
