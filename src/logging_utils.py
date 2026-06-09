from datetime import datetime
from pathlib import Path
import shutil


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_log_path(project_root, stem):
    logs_dir = Path(project_root) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{stem}_{timestamp()}.txt"


def archive_existing_output_txts(project_root):
    project_root = Path(project_root)
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        project_root / "prediction_log.txt",
        project_root / "models" / "results.txt",
        project_root / "models" / "hierarchical_results.txt",
    ]

    archived = []
    for path in output_paths:
        if not path.exists():
            continue

        moved_at = datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y%m%d_%H%M%S"
        )
        destination = logs_dir / f"archived_{path.stem}_{moved_at}{path.suffix}"
        counter = 1
        while destination.exists():
            destination = logs_dir / (
                f"archived_{path.stem}_{moved_at}_{counter}{path.suffix}"
            )
            counter += 1

        shutil.move(str(path), destination)
        archived.append(destination)

    return archived
