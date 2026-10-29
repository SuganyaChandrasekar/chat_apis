from pathlib import Path

from dotenv import load_dotenv


def load_local_environment(root_dir: Path | None = None) -> None:
    project_root = (root_dir or Path(__file__).resolve().parent.parent).resolve()
    load_dotenv(project_root / ".env", override=False)


load_local_environment()
