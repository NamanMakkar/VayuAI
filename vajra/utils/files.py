# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import contextlib
import glob
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

class WorkingDirectory(contextlib.ContextDecorator):
    def __init__(self, new_dir) -> None:
        super().__init__()
        self.dir = new_dir
        self.cwd = Path.cwd().resolve()

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    return path

@contextmanager
def spaces_in_path(path):
    if " " in str(path):
        string = isinstance(path, str)
        path = Path(path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / path.name.replace(" ", "_")

            if path.is_dir():
                shutil.copytree(path, tmp_path)

            elif path.is_file():
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, tmp_path)

            try:
                yield str(tmp_path) if string else tmp_path

            finally:
                if tmp_path.is_dir():
                    shutil.copytree(tmp_path, path, dirs_exist_ok=True)

                elif tmp_path.is_file():
                    shutil.copy2(tmp_path, path)

    else:
        yield path

def file_age(path=__file__):
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)
    return dt.days

def file_date(path=__file__):
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"

def file_size(path):
    if isinstance(path, (str, Path)):
        megabytes = 2 ** 20
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / megabytes
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / megabytes
    return 0.0

def get_latest_run(search_dir="."):
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""

def update_models(model_names=("vajra-v1-nano.pt",), source_dir=Path("."), update_names=False):
    from vajra import Vajra
    from vajra.dataset.utils import default_class_names
    target_dir = source_dir / "updated_models"
    target_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        model_path = source_dir / model_name
        print(f"Loading model from {model_path}")

        model = Vajra(model_path)
        model.half()
        if update_names:
            model.model.names = default_class_names("coco8.yaml")

        save_path = target_dir / model_name

        print(f"Re-saving {model_name} model to {save_path}")
        model.save(save_path, use_dill=False)