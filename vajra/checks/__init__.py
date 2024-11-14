# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import os
import re
import glob
import math
import time
import shutil
import contextlib
import inspect
import platform
import subprocess
from pathlib import Path
try:
    from importlib import metadata
except ImportError as e:
    import importlib_metadata as metadata
from typing import Optional
import cv2
import torch
import numpy as np
import requests
from matplotlib import font_manager

from vajra.utils import (
    ASSETS,
    AUTOINSTALL,
    LINUX,
    LOGGER,
    ONLINE,
    ROOT,
    USER_CONFIG_DIR,
    SimpleNamespace,
    IterableSimpleNamespace,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    downloads,
    emojis,
    is_colab,
    is_docker,
    is_github_action_running,
    is_jupyter,
    is_kaggle,
    is_online,
    is_pip_package,
    url2file
)


PYTHON_VERSION = platform.python_version()

def parse_version(version="0.0.0") -> tuple:
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))
    except Exception as e:
        LOGGER.warning(f'WARNING! failure for parse_version({version}), returning (0, 0, 0): {e}')
        return 0, 0, 0

def is_ascii(string) -> bool:
    string = str(string)
    return all(ord(c) < 128 for c in string)

def check_version(current="0.0.0", required="0.0.0", name="version", strict=False, verbose=False, message=""):
    if not current:
        LOGGER.warning(f'WARNING! Invalid check_verison({current}, {required}) requested, please check requested version')
        return True
    elif not current[0].isdigit():
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError as e:
            if strict:
                raise ModuleNotFoundError(emojis(f'WARNING! {current} package is required but not installed')) from e
            else:
                return False
    if not required:
        return True

    op = ""
    version = ""
    result = True
    curr_version_digits = parse_version(current)

    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()
        v = parse_version(version)

        if op == "==" and curr_version_digits != v:
            result = False
        
        elif op == "!=" and curr_version_digits == v:
            result = False
        
        elif op in (">=", "") and not (curr_version_digits >= v):
            result = False

        elif op == "<=" and not (curr_version_digits <= v):
            result = False

        elif op == ">" and not (curr_version_digits > v):
            result = False
        
        elif op == "<" and not (curr_version_digits < v):
            result = False

    if not result:
        warning = f'WARNING! {name}{op}{version} is required but {name}=={current} is currently installed {message}'
        if strict:
            raise ModuleNotFoundError(warning)
        if verbose:
            LOGGER.warning(warning)
    return result

def parse_requirements(file_path = ROOT.parent / "requirements.txt", package=""):
    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra ==" not in x]
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.split('#')[0].strip()
            match = re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line)
            if match:
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))
    
    return requirements

@TryExcept()
def check_requirements(requirements=ROOT.parent / "requirements.txt", exclude=(), install=True, cmds=""):
    prefix = colorstr("red", "bold", "Requirements:")
    if isinstance(requirements, Path):
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        requirements = [f"{x.name}{x.specifier}" for x in parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]
    packages = []
    for r in requirements:
        r_stripped = r.split("/")[-1].replace(".git", "")
        match = re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", r_stripped)
        name, required = match[1], match[2].strip() if match[2] else ""

        try:
            assert check_version(metadata.version(name), required)
        except (AssertionError, metadata.PackageNotFoundError):
            packages.append(r)

    console_string = " ".join(f'"{x}"' for x in packages)
    if console_string:
        if install and AUTOINSTALL:
            num_packages = len(packages)
            LOGGER.info(f'{prefix} Vajra requirement{"s" * (num_packages > 1)}{packages} not found, attempting AutoUpdate...')
            try:
                t = time.time()
                assert is_online(), "AutoUpdate skipped (offline)"
                LOGGER.info(subprocess.check_output(f'pip install --no-cache {console_string} {cmds}', shell=True).decode())
                dt = time.time() - t
                LOGGER.info(
                    f"{prefix} AutoUpdate success! {dt:.1f}s, installed {num_packages} package{'s' * (num_packages > 1)}: {packages}\n"
                    f"{prefix} {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
                )
            except Exception as e:
                LOGGER.warning(f'{prefix} {e}')
                return False
    else:
        return False

    return True

def check_img_size(img_size, stride=32, min_dim=1, max_dim=2, floor=0):
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    if isinstance(img_size, int):
        img_size = [img_size]
    elif isinstance(img_size, (list, tuple)):
        img_size = list(img_size)
    elif isinstance(img_size, str):
        img_size = [int(img_size) if img_size.isnumeric() else eval(img_size)]
    else:
        raise TypeError(
            f'"img_size = {img_size}" is of invalid type {type(img_size).__name__}.'
            f'Valid img_size types are int i.e "img_size = 640" or list i.e "img_size = [640, 640]"'
        )

    if len(img_size) > max_dim:
        message = (
            '"train" and "val" img_size must be an integer, while "predict" and "export" img_size may be a [h, w] list'
            'or an integer, i.e. "vajra export img_size=640,480" or "vajra export img_size=640"'
        )

        if max_dim != 1:
            raise ValueError(f'img_size = {img_size} is not a valid image size. {message}')
        LOGGER.warning(f'WARNING! updating to "img_size={max(img_size)}". {message}')
        img_size = [max(img_size)]
    size = [max(math.ceil(x / stride) * stride, floor) for x in img_size]

    if size != img_size:
        LOGGER.warning(f'WARNING! img_size={img_size} must be a multiple of max stride {stride}, updating to {size}')

    size = [size[0], size[0]] if min_dim == 2 and len(size) == 1 else size[0] if min_dim == 1 and len(size) == 1 else size

    return size

def check_latest_pypi_version(package_name="vajra"):
    with contextlib.suppress(Exception):
        requests.packages.urllib3.disable_warnings()
        response = requests.get(f"https://pypi.org/pypi/{package_name}.json", timeout=3)
        if response.status_code == 200:
            return response.json()["info"]["version"]

def check_pip_update_available():
    if ONLINE and is_pip_package():
        with contextlib.suppress(Exception):
            from vajra import __version__
            latest = check_latest_pypi_version()
            if check_version(__version__, f"<{latest}>"):
                LOGGER.info(
                    f'New https://pypi.org/project/vajra/{latest} available !'
                    f'Update with "pip install -U vajra"'
                )
                return True
    return False

def check_python(min: str = "3.8.0") -> bool:
    return check_version(PYTHON_VERSION, min, name="Python ", strict=True)

def check_suffix(file='vajra-v1-nano.pt', suffix='.pt', msg=""):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            path_suffix = Path(f).suffix.lower().strip()
            if len(path_suffix):
                assert path_suffix in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {path_suffix}"

def check_file(file, suffix="", download=True, strict=True):
    check_suffix(file, suffix)
    file = str(file).strip()
    if (not file or ("://" not in file and Path(file).exists()) or file.lower().startswith('grpc://')):
        return file
    
    elif download and file.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
        url = file
        file = url2file(file)
        if Path(file).exists():
            LOGGER.info(f'Found {clean_url(url)} locally at {file}')
        
        else:
            downloads.safe_download(url=url, file=file, unzip=False)

        return file
    else:
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))

        if not files and strict:
            raise FileNotFoundError(f'"{file}" does not exist')
        
        elif len(files) > 1 and strict:
            raise FileNotFoundError(f'Multiple files match "{file}", specify exact path: {files}')
        return files[0] if len(files) else [] if strict else file

def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    return check_file(file, suffix, hard=hard)

def check_model_file_from_stem(model='vajra-v1-nano'):
    if model and not Path(model).suffix and Path(model).stem in downloads.GITHUB_ASSETS_STEMS:
        return Path(model).with_suffix(".pt")
    else:
        return model

def check_is_path_safe(basedir, path):
    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()

    return path_resolved.is_file() and path_resolved.parts[: len(base_dir_resolved.parts)] == base_dir_resolved.parts

def check_imshow(warn=False):
    try:
        if LINUX:
            assert "DISPLAY" in os.environ and not is_docker() and not is_colab() and not is_kaggle()
        cv2.imshow("test", np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"WARNING! Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False

def collect_systems_info():
    import psutil
    from vajra.utils import ENVIRONMENT, is_git_dir
    from vajra.utils.torch_utils import get_cpu_info

    ram_info = psutil.virtual_memory().total / (1024 ** 3)
    check_vajra()
    LOGGER.info(
        f'\n{"OS":<20}{platform.platform()}\n'
        f'{"Environment":<20}{ENVIRONMENT}\n'
        f'{"Python":<20}{PYTHON_VERSION}\n'
        f'{"Install":<20}{"git" if is_git_dir() else "pip" if is_pip_package() else "other"}\n'
        f'{"RAM":<20}{ram_info:.2f} GB\n'
        f'{"CPU":<20}{get_cpu_info()}\n'
        f'{"CUDA":<20}{torch.version.cuda if torch and torch.cuda.is_available() else None}\n'
    )

    for r in parse_requirements(package='vajra'):
        try:
            current = metadata.version(r.name)
            is_met = "Yes " if check_version(current, str(r.specifier), strict=True) else "No "
        except metadata.PackageNotFoundError:
            current = "(not installed)"
            is_met = "No "
        LOGGER.info(f"{r.name:<20}{is_met}{current}{r.specifier}")
    
    if is_github_action_running():
        LOGGER.info(
            f"\nRUNNER_OS: {os.getenv('RUNNER_OS')}\n"
            f"GITHUB_EVENT_NAME: {os.getenv('GITHUB_EVENT_NAME')}\n"
            f"GITHUB_WORKFLOW: {os.getenv('GITHUB_WORKFLOW')}\n"
            f"GITHUB_ACTOR: {os.getenv('GITHUB_ACTOR')}\n"
            f"GITHUB_REPOSITORY: {os.getenv('GITHUB_REPOSITORY')}\n"
            f"GITHUB_REPOSITORY_OWNER: {os.getenv('GITHUB_REPOSITORY_OWNER')}\n"
        )

def check_vajra(verbose=True, device=""):
    import psutil
    from vajra.utils.torch_utils import select_device

    if is_jupyter():
        if check_requirements("wandb", install=False):
            os.system("pip uninstall -y wandb")
        if is_colab():
            shutil.rmtree("sample_data", ignore_errors=True)

    if verbose:
        gib = 1 << 30
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        s = f"({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)"
        with contextlib.suppress(Exception):  # clear display if ipython is installed
            from IPython import display

            display.clear_output()
    else:
        s = ""

    select_device(device=device, newline=False)
    LOGGER.info(f'Setup complete! {s}')

def check_amp(model):
    from vajra.utils.torch_utils import autocast
    
    device = next(model.parameters()).device
    if device.type in ("cpu", "mps"):
        return False

    def amp_allclose(model, img):
        fp32 = model(img, device = device, verbose=False)[0].boxes.data
        with autocast(True):
            amp = model(img, device=device, verbose=False)[0].boxes.data
        del model
        
        return fp32.shape == amp.shape and torch.allclose(fp32, amp.float(), atol=0.5)
    
    img = ASSETS / "bus.jpg"
    prefix = colorstr("AMP:")
    LOGGER.info(f'{prefix} running Automatic Mixed Precision (AMP) checks with Vajra-v1-nano...')
    warning_msg = 'Setting "amp=True". If you experience zero-mAP or NaN losses you can disable AMP with amp=False'

    try:
        from vajra import Vajra

        assert amp_allclose(model=Vajra(model_name="vajra-v1-nano-det"), img=img)
        LOGGER.info(f'{prefix} checks passed !')
    
    except ConnectionError:
        LOGGER.warning(f'{prefix} checks skipped, offline and unable to download Vajra-v1-nano. {warning_msg}')
    
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f'{prefix} Checks skipped! '
            #f'WARNING! {e}'
            f'Unable to load Vajra-v1-nano due to possible Vajra package modifications. {warning_msg}'
        )
    
    except AssertionError:
        LOGGER.warning(
            f'{prefix} checks failed! Anomalies were detected with AMP on your system that may lead to'
            f"NaN losses or zero-mAP results, so AMP will be disabled during training"
        )
        return False

    return True

def git_describe(path=ROOT):
    with contextlib.suppress(Exception):
        return subprocess.check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]
    return ""

def cuda_device_count() -> int:
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'], encoding="utf-8"
        )
        first_line = output.strip().split("\n")[0]
        return int(first_line)

    except(subprocess.CalledProcessError, FileNotFoundError):
        return 0

def cuda_is_available() -> bool:
    return cuda_device_count() > 0

def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    def strip_auth(v):
        return clean_url(v) if (isinstance(v, str) and v.startswith("http") and len(v) > 100) else v

    x = inspect.currentframe().f_back
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    string = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(string) + ", ".join(f"{k} = {strip_auth(v)}" for k, v in args.items()))

IS_PYTHON_3_12 = PYTHON_VERSION.startswith("3.12")