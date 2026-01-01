# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import re
import os
import cv2
import sys
import numpy as np
import pandas as pd
import subprocess
import yaml
import uuid
import torch
import time
import IPython
import platform
import inspect
import contextlib
import importlib.metadata
import logging
import logging.config
import urllib
import threading
from pathlib import Path
from collections import abc
from itertools import repeat
from typing import Union, Dict, List
from types import SimpleNamespace
from tqdm import tqdm as tqdm_original
import matplotlib.pyplot as plt
from vajra import __version__

__all__ = {
    "functions":[],
    "classes":["TQDM", "StringOps", "IterableSimpleNamespace", "TryExcept", 
               "Retry", "ThreadingLocked", "SettingsManager"],
    "global_variables":["RANK", "LOCAL_RANK", "FILE", 
                        "ROOT", "ASSETS", "DEFAULT_CFG_PATH", 
                        "NUM_THREADS", "AUTOINSTALL", "VERBOSE", 
                        "TQDM_BAR_FORMAT", "LOGGING_NAME", "MACOS",
                        "WINDOWS", "LINUX", "ARM64", "HELP_MSG",
                        "PREFIX", "SETTINGS", "DATASETS_DIR", 
                        "WEIGHTS_DIR", "RUNS_DIR", "ENVIRONMENT",
                        "TESTS_RUNNING"]
}

_imshow = cv2.imshow

RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
HYPERPARAMS_CFG_PATH = ROOT / "configs/hyperparams/default_hyp.yaml"
HYPERPARAMS_DETR_ARGS_PATH = ROOT / "configs/hyperparams/detr_hyp.yaml"
ASSETS = ROOT / "assets"
ASSETS_URL = "https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4"
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
MACOS, WINDOWS, LINUX = (platform.system() == x for x in ["Darwin", "Windows", "Linux"]) # platform booleans
MACOS_VERSION = platform.mac_ver()[0] if MACOS else None
NOT_MACOS14 = not (MACOS and MACOS_VERSION.startswith("14."))
ARM64 = platform.machine() in ("arm64", "aarch64") # arm64 boolean
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = str(torch.__version__) 
TORCHVISION_VERSION = importlib.metadata.version("torchvision")
HELP_MSG = """
            1. Install the Vajra package from GitHub

                git clone https://github.com/NamanMakkar/VayuAI.git
                cd VayuAI/
                pip install .

            2. Use the SDK

                import requests
                from vajra import Vajra
                from PIL import Image

                # Build or Load a model
                model = Vajra('vajra-v1-nano-det') # Build a new model from the config file
                model = Vajra('visdrone-best-vajra-v1-nano-det.pt') # Load a pretrained model

                # Use the model for training, validation, prediction and exporting
                results = model.train(data='coco128.yaml', epochs=3)
                results = model.val()
                result = model(Image.open(requests.get(url, stream=True).raw))
                success = model.export(format='onnx')

            3. Use the CLI

                Vajra 'vajra' CLI commands use the following syntax:

                    vajra TASK MODE ARGS

                    Where TASK (optional) is one of [detection, segmentation, classification], detection is used by default if none provided
                          MODE (required) is one of [train, val, predict, export]
                          ARGS (optional) are any number of custom 'arg=value' pairs that are used for overriding defaults i.e img_size=1280
                    
                    - Run special commands:
                        vajra help
                        vajra checks
                        vajra version
                        vajra settings
                        vajra copy-config
                        vajra config
            
           """
#DATASETS_DIR = Path(os.getenv('VAJRA_DATASETS_DIR', ROOT.parent / 'dataset'))  # global datasets directory
AUTOINSTALL = str(os.getenv('VAJRA_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
VERBOSE = str(os.getenv('VAJRA_VERBOSE', True)).lower() == 'true'  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format
FONT = 'Arial.ttf'

torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['OMP_NUM_THREADS'] = '1' if platform.system() == 'darwin' else str(NUM_THREADS)  # OpenMP (PyTorch and SciPy)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

LOGGING_NAME = "Vajra"
def set_logging(name=LOGGING_NAME, verbose=True):
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR
    formatter = logging.Formatter("%(message)s")

    if WINDOWS and sys.stdout.encoding != "utf-8":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding='utf-8')
            elif hasattr(sys.stdout, "buffer"):
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            else:
                sys.stdout.encoding = "utf-8"
        except Exception as e:
            print(f'Creating custom formatter for non UTF-8 environments due to {e}')

            class CustomFormatter(logging.Formatter):
                def format(self, record):
                    return emojis(super().format(record))
            
            formatter = CustomFormatter("%(message)s")
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate=False
    return logger

LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)
for logger in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)

def yaml_save(file='data.yaml', data=None, header=""):
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

    valid_types = int, float, str, bool, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

_torch_load = torch.load

def torch_load(*args, **kwargs):
    """
    Load a PyTorch model with updated arguments to avoid warnings.

    This function wraps torch.load and adds the 'weights_only' argument for PyTorch 1.13.0+ to prevent warnings.

    Args:
        *args (Any): Variable length argument list to pass to torch.load.
        **kwargs (Any): Arbitrary keyword arguments to pass to torch.load.

    Returns:
        (Any): The loaded PyTorch object.

    Note:
        For PyTorch versions 2.0 and above, this function automatically sets 'weights_only=False'
        if the argument is not provided, to avoid deprecation warnings.
    """
    from vajra.utils.torch_utils import TORCH_1_13

    if TORCH_1_13 and "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return _torch_load(*args, **kwargs)

def read_device_model() -> str:
    """
    Reads the device model information from the system and caches it for quick access. Used by is_jetson() and
    is_raspberrypi().

    Returns:
        (str): Model file contents if read successfully or empty string otherwise.
    """
    with contextlib.suppress(Exception):
        with open("/proc/device-tree/model") as f:
            return f.read()
    return ""

PROC_DEVICE_MODEL = read_device_model()

def is_raspberrypi() -> bool:
    """
    Determines if the Python environment is running on a Raspberry Pi by checking the device model information.

    Returns:
        (bool): True if running on a Raspberry Pi, False otherwise.
    """
    return "Raspberry Pi" in PROC_DEVICE_MODEL


def is_jetson() -> bool:
    """
    Determines if the Python environment is running on a Jetson Nano or Jetson Orin device by checking the device model
    information.

    Returns:
        (bool): True if running on a Jetson Nano or Jetson Orin, False otherwise.
    """
    return "NVIDIA" in PROC_DEVICE_MODEL

IS_JETSON = is_jetson()
IS_RASPBERRY_PI = is_raspberrypi()

def yaml_load(file="data.yaml", append_filename=False):
    assert Path(file).suffix in (".yaml", ".yml"), f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors='ignore', encoding='utf-8') as f:
        string = f.read()
        if not string.isprintable():
            string = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", string)
        data = yaml.safe_load(string) or {}
        if append_filename:
            data["yaml_file"] = str(file)
        return data

def _ntuple(n):
    def parse(x):
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))
    return parse

two_tuple = _ntuple(2)
four_tuple = _ntuple(4)

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('red', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def remove_colorstr(input_string):
    ansi_escape = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)

def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True)
    LOGGER.info(f'Printing "{colorstr("bold", "black", yaml_file)}"\n\n{dump}')

def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

def is_hindi(s='नमस्ते'):
    return bool(re.search('[\u0900-\u097F]', str(s)))

def is_russian(s='привет'):
    # Is string composed of any Russian characters?
    return bool(re.search('[\u0400-\u04FF]', str(s)))

def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return bool(re.search('[\u4e00-\u9fff]', str(s)))

def is_vietnamese(s='xin chào'):
    # Is string composed of any Vietnamese characters?
    # This pattern includes common Vietnamese characters with diacritical marks.
    return bool(re.search('[\u00C0-\u00FF\u0102\u0103\u1EA0-\u1EFF]', str(s)))

def is_japanese(s='こんにちは'):
    # Is string composed of any Japanese characters?
    return bool(re.search('[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FBF]', str(s)))

def is_colab():
    # Is environment a Google Colab instance?
    return 'google.colab' in sys.modules

def is_jupyter():
    with contextlib.suppress(Exception):
        from IPython import get_ipython

        return get_ipython() is not None
    return False
    
def is_notebook():
    # Is environment a Jupyter notebook? Verified on Colab, Jupyterlab, Kaggle, Paperspace
    ipython_type = str(type(IPython.get_ipython()))
    return 'colab' in ipython_type or 'zmqshell' in ipython_type

def is_kaggle():
    # Is environment a Kaggle Notebook?
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'

def is_docker() -> bool:
    """Check if the process runs inside a docker container."""
    if Path("/.dockerenv").exists():
        return True
    try:  # check if docker is in control groups
        with open("/proc/self/cgroup") as file:
            return any("docker" in line for line in file)
    except OSError:
        return False

def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

def is_ubuntu() -> bool:
    with contextlib.suppress(FileNotFoundError):
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    return False

def get_ubuntu_version():
    if is_ubuntu():
        with contextlib.suppress(FileNotFoundError, AttributeError):
            with open("/etc/os-release") as f:
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]
                
def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if not test:
        return os.access(dir, os.W_OK)  # possible issues on Windows
    file = Path(dir) / 'tmp.txt'
    try:
        with open(file, 'w'):  # open file with write permissions
            pass
        file.unlink()  # remove file
        return True
    except OSError:
        return False

def is_online() -> bool:
    import socket
    for host in "1.1.1.1", "8.8.8.8", "223.5.5.5":
        try:
            test_connection = socket.create_connection(address=(host, 53), timeout=2)
        except (socket.timeout, socket.gaierror, OSError):
            continue
        else:
            test_connection.close()
            return True
    return False

ONLINE = is_online()

def is_pip_package(filepath: str = __name__) -> bool:
    import importlib.util
    spec = importlib.util.find_spec(filepath)
    return spec is not None and spec.origin is not None

def is_pytest_runnning():
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(sys.argv[0]).stem)

def is_github_action_running() -> bool:
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ

def is_git_dir():
    return get_git_dir() is not None

def get_git_dir():
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d

def get_git_origin_url():
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            return origin.decode().strip()

def get_git_branch():
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            return origin.decode().strip()


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper

def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib turns :// -> :/, as_posix() for Windows
    return urllib.parse.unquote(url).split("?")[0]  # '%2F' to '/', split https://url.com/file.txt?auth

def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name

def yaml_load(file='default_hyp.yaml', append_filename=False):
    assert Path(file).suffix in ('.yaml', '.yml'), f'Attempted to load non-YAML file {file} with yaml_load()'
    with open(file, errors = "ignore", encoding="utf-8") as f:
        string = f.read()
        if not string.isprintable():
            string = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", string)

        data = yaml.safe_load(string) or {}
        if append_filename:
            data["yaml_file"] = str(file)
        return data

HYPERPARAMS_CFG_DICT = yaml_load(HYPERPARAMS_CFG_PATH)
HYPERPARAMS_DETR_CFG_DICT = yaml_load(HYPERPARAMS_DETR_ARGS_PATH)

class TQDM(tqdm_original):
    def __init__(self, *args, **kwargs) -> None:
        kwargs['disable'] = not VERBOSE or kwargs.get('disable', False)
        kwargs.setdefault('bar_format', TQDM_BAR_FORMAT)
        super().__init__(*args, **kwargs)

class IterableSimpleNamespace(SimpleNamespace):
    def __iter__(self):
        return iter(vars(self).items())

    def __str__(self):
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}', you might need to make some changes to {HYPERPARAMS_CFG_PATH}
            """
        )

    def get(self, key, default=None):
        return getattr(self, key, default)

for k, v in HYPERPARAMS_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        HYPERPARAMS_CFG_DICT[k] = None

HYPERPARAMS_CFG_KEYS = HYPERPARAMS_CFG_DICT.keys()
HYPERPARAMS_DETR_CFG_KEYS = HYPERPARAMS_DETR_CFG_DICT.keys()
HYPERPARAMS_CFG = IterableSimpleNamespace(**HYPERPARAMS_CFG_DICT)
HYPERPARAMS_DETR_CFG = IterableSimpleNamespace(**HYPERPARAMS_DETR_CFG_DICT)

class TryExcept(contextlib.ContextDecorator):
    def __init__(self, msg='', verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True

class ThreadingLocked:
    def __init__(self):
        self.lock = threading.Lock()

    def __call__(self, f):
        from functools import wraps

        @wraps
        def decorated(*args, **kwargs):
            with self.lock:
                return f(*args, **kwargs)

        return decorated

def threaded(func):
    """
    Used for multi-threading a funciton by default and return the thread or function result
    Used as @threaded decorator, function runs in a separate thread until threaded=False is passed
    """

    def wrapper(*args, **kwargs):
        """A wrapper function that multi-threads the target function given a "threaded" kwarg 
           and returns the thread or the function result
        """
        if kwargs.pop('threaded', True):
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)
        
    return wrapper

def join_threads(verbose=False):
    # Join all daemon threads, i.e. atexit.register(lambda: join_threads())
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f'Joining thread {t.name}')
            t.join()

def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable

    """
    return os.access(str(dir_path), os.W_OK)

def is_dist_available_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()

def notebook_init(verbose=True):
    # Check system software and hardware
    print('Checking setup...')

    import os
    import shutil

    from utils.general import check_font, check_requirements, is_colab
    from utils.torch_utils import select_device  # imports

    check_font()

    import psutil
    from IPython import display  # to display images and clear console output

    if is_colab():
        shutil.rmtree('/content/sample_data', ignore_errors=True)  # remove colab /sample_data directory

    # System info
    if verbose:
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        display.clear_output()
        s = f'({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)'
    else:
        s = ''

    select_device(newline=False)
    print(emojis(f'Setup complete! {s}'))
    return display

class StringOps:
    def __str__(self) -> str:
        attributes = []
        for attr in dir(self):
            m = getattr(self, attr)
            if not callable(m) and not attr.startswith("_"):
                if isinstance(m, StringOps):
                    string = f'{attr}: {m.__module__}.{m.__class__.__name__} object'
                else:
                    string = f'{attr}: {repr(m)}'
                attributes.append(string)
        return f'{self.__module__}.{self.__class__.__name__} object with attributes:\n\n' + '\n'.join(attributes)

    def __repr__(self) -> str:
        return self.__str__()
    
    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f'{name} object has no attribute {attr}. The valid attributes are - \n{self.__doc__}')

def get_user_config_dir(sub_dir='Vajra'):
    if WINDOWS:
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:
        path = Path.home() / "Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir

    else:
        raise ValueError(f'Unsupported operating system: {platform.system()}')
    
    if not is_dir_writeable(path.parent):
        LOGGER.warning(f'WARNING! user config directory {path} is not writeable, defaulting to "/tmp" or current working directory'
                        'You can provide a CONFIG_DIR environment variable for this path')
        path = Path('/tmp') / sub_dir if is_dir_writeable('/tmp') else Path().cwd() / sub_dir
    
    path.mkdir(parents=True, exist_ok=True)
    return path

USER_CONFIG_DIR = Path(os.getenv("CONFIG_DIR") or get_user_config_dir())
SETTINGS_YAML = USER_CONFIG_DIR / "settings.yaml"

class SettingsManager(dict):
    def __init__(self, file=SETTINGS_YAML, version="0.0.1"):
        import copy
        import hashlib
        from vajra.checks import check_version
        from vajra.utils.torch_utils import torch_distributed_zero_first

        git_dir = get_git_dir()
        root = git_dir or Path()
        datasets_root = (root.parent if git_dir and is_dir_writeable(root.parent) else root).resolve()
        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,
            "datasets_dir": str(datasets_root / "data"),
            "weights_dir": str(root / "weights"),
            "runs_dir": str(root / "runs"),
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),
            "sync": True,
            "api_key": "",
            "openai_api_key": "",
            "clearml": True,
            "comet": True,
            "dvc": True,
            "mlflow": True,
            "neptune": True,
            "raytune": True,
            "tensorboard":True,
            "wandb": True,
        }
        super().__init__(copy.deepcopy(self.defaults))

        with torch_distributed_zero_first(RANK):
            if not self.file.exists():
                self.save()

            self.load()
            correct_keys = self.keys() == self.defaults.keys()
            correct_types = all(type(a) is type(b) for a, b in zip(self.values(), self.defaults.values()))
            correct_version = check_version(self["settings_version"], self.version)
            help_message = (
                f'\nView settings with "vajra settings" or at {self.file}'
                '\nUpdate settings with "vajra settings key=value", "vajra settings runs_dir=path/to/dir".'
            )

            if not (correct_keys and correct_types and correct_version):
                LOGGER.warning(
                    'WARNING! Vajra settings reset to default values. This may be due to a possible problem'
                    f'with your settings or a recent Vajra package update. {help_message}'
                )
                self.reset()

            if self.get("datasets_dir") == self.get("runs_dir"):
                LOGGER.warning(
                    f'WARNING! Vajra setting "datasets_dir: {self.get("datasets_dir")}"'
                    f'must be different from "runs_dir: {self.get("runs_dir")}"'
                    f'Please change one to avoid possible issues during training. {help_message}'
                )
    
    def load(self):
        super().update(yaml_load(self.file))

    def save(self):
        yaml_save(self.file, dict(self))

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.save()

    def reset(self):
        self.clear()
        self.update(self.defaults)
        self.save()

PREFIX = colorstr("Vayuvahan Technologies Vajra: ")
SETTINGS = SettingsManager()
DATASETS_DIR = Path(SETTINGS["datasets_dir"])
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])
RUNS_DIR = Path(SETTINGS["runs_dir"])

ENVIRONMENT = (
    "Colab"
    if is_colab()
    else "Jupyter"
    if is_jupyter()
    else "Kaggle"
    if is_kaggle()
    else "Docker"
    if is_docker()
    else platform.system()
)

class TryExcept(contextlib.ContextDecorator):
    def __init__(self, message="", verbose=True) -> None:
        self.message = message
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            print(emojis(f'{self.message}{": " if self.message else ""}{value}'))
        return True

class Retry(contextlib.ContextDecorator):
    def __init__(self, times=3, delay=2) -> None:
        self.times = times
        self.delay = delay
        self._attempts = 0

    def __call__(self, func):
        def wrapped_func(*args, **kwargs):
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    print(f'Retry {self._attempts}/{self.times} failed: {e}')
                    if self._attempts >= self.times:
                        raise e
                    time.sleep(self.delay * (2**self._attempts))
        return wrapped_func

    def __enter__(self):
        self._attempts = 0

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self._attempts += 1
            if self._attempts < self.times:
                print(f'Retry {self._attempts}/{self.times} failed: {exc_value}')
                time.sleep(self.delay * (2**self._attempts))
                return True
        return False

def deprecation_warn(arg, new_arg, version=None):
    if not version:
        version = float(__version__[:3]) + 0.2
    LOGGER.warning(
        f"WARNING! '{arg} is deprecated'",
        f"Please use {new_arg} instead"
    )

def string_to_val(string):
    string_lower = string.lower()

    if string_lower == "none":
        return None

    elif string_lower == "true":
        return True

    elif string_lower == "false":
        return False
    
    else:
        try:
            st = eval(string)
            return st
        except Exception:
            return string

def merge_equals_args(args: List[str]) -> List[str]:
    new_args = []
    current = ""
    i = 0
    depth = 0
    while i < len(args):
        arg = args[i]
        if arg == "=" and 0 < i < len(args) - 1:
            new_args[-1] += f"={args[i+1]}"
            i += 2
            continue
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i+1]:
            new_args.append(f"{arg}{args[i+1]}")
            i += 2
            continue
        elif arg.startswith("=") and i > 0:
            new_args[-1] += arg
            i += 1
            continue

        depth += arg.count("[") - arg.count("]")
        current += arg

        if depth == 0:
            new_args.append(current)
            current = ""

        i += 1
    
    if current:
        new_args.append(current)

    return new_args

def parse_key_value_pair(pair):
    k, v = pair.split("=", 1)
    k, v = k.strip(), v.strip()
    assert v, f"missing '{k}' value"
    return k, string_to_val(v)

def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)

def imwrite(filename: str, img: np.ndarray, params=None):
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False

def imshow(winname: str, mat: np.ndarray):
    _imshow(winname.encode("unicode_escape").decode(), mat)

_torch_save = torch.save

def torch_save(*args, use_dill=True, **kwargs):
    try:
        assert use_dill
        import dill as pickle
    except (AssertionError, ImportError):
        import pickle

    if "pickle_module" not in kwargs:
        kwargs["pickle_module"] = pickle

    for i in range(4):
        try:
            return _torch_save(*args, **kwargs)
        except RuntimeError as e:
            if i==3:
                raise e
            time.sleep((2**i) / 2)

torch.save = torch_save
torch.load = torch_load
if WINDOWS:
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow

TESTS_RUNNING = is_pytest_runnning() or is_github_action_running()