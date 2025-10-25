import contextlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union
from vajra.utils import (
    ASSETS, 
    HYPERPARAMS_CFG_DICT,
    HYPERPARAMS_CFG,
    HYPERPARAMS_CFG_PATH,
    LOGGER,
    RANK,
    ROOT,
    SETTINGS,
    SETTINGS_YAML,
    RUNS_DIR,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    __version__,
    colorstr,
    yaml_load,
    yaml_print,
    deprecation_warn
)

modes = {'train', 'val', 'predict', 'export', 'track'}
tasks = {'detect', 'segment', 'classify', 'multilabel_classify', 'pose', 'obb', 'small_obj_detect'}

models_for_tasks = {'detect' : 'vajra-v1-nano-det.pt',
                    'small_obj_detect': 'vajra-v1-nano-pose.pt',
                    'segment' : 'vajra-v1-nano-seg.pt',
                    'classify' : 'vajra-v1-nano-cls.pt',
                    'multilabel_classify': 'vajra-v1-nano-cls.pt',
                    'pose' : 'vajra-v1-nano-pose.pt',
                    'obb': 'vajra-v1-nano-obb.pt'}

metrics_for_tasks = {'detect' : 'metrics/mAP50-95(Box)',
                     'small_obj_detect': 'metrics/mAP50-95(Box)',
                     'segment' : 'metrics/mAP50-95(Mask)',
                     'classify' : 'metrics/accuracy_top1',
                     'multilabel_classify': 'metrics/accuracy_top1',
                     'pose': 'metrics/mAP50-95(Pose)',
                     'obb': 'metrics/mAP50-95(Box)'}

data_for_tasks = {'detect' : 'coco8.yaml',
                  'small_obj_detect': 'coco8.yaml',
                  'segment': 'coco8-seg.yaml',
                  'classify': 'imagenet10',
                  'multilabel_classify': 'imagenet10',
                  'pose': 'coco8-pose.yaml',
                  'obb':'dota8.yaml'}

CLI_HELP_MESSAGE = ' '

CFG_INT_KEYS = {'epochs', 'patience', 'batch', 'workers', 'seed', 'close_mosaic', 
                'mask_ratio', 'max_det', 'vid_stride', 'line_width', 'workspace', 
                'nominal_batch_size', 'save_period'}

CFG_FLOAT_KEYS = {'box', 'cls', 'dfl', 'degrees', 'shear', 'time', 'warmup_epochs'}

CFG_FRACTION_KEYS = {'dropout', 'iou', 'lr0', 'lrf', 'momentum', 'weight_decay', 
                     'warmup_momentum', 'warmup_bias_lr', 'label_smoothing', 
                     'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 
                     'perspective', 'flipud', 'fliplr', 'bgr', 'mosaic', 
                     'mixup', 'copy_paste', 'conf', 'fraction'}

CFG_BOOL_KEYS = {'save', 'exist_ok', 'verbose', 'deterministic', 'single_cls', 
                 'rect', 'cos_lr', 'overlap_mask', 'val', 'save_json', 
                 'save_hybrid', 'half', 'dnn', 'plots', 'show', 'save_txt',
                 'save_conf', 'save_crop', 'save_frames', 'show_labels',
                 'show_conf', 'visualize', 'augment', 'agnostic_nms', 'retina_masks',
                 'show_boxes', 'keras', 'optimize', 'int8', 'dynamic', 'simplify',
                 'nms', 'profile', 'multi_scale'}

MODELS = frozenset({models_for_tasks[task] for task in tasks})


def config_to_dict(config):
    if isinstance(config, (str, Path)):
        #if Path(config).suffix == '.py':
        #    config, _ = Config._file2dict(config)
        #else:
        config = yaml_load(config)
    elif isinstance(config, SimpleNamespace):
        config = vars(config) # Converts to dict

    return config

def get_config(config: Union[str, Path, Dict, SimpleNamespace] = HYPERPARAMS_CFG_DICT, model_configuration: Dict = None):
    config = config_to_dict(config)

    if model_configuration:
        model_configuration = config_to_dict(model_configuration)
        if "save_dir" not in config:
            model_configuration.pop("save_dir", None)
        check_dict_alignment(config, model_configuration)
        config = {**config, **model_configuration}

    for k in "project", "name":
        if k in config and isinstance(config[k], (int, float)):
            config[k] = str(config[k])
    if config.get("name") == "model":
        config["name"] = config.get("model", "").split(".")[0]
        LOGGER.warning(f'WARNING! "name = model" automatically updated to "name = {config["name"]}".')

    check_config(config)

    return IterableSimpleNamespace(**config)

def check_config(config, strict=True):
    for k, v in config.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if strict:
                    raise TypeError(f'{k}={v} is of invalid type {type(v).__name__}'
                                    f'Valid {k} types are either int (i.e {k} = 0) or float (i.e {k} = 0.5)')
                config[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if strict:
                        raise TypeError(f'{k}={v} is of invalid type {type(v).__name__}'
                                        f'Valid {k} types are int (i.e {k} = 0) or float (i.e {k} = 0.5)')
                    config[k] = v = float(v)

                if not (0.0 <= v <= 1.0):
                    raise ValueError(f'{k}={v} is an invalid value'
                                     f'Valid {k} values are between 0.0 and 1.0')
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if strict:
                    raise TypeError(f'{k}={v} is of invalid type {type(v).__name__}'
                                    f'{k} must be an int (i.e {k} = 1)')
                config[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if strict:
                    raise TypeError(f'{k}={v} is of invalid type {type(v).__name__}'
                                    f'{k} must be a bool (i.e {k} = True or {k} = False)')
                config[k] = bool(v)
                

def get_save_dir(args, name=None):
    """Return save_dir as created from train/val/predict arguments."""

    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from vajra.utils.files import increment_path
        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in (-1, 0) else True)

    return Path(save_dir)

def _handle_deprecation(custom):
    """Hardcoded function to handle deprecated config keys."""

    for key in custom.copy().keys():
        if key == "boxes":
            deprecation_warn(key, "show_boxes")
            custom["show_boxes"] = custom.pop("boxes")
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")
            custom["show_labels"] = custom.pop("hide_labels") == "False"
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")
            custom["show_conf"] = custom.pop("hide_conf") == "False"
        if key == "line_thickness":
            deprecation_warn(key, "line_width")
            custom["line_width"] = custom.pop("line_thickness")

    return custom

def check_dict_alignment(base: Dict, custom: Dict, e=None):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list. If
    any mismatched keys are found, the function prints out similar keys from the base list and exits the program.

    Args:
        custom (dict): a dictionary of custom configuration options
        base (dict): a dictionary of base configuration options
        e (Error, optional): An optional error that is passed by the calling function.
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MESSAGE) from e

def copy_hyps_config():
    new_file = Path.cwd() / HYPERPARAMS_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(HYPERPARAMS_CFG_PATH, new_file)
    LOGGER.info(
        f"{HYPERPARAMS_CFG_PATH} copied to {new_file}\n"
        f"Example command with this new config:\n   vajra config={new_file} img_size=640 batch=16"
    )