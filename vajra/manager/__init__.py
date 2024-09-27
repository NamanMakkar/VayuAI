# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import sys
from typing import List
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
    TESTS_RUNNING,
    IterableSimpleNamespace,
    __version__,
    colorstr,
    yaml_load,
    yaml_print,
    deprecation_warn,
    parse_key_value_pair,
    merge_equals_args
)
from vajra.configs import CLI_HELP_MESSAGE, tasks, data_for_tasks, metrics_for_tasks, models_for_tasks, check_dict_alignment, copy_hyps_config, modes
from vajra import Vajra
from vajra import checks


def handle_settings(args: List[str]) -> None:
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_YAML.unlink()
                SETTINGS.reset()
                LOGGER.info("Settings reset sucessfully")
            else:
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)
        yaml_print(SETTINGS_YAML)
    except Exception as e:
        LOGGER.warning(f"WARNING! Settings error: '{e}'.")

def manage(debug=""):
    args = (debug.split(" ") if debug else sys.argv)[1:]

    if not args:
        LOGGER.info(CLI_HELP_MESSAGE)
        return
    
    special = {
        "help": lambda: LOGGER.info(CLI_HELP_MESSAGE),
        "checks": checks.collect_systems_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_settings(args[1:]),
        "hyp-config": lambda: yaml_print(HYPERPARAMS_CFG_PATH),
        "copy-hyp": copy_hyps_config,
    }

    full_args_dict = {**HYPERPARAMS_CFG_DICT, **{k: None for k in tasks}, **{k: None for k in modes}, **special}

    special.update({k[0]: v for k, v in special.items()})
    special.update({k[:-1]: v for k, v in special.items()})
    special = {**special, **{f"-{k}":v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    model_configuration = {}

    for a in merge_equals_args(args):
        if a.startswith("--"):
            LOGGER.warning(f"WARNING! argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'. ")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"WARNING! argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "config" and v is not None:
                    LOGGER.info(f"Overriding {HYPERPARAMS_CFG_PATH} with {v}")
                    model_configuration = {k: val for k, val in yaml_load(checks.check_yaml(v)).items() if k != "config"}
                else:
                    model_configuration[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)
        elif a in tasks:
            model_configuration["task"] = a
        elif a in modes:
            model_configuration["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in HYPERPARAMS_CFG_DICT and isinstance(HYPERPARAMS_CFG_DICT[a], bool):
            model_configuration[a] = True
        elif a in HYPERPARAMS_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={HYPERPARAMS_CFG_DICT[a]}'\n{CLI_HELP_MESSAGE}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})
    
    check_dict_alignment(full_args_dict, model_configuration)
    mode = model_configuration.get("mode")
    if mode is None:
        mode = HYPERPARAMS_CFG.mode or "predict"
        LOGGER.warning(f"WARNING! 'mode' argument is missing. Valid modes are {modes}. Using default 'mode={mode}'.")
    elif mode not in modes:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {modes}.\n{CLI_HELP_MESSAGE}")
    
    task = model_configuration.pop("task", None)
    if task:
        if task not in tasks:
            raise ValueError(f"Invalid 'task={task}'. Valid tasks are {tasks}.\n{CLI_HELP_MESSAGE}")
        if "model" not in model_configuration:
            model_configuration["model"] = models_for_tasks[task]
    
    model = model_configuration.pop("model", HYPERPARAMS_CFG.model)
    if model is None:
        model = "vajra-v1-det-nano.pt"
        LOGGER.warning(f"WARNING! 'model' argument is missing. Using default 'model={model}'.")
    model_configuration["model"] = model
    stem = model.lower()

    from vajra import Vajra
    model = Vajra(model, task=task)
    if isinstance(model_configuration.get("pretrained"), str):
        model.load(model_configuration["pretrained"])
    
    if task != model.task:
        if task:
            LOGGER.warning(
                f"WARNING! conflicting 'task={task}' passed with 'task={model.task}' model. "
                f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model."
            )
        task = model.task

    if mode in ("predict", "track") and "source" not in model_configuration:
        model_configuration["source"] = HYPERPARAMS_CFG.source or ASSETS
        LOGGER.warning(f"WARNING! 'source' argument is missing. Using default 'source={model_configuration['source']}'.")
    elif mode in ("train", "val"):
        if "data" not in model_configuration and "resume" not in model_configuration:
            model_configuration["data"] = HYPERPARAMS_CFG.data or data_for_tasks.get(task or HYPERPARAMS_CFG.task, HYPERPARAMS_CFG.data)
            LOGGER.warning(f"WARNING! 'source' argument is missing. Using default 'source={model_configuration['source']}'.")
        
        elif mode in ("train", "val"):
            if "data" not in model_configuration and "resume" not in model_configuration:
                model_configuration["data"] = HYPERPARAMS_CFG.data or data_for_tasks.get(task or HYPERPARAMS_CFG.task, HYPERPARAMS_CFG.data)
                LOGGER.warning(f"WARNING! 'data' argument is missing. Using default 'data={model_configuration['data']}'.")
        elif mode == "export":
            if "format" not in model_configuration:
                model_configuration["format"] = HYPERPARAMS_CFG.format or "torchscript"
                LOGGER.warning(f"WARNING! 'format' argument is missing. Using default 'format={model_configuration['format']}'.")

        getattr(model, mode)(**model_configuration)

        LOGGER.info(f"Learn more at: ")

if __name__ == "__main__":
    manage(debug="")