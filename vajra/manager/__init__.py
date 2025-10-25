# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import sys
import subprocess
from typing import List
from pathlib import Path
from vajra.utils import (
    ASSETS, 
    HYPERPARAMS_CFG_DICT,
    HYPERPARAMS_CFG,
    HYPERPARAMS_CFG_PATH,
    LOGGER,
    RANK,
    RUNS_DIR,
    ROOT,
    SETTINGS,
    SETTINGS_YAML,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    SimpleNamespace,
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

SOLUTION_MAP = {
    "count": "ObjectCounter",
    "crop" : "ObjectCropper",
    "blur" : "ObjectBlurrer",
    "workout": "AIGym",
    "heatmap": "Heatmap",
    "isegment": "InstanceSegmentation",
    "visioneye": "VisionEye",
    "detection_depth": "DetectionDepthSolution",
    "speed": "SpeedEstimator",
    "queue": "QueueManager",
    "analytics": "Analytics",
    "inference": "Inference",
    "trackzone": "TrackZone",
    "help": None
}

ARGV = sys.argv or ["", ""]
SOLUTIONS_HELP_MSG = f"""
    Arguments received: {str(["vajra"] + ARGV[1:])}. Vayuvahana 'vajra solutions' usage overview:

        vajra solutions SOLUTION ARGS

        Where SOLUTION (optional) is one of {list(SOLUTION_MAP.keys())[:-1]}
            ARGS (optional) are any number of custom 'arg=value' pairs like 'show_in=True' that override defaults
    
    1. Call object counting solution
        vajra solutions count source="path/to/video.mp4" region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
    
    2. Call heatmaps solution
        vajra solutions heatmap colormap=cv2.COLORMAP_PARULA model=vajra-v1-nano-det.pt

    3. Call queue management solution
        vajra solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]" model=vajra-v1-nano-det.pt

    4. Call workouts monitoring solution for push-ups
        vajra solutions workout model=vajra-v1-nano-pose.pt kpts=[6, 8, 10]

    5. Generate analytical graphs
        vajra solutions analytics analytics_type="pie"

    6. Track objects within specific zones
        vajra solutions trackzone source="path/to/video.mp4" region="[(150, 150), (1130, 150), (1130, 570), (150, 570)]"

    7. Streamlit real-time webcam inference GUI
        vajra streamlit-predict
    
    8. Object Detection and Monocular Depth Estimation (using Apple's Depth Pro model)
        vajra solutions detection_depth source="path/to/video.mp4" model=vajra-v1-nano-det.pt
"""

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

def get_save_dir(args: SimpleNamespace, name: str = None) -> Path:
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir

    else:
        from vajra.utils.files import increment_path

        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir).resolve()

def handle_solutions(args: List[str]) -> None:
    from vajra.solutions.config import SolutionConfig
    full_args_dict = vars(SolutionConfig())
    overrides = {}

    for arg in merge_equals_args(args):
        arg = arg.lstrip("-").rstrip(",")
        if "=" in arg:
            try:
                k, v = parse_key_value_pair(arg)
                overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {arg: ""}, e)
        elif arg in full_args_dict and isinstance(full_args_dict.get(arg), bool):
            overrides[arg] = True
    check_dict_alignment(full_args_dict, overrides)

    if not args:
        LOGGER.warning("No solution name provided. i.e `vajra solutions count`. Defaulting to 'count'.")
        args = ["count"]
    if args[0] == "help":
        LOGGER.info(SOLUTIONS_HELP_MSG)
        return
    elif args[0] in SOLUTION_MAP:
        solution_name = args.pop(0)
    else:
        LOGGER.warning(
            f"❌ '{args[0]}' is not a valid solution. Defaulting to 'count'.\n"
            f"Available solutions: {', '.join(list(SOLUTION_MAP.keys())[:-1])}\n"
        )
        solution_name = "count"
    
    if solution_name == "inference":
        checks.check_requirements("streamlit>=1.29.0")
        LOGGER.info("Loading Vayuvahana Technologies live inference app...")
        subprocess.run(
            [
                "streamlit",
                "run",
                str(ROOT / "solutions/streamlit_inference.py"),
                "--server.headless",
                "true",
                overrides.pop("model", "vajra-v1-nano-det.pt"),
            ]
        )
    else:
        import cv2
        from vajra import solutions

        solution = getattr(solutions, SOLUTION_MAP[solution_name])(is_cli=True, **overrides)
        cap = cv2.VideoCapture(solution.CFG["source"])
        if solution_name != "crop":
            w, h, fps = (
                int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
            )

            if solution_name == "analytics":
                w, h = 1080, 720
            save_dir = get_save_dir(SimpleNamespace(projct="runs/solutions", name="exp", exist_ok=False))
            save_dir.mkdir(parents=True)
            vw = cv2.VideoWriter(str(save_dir / f"{solution_name}.avi"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        
        try:
            f_n = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                results = solution(frame, f_n := f_n + 1) if solution_name == "analytics" else solution(frame)
                if solution_name != "crop":
                    vw.write(results.plot_im)
                if solution.CFG["show"] and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()


def manage(debug=""):
    args = (debug.split(" ") if debug else sys.argv)[1:]
    #LOGGER.info(f"\nDEBUG: Printing args: {args}\n")
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
        "solutions": lambda: handle_solutions(args[1:]),
    }

    full_args_dict = {**HYPERPARAMS_CFG_DICT, **{k: None for k in tasks}, **{k: None for k in modes}, **special}

    special.update({k[0]: v for k, v in special.items()})
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})
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
                if k == "hyp-config" and v is not None:
                    model_configuration = {k: val for k, val in yaml_load(checks.check_yaml(v)).items() if k != "hyp-config"}
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
        model = "vajra-v1-nano-det.pt"
        LOGGER.warning(f"WARNING! 'model' argument is missing. Using default 'model={model}'.")
    model_configuration["model"] = model
    stem = Path(model).stem.lower()
    if "sam_" in stem or "sam2_" in stem or "sam2.1_" in stem:
        from vajra import SAM
        model = SAM(model)

    elif "fastsam" in stem:
        from vajra import FastSAM
        model = FastSAM(model)

    #elif "vajra" in stem and "deyo" in stem:
        #from vajra import VajraDEYO
        #model = VajraDEYO(model, task="detect")

    else:
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
            LOGGER.warning(f"WARNING! 'data' argument is missing. Using default 'data={model_configuration['data']}'.")
    elif mode == "export":
        if "format" not in model_configuration:
            model_configuration["format"] = HYPERPARAMS_CFG.format or "torchscript"
            LOGGER.warning(f"WARNING! 'format' argument is missing. Using default 'format={model_configuration['format']}'.")

    getattr(model, mode)(**model_configuration)

if __name__ == "__main__":
    manage(debug="")