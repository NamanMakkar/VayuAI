from vajra import Vajra
from vajra.configs import get_config, get_save_dir
import subprocess
import numpy as np
import torch
from pathlib import Path
from vajra.utils import LOGGER


scales = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
fitness_list = {}
val_dfl_list = {}
i = 0
for scale in scales:
    LOGGER.info(f"\nITERATION {i}; scale: {scale}\n")
    train_args = {
            "model": "vajra-v1-medium-det",
            "data": "coco.yaml",
            "project": "COCO_Scale_Tune_Twelve_Eps",
            "name": "vajra-v1-medium-det-manual-tune",
            "batch": 54,
            "img_size": 640,
            "patience": 100,
            "epochs": 600,
            "stop_epoch": 12,
            "optimizer": "SGD",
            "lr0": 0.01,
            "device": 0,
            "seed": 0,
            "scale": scale,
            "deterministic": True,
            "copy_paste": 0.2,
            "mixup": 0.0,
        }
    save_dir = "COCO_Scale_Tune_Twelve_Eps/vajra-v1-medium-det-manual-tune" if i==0 else f"COCO_Scale_Tune_Twelve_Eps/vajra-v1-medium-det-manual-tune{i+1}"
    weights_dir = Path(f"{save_dir}/weights")
    cmd = ["vajra", "train", *(f"{k}={v}" for k, v in train_args.items())]
    return_code = subprocess.run(cmd, check=True).returncode
    checkpt_file = weights_dir / ("best-vajra-v1-medium-det.pt" if (weights_dir / "best-vajra-v1-medium-det.pt").exists() else "last-vajra-v1-medium-det.pt")
    metrics = torch.load(checkpt_file)["train_metrics"]
    val_dfl_loss = metrics["val/dfl_loss"]
    val_dfl_list[f"scale_{scale}"] = val_dfl_loss
    fitness = metrics["fitness"]
    fitness_list[f"scale_{scale}"] = fitness
    i += 1
    LOGGER.info(f"\nFitness: {fitness}\n")
    LOGGER.info(f"\nVal DFL Loss: {val_dfl_loss}\n")
    LOGGER.info(f"\nITERATION {i - 1}: scale: {scale}\n")

max_fitness = max(fitness_list, key=fitness_list.get)
min_val_dfl = min(val_dfl_list, key=val_dfl_list.get)
LOGGER.info(f"\nKey for min dfl_loss: {min_val_dfl}; Val DFL loss: {val_dfl_list[min_val_dfl]}; Fitness: {fitness_list[min_val_dfl]}\n")
LOGGER.info(f"\nKey for max fitness: {max_fitness}; Val DFL loss: {val_dfl_list[max_fitness]}; Fitness: {fitness_list[max_fitness]}\n") 
for (k, v) in val_dfl_list.items():
    LOGGER.info(f"\n{k}: val dfl loss: {v}; Val Accuracy: {fitness_list[k]}\n")