from vajra import Vajra
from vajra.configs import get_config, get_save_dir
import subprocess
import numpy as np
import torch
from pathlib import Path
from vajra.utils import LOGGER


copy_paste_hparams = np.arange(0., 0.41, 0.05)
mixup_hparams = np.arange(0., 0.3, 0.05)
fitness_list = {}
val_dfl_list = {}
i = 0
for cp in copy_paste_hparams:
    for mxp in mixup_hparams:
        LOGGER.info(f"\nITERATION {i}; copy_paste: {cp}; mixup: {mxp}\n")
        train_args = {
            "model": "vajra-v1-medium-det",
            "data": "coco.yaml",
            "project": "COCO_Manual_Tuning_SixEps",
            "name": "vajra-v1-medium-det-manual-tune",
            "batch": 120,
            "img_size": 640,
            "patience": 100,
            "epochs": 600,
            "stop_epoch": 7,
            "optimizer": "SGD",
            "lr0": 0.01,
            "device": 0,
            "seed": 0,
            "deterministic": True,
            "copy_paste": cp,
            "mixup": mxp,
        }
        save_dir = "COCO_Manual_Tuning_SixEps/vajra-v1-medium-det-manual-tune" if i==0 else f"COCO_Manual_Tuning_SixEps/vajra-v1-medium-det-manual-tune{i+1}"
        weights_dir = Path(f"{save_dir}/weights")
        cmd = ["vajra", "train", *(f"{k}={v}" for k, v in train_args.items())]
        return_code = subprocess.run(cmd, check=True).returncode
        checkpt_file = weights_dir / ("best-vajra-v1-medium-det.pt" if (weights_dir / "best-vajra-v1-medium-det.pt").exists() else "last-vajra-v1-medium-det.pt")
        metrics = torch.load(checkpt_file)["train_metrics"]
        val_dfl_loss = metrics["val/dfl_loss"]
        val_dfl_list[f"cp_{cp}_mxp_{mxp}"] = val_dfl_loss
        fitness = metrics["fitness"]
        fitness_list[f"cp_{cp}_mxp_{mxp}"] = fitness
        i += 1
        LOGGER.info(f"\nFitness: {fitness}\n")
        LOGGER.info(f"\nVal DFL Loss: {val_dfl_loss}\n")
        LOGGER.info(f"\nITERATION {i - 1}: copy_paste: {cp}; mixup: {mxp}\n")

max_fitness = max(fitness_list, key=fitness_list.get)
min_val_dfl = min(val_dfl_list, key=val_dfl_list.get)
LOGGER.info(f"\nKey for min dfl_loss: {min_val_dfl}; Val DFL loss: {val_dfl_list[min_val_dfl]}; Fitness: {fitness_list[min_val_dfl]}\n")
LOGGER.info(f"\nKey for max fitness: {max_fitness}; Val DFL loss: {val_dfl_list[max_fitness]}; Fitness: {fitness_list[max_fitness]}\n") 
for (k, v) in val_dfl_list.items():
    LOGGER.info(f"\n{k}: val dfl loss: {v}; Val Accuracy: {fitness_list[k]}\n")