# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import random
import shutil
import subprocess
import time

import numpy as np
import torch

from vajra.configs import get_config, get_save_dir
from vajra.utils import LOGGER, HYPERPARAMS_CFG, colorstr, remove_colorstr, yaml_print, yaml_save
from vajra.plotting import plot_tune_results
from vajra import callbacks


class Tuner:
    def __init__(self, args=HYPERPARAMS_CFG, _callbacks=None) -> None:
        self.space = args.pop("space", None) or {
            "lr0": (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
            "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
            "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
            "box": (1.0, 20.0),  # box loss gain
            "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
            "dfl": (0.4, 6.0),  # dfl loss gain
            "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (0.0, 45.0),  # image rotation (+/- deg)
            "translate": (0.0, 0.9),  # image translation (+/- fraction)
            "scale": (0.0, 0.95),  # image scale (+/- gain)
            "shear": (0.0, 10.0),  # image shear (+/- deg)
            "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (0.0, 1.0),  # image flip left-right (probability)
            "bgr": (0.0, 1.0),  # image channel bgr (probability)
            "mosaic": (0.0, 1.0),  # image mixup (probability)
            "mixup": (0.0, 1.0),  # image mixup (probability)
            "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)
        }
        self.args = get_config(model_configuration=args)
        self.tune_dir = get_save_dir(self.args, name="tune")
        self.tune_csv = self.tune_dir / "tune_results.csv"
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.prefix = colorstr("Tuner: ")
        callbacks.add_integration_callbacks(self)
        LOGGER.info(
            f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'"
        )

    def _mutate(self, parent="single", n=5, mutation=0.8, sigma=0.2):
        if self.tune_csv.exists():
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            fitness = x[:, 0]
            n = min(n, len(x))
            x = x[np.argsort(-fitness)][:n]
            w = x[:, 0] - x[:, 0].min() + 1e-6
            if parent == "single" or len(x) == 1:
                x = x[random.choices(range(n), weights=w)[0]]
            elif parent == "weighted":
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()
            
            r = np.random
            r.seed(int(time.time()))
            g = np.array([v[2] if len(v) == 3 else 1.0 for k, v in self.space.items()])
            ng = len(self.space)
            v = np.ones(ng)

            while all(v == 1):
                v = (g * (r.random(ng) < mutation) * r.randn(ng) * r.random() * sigma + 1).clip(0.3, 3.0)
            
            hyp = {k: float(x[i + 1] * v[i]) for i, k in enumerate(self.space.keys())}
        else:
            hyp = {k: getattr(self.args, k) for k in self.space.keys()}

        for k, v in self.space.items():
            hyp[k] = max(hyp[k], v[0])
            hyp[k] = min(hyp[k], v[1])
            hyp[k] = round(hyp[k], 5)

        return hyp

    def __call__(self, model=None, iterations=10, cleanup=True):
        t0 = time.time()
        best_save_dir, best_metrics = None, None
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)

        for i in range(iterations):
            mutated_hyp = self._mutate()
            LOGGER.info(f"{self.prefix}Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}")

            metrics = {}
            train_args = {**vars(self.args), **mutated_hyp}
            save_dir = get_save_dir(get_config(train_args))
            weights_dir = save_dir / "weights"
            try:
                cmd = ["vajra", "train", *(f"{k}={v}" for k, v in train_args.items())]
                LOGGER.info(f"\nDebugging command: {cmd}\n")
                return_code = subprocess.run(cmd, check=True).returncode
                checkpt_file = weights_dir / ("best-vajra-v1-medium-det.pt" if (weights_dir / "best-vajra-v1-medium-det.pt").exists() else "last-vajra-v1-medium-det.pt")
                metrics = torch.load(checkpt_file)["train_metrics"]
                assert return_code == 0, "training failed"
            except Exception as e:
                LOGGER.warning(f"WARNING! training failure for hyperparameter tuning iteration {i + 1}\n{e}")
            
            fitness = metrics.get("fitness", 0.0)
            log_row = [round(fitness, 5)] + [mutated_hyp[k] for k in self.space.keys()]
            headers = "" if self.tune_csv.exists() else (",".join(["fitness"] + list(self.space.keys())) + "\n")

            with open(self.tune_csv, "a") as f:
                f.write(headers + ",".join(map(str, log_row)) + "\n")
            
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            fitness = x[:, 0]
            best_idx = fitness.argmax()
            best_is_current = best_idx == i

            if best_is_current:
                best_save_dir = save_dir
                best_metrics = {k: round(v, 5) for k, v in metrics.items()}
                for checkpt in weights_dir.glob("*.pt"):
                    shutil.copy2(checkpt, self.tune_dir / "weights")
            elif cleanup:
                shutil.rmtree(checkpt_file.parent)
            
            plot_tune_results(self.tune_csv)
            header = (
                f'{self.prefix}{i + 1}/{iterations} iterations complete! ({time.time() - t0:.2f}s)\n'
                f'{self.prefix}Results saved to {colorstr("bold", self.tune_dir)}\n'
                f'{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}\n'
                f'{self.prefix}Best fitness metrics are {best_metrics}\n'
                f'{self.prefix}Best fitness model is {best_save_dir}\n'
                f'{self.prefix}Best fitness hyperparameters are printed below.\n'
            )
            LOGGER.info("\n" + header)

            data = {k: float(x[best_idx, i+1]) for i, k in enumerate(self.space.keys())}
            yaml_save(
                self.tune_dir / "best_hyperparameters.yaml",
                data=data,
                header=remove_colorstr(header.replace(self.prefix, "# ")) + "\n",
            )
            yaml_print(self.tune_dir / "best_hyperparameters.yaml")