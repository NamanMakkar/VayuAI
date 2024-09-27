# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import os
import sys
import shutil
import socket
import json
import tempfile
from vajra.utils import USER_CONFIG_DIR, ROOT, HYPERPARAMS_CFG_DICT, LOGGER
from vajra.utils.torch_utils import TORCH_1_9

def find_free_network_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def generate_ddp_file(trainer):
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)
    #LOGGER.info(f"{trainer.args}\n")
    #LOGGER.info(f"Trainer model: {trainer.args.model}\n")
    content = f"""
# Vajra Multi-GPU training temp file (should be automatically deleted after use)
model_configuration = {vars(trainer.args)}

if __name__ == "__main__":
    from {module} import {name}
    from vajra.utils import HYPERPARAMS_CFG_DICT

    config = HYPERPARAMS_CFG_DICT.copy()
    config.update(save_dir='')
    trainer = {name}(config=config, model_configuration=model_configuration)
    trainer.args.model = "{trainer.args.model}"
    results = trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name

def generate_ddp_command(world_size, trainer):
    import __main__

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)

    file = generate_ddp_file(trainer)
    #trainer_args_json = json.dumps(vars(trainer.args))    "--trainer_args", f"'{trainer_args_json}'"
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [sys.executable, "-m", dist_cmd, "--nproc_per_node", f"{world_size}", "--master_port", f"{port}", file]
    return cmd, file

def ddp_cleanup(trainer, file):
    if f"{id(trainer)}.py" in file:
        os.remove(file)