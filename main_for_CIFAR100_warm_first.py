import json
import os
import subprocess
import copy
import sys
from concurrent.futures import ProcessPoolExecutor
import time
import logging
# -------------------------------- SETTINGS ------------------------------
VERSION = ""
CONFIG_FILE = "exps/grail_CIFAR100.json"
SEEDS = [1993, 2017, 2020]
INCREMENTS= [(50),(40)]
GPUS = [3,4,5]
MAX_CONCURRENT_PROCESSES = len(GPUS)

Description = "final"
# --------------------------RECOMMEND PARAMS FOR EACH TASKS---------------
param_grid_diff_tasks = {
    "50":[
        {"init_lr": 0.1, "batch_size": 64, "init_epoch": 200, "is_task0":True, "use_past_model": False, "save_model": True},
    ],
    "40":[
        {"init_lr": 0.1, "batch_size": 64, "init_epoch": 200, "is_task0":True, "use_past_model": False, "save_model": True},
    ]
}
# ------------------------ LOGGING CONFIGURATION ------------------------
logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(filename="logs/GRAIL/" + "tune_CIFAR100.log"),
                logging.StreamHandler(sys.stdout),
            ],
)


def run_experiment(params, gpu_id, init_cls, seed, process_id):
    # load config
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    # update config
    config_updated = copy.deepcopy(config)
    config_updated.update(params)

    config_updated['resume'] = False
    config_updated["device"] = [str(gpu_id)]
    config_updated["seed"] = [seed]
    config_updated["process_id"] = process_id

    config_updated["init_cls"] = init_cls
    
    config_updated["note"] = Description + str(params)
    config_updated["version"] = VERSION
    cmd = ["python", "main_tune.py"]
    env = os.environ.copy()
    logging.info(f"Running experiment on seed {seed}, init_cls {init_cls}, on GPU {gpu_id}, params:{params}")
    env["CONFIG_JSON"] = json.dumps(config_updated)
    subprocess.run(cmd, env=env)


with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_PROCESSES) as executor:
    futures = []
    idx = 0
    task_count = 0  # Task counter
    process_ids = list(range(MAX_CONCURRENT_PROCESSES))

    for init_cls in INCREMENTS:
        for _, params in enumerate(param_grid_diff_tasks["{}".format(init_cls)]):
            for seed in SEEDS:
                gpu_id = GPUS[idx % len(GPUS)]
                idx += 1
                process_id = process_ids[task_count % MAX_CONCURRENT_PROCESSES]
                task_count += 1
                futures.append(executor.submit(run_experiment, params, gpu_id, init_cls, seed, process_id))
                time.sleep(3)
    # wait for all futures to complete
    for future in futures:
        future.result()
