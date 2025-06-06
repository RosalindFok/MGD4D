import re
import json
import argparse
import subprocess 

from config import Configs
from path import Slurm, Paths

subprocess.run(["chmod", "777", "*.sh"], capture_output=True, text=True)  

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str) # T/t - train-valid-test; X/x - XAI
args = parser.parse_args()
task = args.task.lower()

slurm_id_dict = {}

if task == "t":
    for file_path in Paths.Run_Files.embedding_dir_path.iterdir():
        file_path.unlink()
    for fold in Configs.n_splits:
        result = subprocess.run(["sbatch", "--gpus=1", 
                                f"--output={fold}.out", 
                                 "-p", "gpu", 
                                 "MGD4D.sh", str(fold)], capture_output=True, text=True)
        print(result.stdout, f"fold: {fold}")
        ids = list(map(int, re.findall(r"\d+", result.stdout)))
        assert len(ids) == 1, f"len(ids) != 1, ids: {ids}"
        slurm_id_dict[fold] = ids[0]
        with open(Slurm.slurm_id_path, "w", encoding="utf-8") as f:
            json.dump(slurm_id_dict, f, indent=4)
elif task == "x":
    result = subprocess.run(["sbatch", "--gpus=1", f"--output=XAI.out", "-p", "gpu", "XAI.sh"], capture_output=True, text=True)
    print(result.stdout)
else:
    raise ValueError(f"task: {task} is not supported, please use T/t or X/x")