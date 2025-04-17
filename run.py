import re
import os
import json
import subprocess 

from path import Slurm
from config import Configs

subprocess.run(["chmod", "777", "*.sh"], capture_output=True, text=True)  

slurm_id_dict = {}

for fold in Configs.n_splits:
    result = subprocess.run(["sbatch", "--gpus=1", 
                            f"--output={fold}.out", 
                             "-p", "gpu", 
                             "MGD4MD.sh", str(fold)], capture_output=True, text=True)
    print(result.stdout, f"fold: {fold}")
    ids = list(map(int, re.findall(r"\d+", result.stdout)))
    assert len(ids) == 1, f"len(ids) != 1, ids: {ids}"
    slurm_id_dict[fold] = ids[0]
    
with open(Slurm.slurm_id_path, "w", encoding="utf-8") as f:
    json.dump(slurm_id_dict, f, indent=4)