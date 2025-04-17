import json
import shutil
import subprocess 
from pathlib import Path

from path import Slurm

if Slurm.slurm_id_path.exists():
    with open(Slurm.slurm_id_path, "r", encoding="utf-8") as f:
        slurm_id = json.load(f)

    for id in slurm_id.values():
        subprocess.run(["scancel", str(id)], capture_output=True, text=True)

# delete *.out
for out_file in Path(".").rglob("*.out"):
    if out_file.is_file():
        out_file.unlink()

# delete fold_*
for fold_dir in Path(".").rglob("fold_*"):
    if fold_dir.is_file():
        fold_dir.unlink()
        
# delete __pycache__
for pycache_dir in Path(".").rglob("__pycache__"):
    if pycache_dir.is_dir():
        shutil.rmtree(pycache_dir)

# delete slurm_ids.json
Slurm.slurm_id_path.unlink(missing_ok=True)

result = subprocess.run(["parajobs"], capture_output=True, text=True)
print(result.stdout)