import json
from pathlib import Path
from collections import defaultdict 

from config import Train_Config

json_files = Path(".").glob('fold_*.json')
result = defaultdict(list)
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            result[key].append(value)

for key, value in result.items():
    assert len(value) == Train_Config.n_splits.stop-1, f"Number of values for {key} does not match n_splits"
    result[key] = {
        "list" : value, 
        "mean" : sum(value)/len(value)
    }

with open("average_result.json", "w") as f:
    json.dump(result, f, indent=4)