#!/bin/bash
for file in slurm-*.out; do  
    if [[ -f "$file" ]]; then  
        id=$(echo "$file" | sed -E 's/slurm-([0-9]+)\.out/\1/')  
        echo "Cancelling job with ID: $id"  
        scancel "$id"  
    fi  
done

log_patterns=("state_*.log" "slurm-*.out")  
for pattern in "${log_patterns[@]}"; do  
  files=$(find . -type f -name "${pattern}")  
  for file in $files; do  
    if [ -f "${file}" ]; then  
      rm -f "${file}"  
    fi  
  done  
done

sleep 3

tmp_file=$(mktemp)  
find ./ -type d -name '__pycache__' > "${tmp_file}"  
while IFS= read -r dir  
do  
  rm -rf "${dir}"  
done < "${tmp_file}"  
rm -f "${tmp_file}"  

parajobs