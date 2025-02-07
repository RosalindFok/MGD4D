#!/bin/bash
export PYTHONUNBUFFERED=1

# Create a state file to control the collection process
STATE_FILE="state_${BATCH_JOB_ID}.log"
/usr/bin/touch ${STATE_FILE}

# Execute the example script
python main.py

# Stop the GPU collection process
echo "over" >> "${STATE_FILE}"