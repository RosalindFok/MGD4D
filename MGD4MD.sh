#!/bin/bash
export PYTHONUNBUFFERED=1

fold=$1

# Execute the example script
python main.py --fold $fold
# python train_Encoder.py