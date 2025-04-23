#!/bin/bash

# Check if the training was successful by checking if the model file exists
if [ -f "./models/ppo_cartpole_final.zip" ]; then
    echo -e "\nTraining completed successfully. Starting validation..."
    python3 ./ppo_cartpole.py --mode validate
else
    echo -e "\nTraining failed or model file not found. Skipping validation."
fi