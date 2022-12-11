#! /bin/bash

source activate ucfadar-relnet
# tensorboard --logdir=/experiment_data/il_experiment/models/summaries &
jupyter notebook --no-browser --notebook-dir=/relnet --ip=0.0.0.0 &

tail -f /dev/null