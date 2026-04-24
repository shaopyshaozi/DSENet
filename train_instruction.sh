#!/bin/bash -l
set -e

#$ -N dsenet_train
#$ -cwd
#$ -l gpu=1
#$ -l mem=32G
#$ -l tmpfs=32G
#$ -l h_rt=48:00:00
#$ -o /home/ucabps5/code/DSENet/logs/train.log
#$ -e /home/ucabps5/code/DSENet/logs/train.err

# connect to interactive server
qrsh -l gpu=1 -l mem=32G,h_rt=48:00:00 -now no

tmux new -s train
tmux attach -t train

# activate env
module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0

source /home/ucabps5/code/DSENet/venv/bin/activate

# For evaluation
# tensorboard --logdir /home/ucabps5/code/DSENet/logs --port 6006

# Go to your project folder
cd /home/ucabps5/code/DSENet
mkdir -p /home/ucabps5/code/DSENet/logs

# Do some random check
echo "Running on $(hostname)"
which python
python --version
nvidia-smi

python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
EOF

# run training
python DOATrainer.py fit \
  --config configs/DOASE.yaml \
  --data.train_dir "/home/ucabps5/code/DSENet/data/dataset_3mic_6spk/train" \
  --data.test_dir "/home/ucabps5/code/DSENet/data/dataset_3mic_6spk/val" \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision 16-mixed \
  --data.batch_size "[2,2]" \
  --trainer.accumulate_grad_batches 2 \
  --data.num_workers 4 \
  --trainer.max_epochs 100 \
  --trainer.num_sanity_val_steps 0 \
  --ckpt_path "/home/ucabps5/code/DSENet/logs/DSENet/version_9/checkpoints/last.ckpt" \
  > logs/train.log 2>&1
