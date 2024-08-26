#!/bin/bash
#SBATCH -N 1
#SBATCH -J "nemo_train"
#SBATCH -p gpu
#SBATCH -c 46
#SBATCH -G 2
#SBATCH --mem-per-cpu=3G
#SBATCH -o "/home/bld56/gsoc/nemo/NeMo-opensource/balu_codes/nemo_train/%j.log"
#SBATCH -w "gput067"
#SBATCH --time="2-00:00:00"
#SBATCH --mail-type=ALL
#SBATCH --mail-user="lakshmipathi.balaji@research.iiit.ac.in"

# bash /home/bld56/gsoc/general/set_up_node.sh
# export PATH="/home/bld56/.miniconda3/bin:$PATH"
# export PATH="$HOME/tools:$PATH"


cd /home/bld56/gsoc/nemo/NeMo-opensource/balu_codes
# gput064
# CUDA_VISIBLE_DEVICES=0 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 1 --snr 0.6 &
# sleep 10
# CUDA_VISIBLE_DEVICES=1 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 2 --snr 0.6 &
# sleep 10
# CUDA_VISIBLE_DEVICES=2 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.6 &

# gput067
# sleep 10
# CUDA_VISIBLE_DEVICES=0 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 6 --snr 0.7 &
# sleep 10
# CUDA_VISIBLE_DEVICES=1 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 1 --snr 0.6 &


# gput068
# sleep 10
# CUDA_VISIBLE_DEVICES=0 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 2 --snr 0.6 &
# sleep 100
# CUDA_VISIBLE_DEVICES=1 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.6 &

# gput066
# sleep 100
# CUDA_VISIBLE_DEVICES=0 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 6 --snr 0.6 &

# gput065
# CUDA_VISIBLE_DEVICES=0 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 1 --snr 0.5 &
# sleep 100
# CUDA_VISIBLE_DEVICES=1 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.5 &

# gput063
# CUDA_VISIBLE_DEVICES=0 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 6 --snr 0.5 &

# PRETRAINING, gput063
# bash /home/bld56/gsoc/general/set_up_node.sh

# PRETRAINED USING
source activate /home/bld56/.miniconda3/envs/nemo
# bash /home/bld56/gsoc/general/set_up_node.sh

# av_ndec_lman_ntok
# CUDA_VISIBLE_DEVICES=0 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.5 --resume_pretrained True &

sleep 20
# av_ndec_uman_ntok
# CUDA_VISIBLE_DEVICES=1 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 9 --snr 0.5 --resume_pretrained True &

# au_ndec_lman_ntok
CUDA_VISIBLE_DEVICES=0 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 10 --snr 0.5 --resume_pretrained True &

sleep 20
# au_ndec_uman_ntok
CUDA_VISIBLE_DEVICES=1 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 11 --snr 0.5 --resume_pretrained True &


wait