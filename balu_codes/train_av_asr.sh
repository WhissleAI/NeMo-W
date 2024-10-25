#!/bin/bash
#SBATCH -N 1
#SBATCH -J "nemo_train"
#SBATCH -p gpu
#SBATCH -c 46
#SBATCH -G 4
#SBATCH --mem-per-cpu=3G
#SBATCH -o "/home/bld56/gsoc/nemo/NeMo-opensource/balu_codes/nemo_train/%j.log"
#SBATCH -w "gput065"
#SBATCH --time="2-00:00:00"
#SBATCH --mail-type=ALL
#SBATCH --mail-user="lakshmipathi.balaji@research.iiit.ac.in"

# bash /home/bld56/gsoc/general/set_up_node.sh
# export PATH="/home/bld56/.miniconda3/bin:$PATH"
# export PATH="$HOME/tools:$PATH"


cd /home/bld56/gsoc/nemo/NeMo-opensource/balu_codes

source activate /home/bld56/.miniconda3/envs/nemo
# bash /home/bld56/gsoc/general/set_up_node.sh

snrs=(0.909091 0.849020 0.759747 0.640065 0.500000 0.359935 0.240253 0.150980 0.090909)

CUDA_VISIBLE_DEVICES=0 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.909091 --resume_pretrained True &

sleep 100

CUDA_VISIBLE_DEVICES=1 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.849020 --resume_pretrained True &

sleep 100

CUDA_VISIBLE_DEVICES=2 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.759747 --resume_pretrained True &

sleep 100

CUDA_VISIBLE_DEVICES=3 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.640065 --resume_pretrained True &

wait

CUDA_VISIBLE_DEVICES=0 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.500000 --resume_pretrained True &

sleep 100

CUDA_VISIBLE_DEVICES=1 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.359935 --resume_pretrained True &

sleep 100

CUDA_VISIBLE_DEVICES=2 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.240253 --resume_pretrained True &

sleep 100

CUDA_VISIBLE_DEVICES=3 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.150980 --resume_pretrained True &

wait

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 5 --snr 0.090909 --resume_pretrained True &

# wait

# CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bld56/.miniconda3/envs/nemo/bin/python train_av_asr.py --config 1 --snr "rand" --resume_pretrained True &
