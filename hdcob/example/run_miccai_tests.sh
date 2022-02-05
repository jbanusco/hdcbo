#!/bin/sh

#source activate py37
source activate /home/jbanusco/py_envs/py36

SAVE_FOLDER="results"
S_NAME="S2"

EPOCHS=500
PRINT_EPOCHS=505
EPOCHS_WARM_UP_KL=10
LR=(0.005)
RESUME=True
BATCH_SIZE=(0)
LATENT_DIM=(1)
HIDDEN_DIM=0
USE_PARAM=(True)

SNR_MISS=(10 5 3 1.5)
SNR_TARGET=(10 5 3 1.5)

MISSING_DIM=4
COND_DIM=5
INPUT_DIM=1
TARGET_DIM=5

# Generate array job
for lr in ${LR[@]}
do
for batch_size in ${BATCH_SIZE[@]}
do
for latent in ${LATENT_DIM[@]}
do
for param in ${USE_PARAM[@]}
do
for kernel in ${KERNEL[@]}
do
    string_input="--save_folder=${SAVE_FOLDER} --epochs=${EPOCHS} --lr=${lr} --kernel=${kernel} --epochs_kl=${EPOCHS_WARM_UP_KL} --batch_size=${batch_size} --latent=${latent} --hidden=${HIDDEN_DIM} --use_param=${param} --decoder_type=${DECODER_TYPE} --resume=${RESUME} --print_epochs=${PRINT_EPOCHS}"
    echo ${string_input} >> ${array_param_filename}

done
done
done
done
done


