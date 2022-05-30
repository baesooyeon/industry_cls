#! /bin/bash

# MODEL='kogpt3'
# LOSS='FCE'
# EPOCH=50
# BATCH=256
# OPTIMIZER='AdamW'
# BETA1=0.5
# UPSAMPLE='shuffle'
# MINIMUM=50
# N_LAYERS=1
# BN='' #  --batchnorm layernorm과 batchnorm은 둘 중 하나만 이용하세요.
# LN='' # ' --layernorm'
# NUM_TEST=50000
# LR=5e-5
# DR_RATE=0.2
# DEVICE='cuda:0'
# SCHEDULER='cosine_with_restarts'
# PATIENCE=3

# eval 'python train2.py --root=/home/jupyter/data_prep/"1. 실습용자료_final.txt" --model=${MODEL} --loss=${LOSS} --epoch=${EPOCH} --batch-size=${BATCH} --optimizer=${OPTIMIZER} --beta1=${BETA1} --upsample=${UPSAMPLE} --minimum=${MINIMUM} --n-layers=${N_LAYERS} --layernorm --num-test=${NUM_TEST} --learning-rate=${LR} --dr-rate=${DR_RATE}  --device=${DEVICE} --lr-scheduler=${SCHEDULER} --patience=${PATIENCE}'


# MODEL='kobart'
# LOSS='FCE'
# EPOCH=50
# BATCH=256
# OPTIMIZER='AdamW'
# BETA1=0.5
# UPSAMPLE='shuffle'
# MINIMUM=50
# N_LAYERS=1
# BN='' #  --batchnorm layernorm과 batchnorm은 둘 중 하나만 이용하세요.
# LN='' # ' --layernorm'
# NUM_TEST=50000
# LR=5e-5
# DR_RATE=0.3
# DEVICE='cuda:0'
# SCHEDULER='cosine_with_restarts'
# PATIENCE=3
# DECAY=0.01

# eval 'python train2.py --root=/home/jupyter/data_prep/"1. 실습용자료_final.txt" --model=${MODEL} --loss=${LOSS} --epoch=${EPOCH} --batch-size=${BATCH} --optimizer=${OPTIMIZER} --beta1=${BETA1} --upsample=${UPSAMPLE} --minimum=${MINIMUM} --n-layers=${N_LAYERS} --layernorm --num-test=${NUM_TEST} --learning-rate=${LR} --dr-rate=${DR_RATE}  --device=${DEVICE} --lr-scheduler=${SCHEDULER} --patience=${PATIENCE} --weight-decay=${DECAY}'

# MODEL='asbart'
# LOSS='FCE'
# EPOCH=50
# OPTIMIZER='AdamW'
# BATCH=128
# BETA1=0.5
# UPSAMPLE='shuffle'
# MINIMUM=50
# N_LAYERS=1
# BN='' #  --batchnorm layernorm과 batchnorm은 둘 중 하나만 이용하세요.
# LN='' # ' --layernorm'
# NUM_TEST=50000
# LR=5e-5
# DECAY=0.01
# DR_RATE=0.5
# PATIENCE=3
# DEVICE='cuda:0'
# SCHEDULER='cosine_with_restarts'


# eval 'python train2.py --root=/home/jupyter/data_prep/"1. 실습용자료_final.txt" --model=${MODEL} --num-test=${NUM_TEST} -lr=${LR} --n-layers=${N_LAYERS} --dr-rate=${DR_RATE} --upsample=${UPSAMPLE} --minimum=${MINIMUM} --lr-scheduler=${SCHEDULER} --batch-size=${BATCH} --loss=${LOSS} --epoch=${EPOCH} --optimizer=${OPTIMIZER} --beta1=${BETA1} --patience=${PATIENCE} --device=${DEVICE}${BN}${LN} --weight-decay=${DECAY}'




# MODEL='kogpt3'
# LOSS='CE'
# EPOCH=15
# BATCH=256
# OPTIMIZER='AdamW'
# BETA1=0.5
# UPSAMPLE='shuffle'
# MINIMUM=200
# N_LAYERS=1
# BN='' #  --batchnorm layernorm과 batchnorm은 둘 중 하나만 이용하세요.
# LN='' # ' --layernorm'
# NUM_TEST=50000
# LR=5e-5
# DR_RATE=0.5
# DEVICE='cuda:0'
# SCHEDULER='cosine_with_restarts'
# PATIENCE=2
# DECAY=0.01

# eval 'python train2.py --root=/home/jupyter/data_prep/"1. 실습용자료_final.txt" --model=${MODEL} --loss=${LOSS} --epoch=${EPOCH} --batch-size=${BATCH} --optimizer=${OPTIMIZER} --beta1=${BETA1} --upsample=${UPSAMPLE} --minimum=${MINIMUM} --n-layers=${N_LAYERS} --layernorm --num-test=${NUM_TEST} --learning-rate=${LR} --dr-rate=${DR_RATE}  --device=${DEVICE} --lr-scheduler=${SCHEDULER} --patience=${PATIENCE} --weight-decay=${DECAY}'


# MODEL='kobart'
# LOSS='CE'
# EPOCH=50
# BATCH=256
# OPTIMIZER='AdamW'
# BETA1=0.5
# UPSAMPLE='shuffle'
# MINIMUM=200
# N_LAYERS=1
# BN='' #  --batchnorm layernorm과 batchnorm은 둘 중 하나만 이용하세요.
# LN='' # ' --layernorm'
# NUM_TEST=50000
# LR=5e-5
# DR_RATE=0.5
# DEVICE='cuda:0'
# SCHEDULER='cosine_with_restarts'
# PATIENCE=2
# DECAY=0.01

# eval 'python train2.py --root=/home/jupyter/data_prep/"1. 실습용자료_final.txt" --model=${MODEL} --loss=${LOSS} --epoch=${EPOCH} --batch-size=${BATCH} --optimizer=${OPTIMIZER} --beta1=${BETA1} --upsample=${UPSAMPLE} --minimum=${MINIMUM} --n-layers=${N_LAYERS} --layernorm --num-test=${NUM_TEST} --learning-rate=${LR} --dr-rate=${DR_RATE}  --device=${DEVICE} --lr-scheduler=${SCHEDULER} --patience=${PATIENCE} --weight-decay=${DECAY}'


# MODEL='kobart'
# LOSS='CE'
# EPOCH=50
# BATCH=256
# OPTIMIZER='AdamW'
# BETA1=0.5
# UPSAMPLE='shuffle'
# MINIMUM=200
# N_LAYERS=1
# BN='' #  --batchnorm layernorm과 batchnorm은 둘 중 하나만 이용하세요.
# LN='' # ' --layernorm'
# NUM_TEST=50000
# LR=5e-5
# DR_RATE=0.5
# DEVICE='cuda:0'
# SCHEDULER='cosine_with_restarts'
# PATIENCE=2
# DECAY=0.1



################## 추가학습 ############################
# MODEL='kobart'
# LOSS='CE'
# EPOCH=50
# BATCH=256
# OPTIMIZER='AdamW'
# BETA1=0.5
# UPSAMPLE='shuffle'
# MINIMUM=200
# N_LAYERS=1
# BN='' #  --batchnorm layernorm과 batchnorm은 둘 중 하나만 이용하세요.
# LN='' # ' --layernorm'
# NUM_TEST=50000
# LR=5e-5
# DR_RATE=0.5
# DEVICE='cuda:0'
# SCHEDULER='constant_with_warmup'
# PATIENCE=2
# DECAY=0.1

# eval 'python train2.py --root=/home/jupyter/data_prep/"1. 실습용자료_final.txt" --model=${MODEL} --loss=${LOSS} --epoch=${EPOCH} --batch-size=${BATCH} --optimizer=${OPTIMIZER} --beta1=${BETA1} --upsample=${UPSAMPLE} --minimum=${MINIMUM} --n-layers=${N_LAYERS} --layernorm --num-test=${NUM_TEST} --learning-rate=${LR} --dr-rate=${DR_RATE}  --device=${DEVICE} --lr-scheduler=${SCHEDULER} --patience=${PATIENCE} --weight-decay=${DECAY}'


MODEL='mlbert'
LOSS='CE'
EPOCH=15
BATCH=256
OPTIMIZER='AdamW'
BETA1=0.1
UPSAMPLE='shuffle'
MINIMUM=200
N_LAYERS=1
BN='' #  --batchnorm layernorm과 batchnorm은 둘 중 하나만 이용하세요.
LN='' # ' --layernorm'
NUM_TEST=50000
LR=5e-5
DECAY=0.1
DR_RATE=0.5
PATIENCE=2
DEVICE='cuda'
SCHEDULER='constant_with_warmup'

eval 'python train2.py --root=/home/jupyter/data_prep/"1. 실습용자료_final.txt" --model=${MODEL} --num-test=${NUM_TEST} -lr=${LR} --n-layers=${N_LAYERS} --dr-rate=${DR_RATE} --upsample=${UPSAMPLE} --minimum=${MINIMUM} --lr-scheduler=${SCHEDULER} --batch-size=${BATCH} --loss=${LOSS} --epoch=${EPOCH} --optimizer=${OPTIMIZER} --beta1=${BETA1} --patience=${PATIENCE} --device=${DEVICE}${BN}${LN} --weight-decay=${DECAY}'



MODEL='funnel'
LOSS='CE'
EPOCH=15
BATCH=256
OPTIMIZER='AdamW'
BETA1=0.1
UPSAMPLE='shuffle'
MINIMUM=200
N_LAYERS=1
BN='' #  --batchnorm layernorm과 batchnorm은 둘 중 하나만 이용하세요.
LN='' # ' --layernorm'
NUM_TEST=50000
LR=5e-5
DECAY=0.01
DR_RATE=0.5
PATIENCE=2
DEVICE='cuda'
SCHEDULER='constant_with_warmup'

eval 'python train2.py --root=/home/jupyter/data_prep/"1. 실습용자료_final.txt" --model=${MODEL} --num-test=${NUM_TEST} -lr=${LR} --n-layers=${N_LAYERS} --dr-rate=${DR_RATE} --upsample=${UPSAMPLE} --minimum=${MINIMUM} --lr-scheduler=${SCHEDULER} --batch-size=${BATCH} --loss=${LOSS} --epoch=${EPOCH} --optimizer=${OPTIMIZER} --beta1=${BETA1} --patience=${PATIENCE} --device=${DEVICE}${BN}${LN} --weight-decay=${DECAY}'












