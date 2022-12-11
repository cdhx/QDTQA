#!/bin/bash
# generate sparql from question along with qdt

export DATA_DIR=data
ACTION=${1:-none}
dataset="CWQ"

if [ "$ACTION" = "train" ]; then
    exp_id=$2

    exp_prefix="exps/gen_${dataset}_${exp_id}/"

    if [ -d ${exp_prefix} ]; then
    echo "${exp_prefix} already exists"
    else
    mkdir -p ${exp_prefix}
    fi

    cp scripts/run_gen_nlq_newqdt_candEnt.sh "${exp_prefix}run_gen_nlq_newqdt_candEnt.sh"
    git rev-parse HEAD > "${exp_prefix}commitid.log"

    # --model_name_or_path t5-base \
    # --cache_dir "hfcache/t5-base"
    # --overwrite_cache
    # --add_relation \
    python -u run_generator.py \
        --data_dir ${DATA_DIR} \
        --dataset ${dataset} \
        --model_type t5 \
        --model_name_or_path "hfcache/t5-base" \
        --do_lower_case \
        --do_train \
        --overwrite_output_dir \
        --use_qdt \
        --new_qdt \
        --add_entity \
        --overwrite_cache \
        --max_source_length 384 \
        --max_target_length 196 \
        --train_file ${DATA_DIR}/${dataset}_train_expr.json \
        --predict_file ${DATA_DIR}/${dataset}_dev_expr.json \
        --learning_rate 3e-5 \
        --num_train_epochs 25 \
        --logging_steps 1000 \
        --eval_steps 10000 \
        --save_steps 10000 \
        --warmup_ratio 0.1 \
        --output_dir "${exp_prefix}output" \
        --eval_beams 10 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 | tee "${exp_prefix}log.txt"
        

elif [ "$ACTION" = "eval" -o "$ACTION" = "predict" ]; then
    exp_id=$2
    # echo $exp_id
    model="exps/gen_${dataset}_${exp_id}/output"
    # echo $model
    #model=$2
    split=${3:-dev}
    
    
    # -u : force the binary I/O layers of stdout and stderr to be unbuffered;
    # --add_relation
    # --overwrite_cache \ 
    python -u run_generator.py \
        --data_dir ${DATA_DIR} \
        --dataset ${dataset} \
        --model_type t5 \
        --model_name_or_path ${model} \
        --eval_beams 50 \
        --do_lower_case \
        --do_eval \
        --overwrite_cache \
        --use_qdt \
        --new_qdt \
        --add_entity \
        --max_source_length 384 \
        --max_target_length 196 \
        --predict_file ${DATA_DIR}/${dataset}_${split}_expr.json \
        --overwrite_output_dir \
        --output_dir results/gen/${dataset}_${split}_${exp_id} \
        --per_device_eval_batch_size 4 
else
    echo "train or eval"
fi
