#!/bin/bash
export INPUT_DIR=/path_to_load/
export OUTPUT_DIR=/path_to_store/

pip install --user --editable .
python src/run.py \
          --task cola \
          --model_type sent-cls \
          --model_name_or_path bert-base-uncased \
          --data_dir   $INPUT_DIR  \
          --output_dir $OUTPUT_DIR \
          --train_file cola.train.tsv.d3.pkl \
          --dev_file cola.dev.tsv.d3.pkl \
          --do_train --do_eval --do_lower_case \
          --learning_rate 3e-5 \
          --num_train_epochs 2 \
          --order_metric mcc \
          --metric_reverse  \
          --remove_unused_ckpts \
          --per_gpu_train_batch_size 16 \
          --eval_all_checkpoints \
          --overwrite_output_dir \
          --save_steps 100 \
          --seed 17

