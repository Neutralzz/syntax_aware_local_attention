#!/bin/bash
export OUTPUT_DIR=/data/dobby_ceph_ir/neutrali/venus_outputs/synbert_task-fce_bs-32_lr-3e-5_epoch-10_seed-17
export PYTORCH_PRETRAINED_BERT_CACHE=/data/dobby_ceph_ir/neutrali/pretrained_weights
pip install --user --editable .
python src/run.py \
          --task cola \
          --model_type sent-cls \
          --model_name_or_path bert-base-uncased \
          --data_dir /data/dobby_ceph_ir/neutrali/workspace/VenusData/SynBert \
          --output_dir $OUTPUT_DIR \
          --train_file cola.train.tsv.pkl \
          --dev_file cola.dev.tsv.pkl \
          --do_train --do_eval --do_lower_case \
          --learning_rate 3e-5 \
          --num_train_epochs 10 \
          --order_metric mcc \
          --metric_reverse  \
          --remove_unused_ckpts \
          --per_gpu_train_batch_size 32 \
          --eval_all_checkpoints \
          --overwrite_output_dir \
          --save_steps 100 \
          --seed 17

