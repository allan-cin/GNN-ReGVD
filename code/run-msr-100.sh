python run.py --output_dir=./saved_models/regcn_e100_dMSR \
  --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
  --do_eval --do_test --do_train \
  --train_data_file=../dataset/msr/train_msr.jsonl --eval_data_file=../dataset/msr/valid_msr.jsonl --test_data_file=../dataset/msr/test_msr.jsonl \
  --block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
  --gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
  --seed 123456 --log_neptune 2>&1 | tee -i ./saved_models/regcn_e100_dMSR/training_log.txt
