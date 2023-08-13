python run.py --output_dir=./saved_models/devign_e100 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/train.jsonl --eval_data_file=../dataset/valid.jsonl --test_data_file=../dataset/test.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 --model=devign --log_neptune 2>&1 | tee $logp/training_log.txt