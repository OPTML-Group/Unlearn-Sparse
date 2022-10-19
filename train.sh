seed=1
arch=resnet18
rewind_epoch=8
data=cifar10

CUDA_VISIBLE_DEVICES=1 python -u main_imp.py \
	--data ./data \
	--dataset "$data"_no_val \
	--arch $arch \
	--seed $seed \
	--prune_type rewind_lt \
	--rewind_epoch $rewind_epoch \
	--save_dir "imp_"$arch"_"$data"_seed"$seed"" \
	--rate 0.2 \
	--pruning_times 2 \
	--num_workers 8 \
	--batch_size 512