# Unlearn vs Sparse

## Pruning

### OMP

```python -u main_imp.py --data ./data --dataset $data --arch $arch --prune_type rewind_lt --rewind_epoch 8 --save_dir ${save_dir} --rate ${rate} --pruning_times 2 --num_workers 8```

### IMP

```python -u main_imp.py --data ./data --dataset $data --arch $arch --prune_type rewind_lt --rewind_epoch 8 --save_dir ${save_dir} --rate 0.2 --pruning_times ${pruning_times} --num_workers 8```

### SynFlow

```python -u main_synflow.py --data ./data --dataset cifar10 --prune_type rewind_lt --rewind_epoch 8 --save_dir ${save_dir} --rate ${rate} --pruning_times 1 --num_workers 8```

## Unlearning

### Retrain

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn retrain --num_indexes_to_replace 4500 --unlearn_epochs 160 --unlearn_lr 0.1```

### FT

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn FT --num_indexes_to_replace 4500 --unlearn_lr 0.01 --unlearn_epochs 10```

### GA

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn GA --num_indexes_to_replace 4500 --unlearn_lr 0.0001 --unlearn_epochs 5```

### FF

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn fisher_new --num_indexes_to_replace 4500 --alpha ${alpha}```

### IU

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn wfisher --num_indexes_to_replace 4500 --alpha ${alpha}```

### l1-sparse

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn FT_prune --num_indexes_to_replace 4500 --alpha ${alpha} --unlearn_lr 0.01 --unlearn_epochs 10```

### AO-sparse

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn FT_prune_bi --num_indexes_to_replace 4500 --unlearn_lr 0.01 --unlearn_epochs 10```

## Backdoor

```python -u main_backdoor.py --save_dir ${save_dir} --mask ${mask_path} --unlearn FT --num_indexes_to_replace 4500 --seed 2 --class_to_replace 0 --prune omp --rate 0 --trigger_size 4 --train_seed 6 --unlearn_epochs 10 --unlearn_lr 0.01```