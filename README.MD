# Unlearn vs Sparse

## Pruning

### OMP

```python -u main_imp.py --data ./data --dataset $data --arch $arch --seed ${seed} --prune_type rewind_lt --rewind_epoch ${rewind_epoch} --save_dir ${save_dir} --rate ${rate} --pruning_times 2 --num_workers 8```

### IMP

```python -u main_imp.py --data ./data --dataset $data --arch $arch --seed ${seed} --prune_type rewind_lt --rewind_epoch ${rewind_epoch} --save_dir ${save_dir} --rate 0.2 --pruning_times ${pruning_times} --num_workers 8```

### SynFlow

```python -u main_synflow.py --data ./data --dataset cifar10 --seed ${seed} --prune_type rewind_lt --rewind_epoch ${rewind_epoch} --save_dir ${save_dir} --rate ${rate} --pruning_times 1 --num_workers 8```

## Unlearning

### Retrain

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn retrain --num_indexes_to_replace ${num} --seed ${seed} --unlearn_epochs ${epoch} --unlearn_lr 0.1```

### FT

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn FT --num_indexes_to_replace ${num} --seed ${seed} --unlearn_lr 0.01 --unlearn_epochs 10```

### GA

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn GA --num_indexes_to_replace ${num} --seed ${seed} --unlearn_lr 0.0001 --unlearn_epochs 5```

### FF

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn fisher_new --num_indexes_to_replace ${num} --seed ${seed}--alpha ${alpha}```

### IU

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn wfisher --num_indexes_to_replace ${num} --seed ${seed}--alpha ${alpha} --batch_size 64```

### l1-sparse

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn FT_prune --num_indexes_to_replace ${num} --seed ${seed}--alpha ${alpha} --unlearn_lr 0.01 --unlearn_epochs 10```

### AO-sparse

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn FT_prune_bi --num_indexes_to_replace ${num} --seed ${seed} --unlearn_lr 0.01 --unlearn_epochs 10```

## Backdoor

```python```