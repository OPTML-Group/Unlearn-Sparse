# Reproduction Matrix

Use this reference after the skill triggers and the user asks to map paper results to commands.

## Paper-to-Repo Map

Paper source: <https://arxiv.org/pdf/2304.04934>.

The paper evaluates "prune first, then unlearn" across class-wise forgetting, random data forgetting, sparsity-aware unlearning, Trojan/backdoor cleanse, and transfer learning. The current repo exposes these entrypoints:

- `python main.py train`: baseline training plus iterative pruning profiles `imp`, `ls`, `sam`, `vit`, `synflow`.
- `python main.py unlearn`: retrain/approximate unlearning and automatic accuracy/MIA evaluation.
- `python main.py backdoor`: backdoor poisoning and cleanse flow.

The paper metrics map to repo outputs as follows:

- UA: accuracy on the forget loader after unlearning.
- RA: accuracy on the retain loader after unlearning.
- TA: accuracy on test/val loader after unlearning.
- MIA-Efficacy: `SVC_MIA_forget_efficacy`.
- MIA-Privacy: `SVC_MIA_training_privacy`.
- RTE: wall-clock timing from logs; the repo does not aggregate it automatically.

## Main Table 3

Purpose: compare dense vs 95%-sparse models for approximate MU under class-wise and random data forgetting.

Default setup:

- Dataset/model: CIFAR-10, ResNet-18.
- Batch size: 256.
- Pruning: OMP-style one-shot 95% sparsity via `train --profile imp --rate 0.95 --pruning_times 1 --prune_type rewind_lt --rewind_epoch 8`.
- Dense mask: `0model_SA_best.pth.tar`.
- Sparse mask: `1model_SA_best.pth.tar` when `--pruning_times 1`.
- Trials: paper reports mean/std over 10 independent trials; use different `--seed` and `--train_seed`.

Forgetting scenarios:

- Class-wise CIFAR-10: `--class_to_replace 0 --num_indexes_to_replace 4500`.
- Random 10% CIFAR-10: `--class_to_replace -1 --num_indexes_to_replace 4500`.

Unlearning methods exposed by repo:

- Retrain: `--unlearn retrain --unlearn_epochs 160 --unlearn_lr 0.1`.
- FT: `--unlearn FT --unlearn_epochs 10 --unlearn_lr 0.01`.
- GA: `--unlearn GA --unlearn_epochs 5 --unlearn_lr 0.0001`.
- FF: `--unlearn fisher_new --alpha 0.2`, then tune `alpha` if reproducing exact table numbers.
- IU: `--unlearn wfisher --alpha 0.2`.

## Sparsity-Aware MU

Purpose: reproduce the paper's sparsity-aware unlearning comparisons.

Repo command:

- `--unlearn FT_prune --alpha 0.2 --unlearn_lr 0.01 --unlearn_epochs 10`.

Scheduling detail:

- The code computes a decaying L1 coefficient inside `src/optim/unlearn/FT.py` / `FT_prune`-related paths based on `alpha`, `unlearn_epochs`, and `no_l1_epochs`.
- To compare constant vs decreasing schedules, inspect the current `FT_prune` implementation before generating commands; the paper studies scheduler variants, but the repo may expose them through a limited parameter surface.

## Appendix Dataset/Architecture Sweeps

Use the same train -> checkpoint -> unlearn pattern.

Dataset/model setups described by the paper:

- CIFAR-10: ResNet-18 and VGG-16, batch size 256.
- CIFAR-100: ResNet-18, batch size 256.
- SVHN: ResNet-18, batch size 256.
- ImageNet: ResNet-18, batch size 1024.

Training details from the paper:

- CIFAR-10/CIFAR-100: 182 training epochs, rewind epoch 8, momentum 0.9, weight decay `5e-4`, LR decay at 50% and 75% epochs.
- SVHN: 160 training epochs, rewind epoch 8, momentum 0.9, weight decay `5e-4`.
- ImageNet: 90 training epochs, rewind epoch 5, momentum 0.875, weight decay `3.05e-5`, warmup 8.
- FT: 10 epochs.
- GA: 5 epochs.
- CIFAR-10 class-wise reference LR: FT `0.01`, GA `0.0001`.
- FF alpha requires search over very small values in the original paper; the repo examples often use `0.2`, so note deviations.

Current repo names:

- Dataset names: `cifar10`, `cifar100`, `svhn`, `imagenet`, `TinyImagenet`.
- Model names: `resnet18`, `resnet50`, `resnet20s`, `resnet44s`, `resnet56s`, `vgg16_bn`, `vgg16_bn_lth`, optional `swin_t`.

## Backdoor / Trojan Cleanse

Purpose: reproduce the paper's backdoor cleanse application.

Repo command:

- `python main.py backdoor --dataset cifar10 --arch resnet18 --unlearn FT --num_indexes_to_replace 4500 --class_to_replace 0 --trigger_size 4 --rate <sparsity> --mask <mask-path>`.

Sparsity sweep:

- Use rates `0`, `0.75`, `0.90`, `0.95`, `0.99`.
- Compare `test_acc`, `attack_acc`, `test_acc_unlearn`, and `attack_acc_unlearn` from the saved evaluation checkpoint.

## ImageNet Unlearning

Prerequisites:

- HuggingFace access for ImageNet-1k.
- Label tensors at `--train_y_file` and `--val_y_file`; the helper in `src/dataio/imagenet.py` can generate them when run as a script.
- Large GPU budget. The paper reports ImageNet RTE in hours.

Command shape:

```bash
python -u main.py unlearn --dataset imagenet --imagenet_arch --arch resnet18 \
  --save_dir ./runs/imagenet_resnet18 \
  --mask ./runs/imagenet_resnet18/0model_SA_best.pth.tar \
  --class_to_replace 0 --unlearn FT --unlearn_epochs 5 --unlearn_lr 0.001 \
  --batch_size 1024 --input_size 224 --train_y_file ./labels/train_ys.pth \
  --val_y_file ./labels/val_ys.pth
```

## Transfer Learning Result

The paper evaluates removing ImageNet classes and downstream linear probing on SUN397 and OxfordPets, with FFCV used for acceleration. The current repo does not provide a complete transfer-learning CLI for this table. To reproduce it, state the missing pieces:

- An ImageNet source-model unlearning stage.
- Downstream dataset loaders for SUN397 and OxfordPets.
- A linear-probing script that freezes the feature extractor and trains only the classification head.
- Runtime logging consistent with the paper's comparison.

Do not claim this result is fully reproducible from `main.py` alone.
