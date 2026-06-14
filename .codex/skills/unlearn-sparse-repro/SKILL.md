---
name: unlearn-sparse-repro
description: Build and verify reproduction plans for the Unlearn-Sparse paper "Model Sparsity Can Simplify Machine Unlearning" using this repository. Use when Codex needs to reproduce paper experiments, generate train/unlearn/backdoor command batches, map paper tables/figures to repo CLI commands, validate expected checkpoints/evaluation outputs, or explain which paper results are and are not directly covered by the current code.
---

# Unlearn Sparse Repro

## Core Workflow

1. Inspect the current checkout before generating commands:
   - `python main.py --help`
   - `python main.py train --help`
   - `python main.py unlearn --help`
   - `python main.py backdoor --help`
   - `python main.py transfer --help`
   - `git status --short --branch`
2. Read `references/repro-matrix.md` when mapping paper results to commands or when the user asks for "all experiments".
3. Use `scripts/generate_repro_commands.py` to create a runnable bash plan, then review the generated commands before running long jobs.
4. Run a smoke plan first on any new machine. Use full-paper settings only after the smoke plan reaches checkpoint and eval-result creation.
5. Preserve stdout/stderr logs for every command. The repo prints metrics to logs and saves unlearn evaluation dictionaries under each `save_dir`.

## Command Generator

From the repository root:

```bash
python .codex/skills/unlearn-sparse-repro/scripts/generate_repro_commands.py \
  --suite core --runs ./runs/repro --data ./data --seeds 1,2,3 \
  > repro_core.sh
```

Available suites:

- `smoke`: short non-paper sanity run with tiny epoch counts.
- `core`: CIFAR-10/ResNet-18 class-wise and random forgetting for dense and 95%-sparse checkpoints.
- `appendix`: extra dataset/model sweeps that are exposed by the current CLI.
- `backdoor`: Trojan cleanse commands across sparsity levels.
- `imagenet`: ImageNet unlearning command skeletons with required label/HuggingFace prerequisites called out.
- `transfer`: downstream OxfordPets/SUN397 linear-probing commands from ImageNet checkpoints.
- `all`: concatenate all reproducible suites.

The generator only emits commands; it does not submit jobs. If the user asks to run them, ask for the GPU budget or choose a small smoke subset.

## Checkpoints

Do not assume the README mask path is correct for every training profile. In the current unified training pipeline:

- Dense checkpoint from iterative pruning is usually `${SAVE_DIR}/0model_SA_best.pth.tar`.
- Final sparse checkpoint from iterative pruning is usually `${SAVE_DIR}/${PRUNING_TIMES}model_SA_best.pth.tar`.
- SynFlow currently writes `${SAVE_DIR}/0model_SA_best.pth.tar`.

Always verify with `ls -lh "${SAVE_DIR}"/*model_SA_best.pth.tar` before launching unlearning.

## Output Verification

For each unlearning run, verify that these files exist:

- `${SAVE_DIR}/${UNLEARN}checkpoint.pth.tar`
- `${SAVE_DIR}/${UNLEARN}eval_result.pth.tar`

Then load the eval result with `torch.load` and check:

- `accuracy` contains retain/forget/val/test keys for non-ImageNet.
- `SVC_MIA_forget_efficacy` exists after MIA evaluation.
- `SVC_MIA_training_privacy` exists for non-ImageNet privacy evaluation.

If a run is resumed, verify that the saved `evaluation_result` is not silently skipping a metric that the current paper table needs.

## Repro Limits

Be explicit about coverage. The current repo directly covers train/prune, unlearn, MIA evaluation, backdoor cleanse, and downstream transfer through `main.py transfer`. The transfer CLI implements the paper's linear-probing shape, but exact Table 4 reproduction still depends on using the same ImageNet class-removal checkpoints and downstream split files; the repo uses torchvision loaders instead of FFCV.
