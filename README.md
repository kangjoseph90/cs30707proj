# 26S CS30707 Project

## Goal
Investigate how state representation quality affects reinforcement learning
performance in Snake, focusing on state aliasing, empirical Markovianity, and
contrastive learned embeddings.

## Method
Train DQN agents with PyTorch on a Pygame Snake environment. Compare four
state-representation regimes and measure both task performance and
representation quality.

## Experiment matrix

| ID | Representation                                  | Encoder training signal       |
|----|-------------------------------------------------|-------------------------------|
| A  | 85-d hand-crafted windowed state                | n/a (linear DQN on features)  |
| B  | (3, 20, 20) global grid + dir one-hot           | reward only (random init)     |
| C  | (3, 20, 20) global grid + dir one-hot           | reward + **joint CURL**       |
| D  | (3, 20, 20) global grid + dir one-hot           | **CURL pretrain → freeze**    |

Setups C and D adapt CURL (Srinivas, Laskin & Abbeel 2020) to discrete-grid
Snake. The CURL contrastive loss uses (i) two random pad-and-crop
augmentations of the same board, (ii) a bilinear similarity, (iii) an
EMA-updated key encoder, and (iv) the same minibatch that the DQN update
draws from the replay buffer.

## Code layout

```
snake.py                 # Snake env: windowed state + global (3,H,W) state w/ frame-stack hook
model.py                 # DQN (baseline) + SnakeEncoder + ZDQN (encoder + Q-head, optional CURL hook)
curl.py                  # CURLModule: EMA key encoder + bilinear InfoNCE
augment.py               # random_pad_and_crop for grid observations
zdqn_loop.py             # shared ZDQN training loop used by setups B/C/D

train_dqn_window.py      # Setup A
train_zdqn_random.py     # Setup B
train_curl_joint.py      # Setup C (CURL main result)
train_curl_pretrain.py   # Setup D (CURL E.3 ablation: detached encoder)

evaluate.py              # deterministic (epsilon=0) eval of a checkpoint
experiments/run_matrix.ps1   # 4 setups × N seeds, then evaluate each

analysis/
  embed_utils.py         # load encoder, collect rollouts, compute embeddings + GT features
  aliasing.py            # KNN-purity aliasing diagnostic
  markov_probe.py        # supervised probe: z → ground-truth state vector (paper E.5)
  curves.py              # aggregate runs/*/scores.json → learning-curve plot
```

## Usage

### Single run
```pwsh
$env:SDL_VIDEODRIVER = "dummy"     # headless pygame
python train_dqn_window.py    --episodes 300 --seed 0
python train_zdqn_random.py   --episodes 300 --seed 0
python train_curl_joint.py    --episodes 300 --seed 0
python train_curl_pretrain.py --episodes 300 --seed 0 --pretrain-steps 20000 --pretrain-epochs 5
```
Each run writes `runs/{setup}_seed{seed}/{model.pt, scores.json}`.

### Full matrix (4 setups × 3 seeds)
```pwsh
pwsh experiments/run_matrix.ps1 -Episodes 300 -Seeds 0,1,2 -EvalEpisodes 30
```

### Evaluate a checkpoint (deterministic, epsilon=0)
```pwsh
python evaluate.py runs/C_curl_joint_seed0 --episodes 30
```

### Representation-quality analyses
```pwsh
# Latent-space state aliasing (KNN purity over coarse labels)
python -m analysis.aliasing       runs/C_curl_joint_seed0 --steps 4000 -k 5

# Markov probe: how much GT state is recoverable from z?  (lower test_mse better)
python -m analysis.markov_probe   runs/C_curl_joint_seed0 --steps 8000 --epochs 50

# Aggregate learning curves across all runs in runs/
python -m analysis.curves --runs-root runs --out learning_curves.png
```

## Requirements
- Python 3.9+, PyTorch, NumPy, Pygame, matplotlib, scikit-learn
- `pip install -r req.txt` covers the original stack; sklearn is needed for
  the aliasing analysis.
