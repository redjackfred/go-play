# 9×9 Go Engine with MCTS + Neural Network AI

From-scratch Python implementation — Chinese rules, no-pass variant, AlphaZero-style AI.

---

## Files

| File | Role |
|------|------|
| `board.py` | `GoEngine` — core rules, legality, scoring, public API |
| `ai.py` | `GoNet` (PyTorch ResNet) + `MCTS` search |
| `gui.py` | pygame GUI — 4 game modes, colour selection, captured counts |
| `main.py` | CLI entry point |
| `train.py` | Self-play training loop + SGF imitation learning |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch CPU-only (smaller download):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pygame
```

---

## Running

### GUI — 4-mode game

```bash
python main.py
```

A start menu appears.  Choose:

| Mode | Description |
|------|-------------|
| Human vs AI | You click, AI responds |
| AI vs Human | AI moves first, you respond |
| Human vs Human | Two humans take turns |
| AI vs AI | Watch two MCTS agents play |

Then choose **your colour** (Black / White) and **AI strength** (50 – 800 sims/move).

Click **Start Game**.  During play, click any intersection to place a stone.
**New Game** resets; **Menu** returns to the start screen.

```bash
python main.py --sims 400      # override default AI strength
```

### Console self-play (no GUI)

```bash
python main.py --selfplay
python main.py --selfplay --sims 50
```

### Benchmark — N games, print win rates

```bash
python main.py --perft 10 --sims 50
```

---

## Training the neural network

### Self-play (AlphaZero style)

```bash
python train.py selfplay --games 200 --sims 50 --epochs 5 --out model.pt
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--games` | 100 | Number of self-play games to generate |
| `--sims` | 50 | MCTS simulations per move during generation |
| `--epochs` | 5 | Training epochs after data collection |
| `--batch` | 256 | Mini-batch size |
| `--lr` | 0.001 | Learning rate |
| `--model` | — | Start from an existing checkpoint |
| `--out` | model.pt | Where to save the trained weights |

### SGF imitation learning

Point `--sgf-dir` at a folder containing `.sgf` files (9×9 games).
Free 9×9 SGF collections: GoGoD, KGS, or CGOS archives.

```bash
python train.py sgf --sgf-dir ./sgf_files --epochs 20 --out model.pt
```

### Using a trained model

```bash
python main.py --sims 400          # then set model_path in code, or:
```

```python
from ai import create_ai
ai = create_ai(num_simulations=400, model_path="model.pt")
```

---

## Rules (assignment spec)

| Rule | Behaviour |
|------|-----------|
| Board | 9×9, Black moves first |
| Captures | Groups at zero liberties removed immediately |
| Suicide | Illegal unless move simultaneously captures opponent stones |
| Ko | Simple ko: move that recreates the position before last half-move is forbidden |
| **Pass** | **Pass = immediate concession — opponent wins** |
| Scoring | Chinese area scoring: stones + enclosed empty intersections |
| Komi | 2.5 (awarded to White) |

---

## Module self-tests

```bash
python board.py   # 9 rule-engine unit tests (capture, ko, suicide, scoring…)
python ai.py      # 4-move AI sanity check with random-weight network
```

---

## AI architecture notes

```
Input (3 × 9 × 9)
  plane 0: Black stones
  plane 1: White stones
  plane 2: current-player flag (all-1 for Black, all-0 for White)

Stem conv → 3 × ResBlock (64 ch) → split:

  Policy head:  conv(2ch) → BN → ReLU → FC(81) → log-softmax
  Value head:   conv(1ch) → BN → ReLU → FC(64) → FC(1) → tanh
```

MCTS uses PUCT selection (`−Q + c·P·√N / (1+n)`).  Values are stored from
each node's own player's perspective; sign flips on backpropagation.

The network starts with **random weights** and plays legal moves immediately.
Train it with `train.py` to develop positional knowledge.

### Approximate play strength vs sims (CPU, random weights)

| `--sims` | Time/move | Level |
|----------|-----------|-------|
| 50 | < 1 s | beginner |
| 200 | 2–5 s | moderate |
| 400 | 8–15 s | stronger |
| 800 | 20–40 s | strong |
