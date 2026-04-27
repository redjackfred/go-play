# train.py — Self-play training loop + SGF imitation learning for GoNet
#
# Two training pathways:
#
#   1. Self-play (AlphaZero style)
#      python train.py selfplay --games 500 --sims 200 --workers 4 --out model.pt
#
#   2. SGF imitation learning  (recommended cold-start before self-play)
#      python train.py sgf --sgf-dir ./sgf_files --epochs 20 --out model_sgf.pt
#
# Speed flags:
#   --workers N      Parallel game generation (N CPU workers, default 1)
#   --steps-per-game N  Gradient steps per game batch (default 4)
#   --no-augment     Disable 8-fold board symmetry augmentation
#   --compile        Enable torch.compile (PyTorch 2+ only)

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import random
import re
import time
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from board import BLACK, BOARD_SIZE, WHITE, GoEngine
from ai import GoNet, MCTS, create_ai, encode_board

# Normalize score difference to [-1, 1].  A 40-point margin → ±1.0.
# Using score difference (instead of binary win/loss) breaks the komi
# feedback loop: positions won only by komi margin (~6.5 pts) produce
# small values near 0 rather than a sharp ±1 discontinuity.
_SCORE_NORM = 40.0

Example = Tuple[np.ndarray, np.ndarray, float]
# (board_planes: (10,9,9), policy_target: (81,), value_target: ±1)

# ---------------------------------------------------------------------------
# Data augmentation — 16-fold: 8 geometric × 2 color-flip
# ---------------------------------------------------------------------------

# (flip_lr, num_rot90) for each of the 8 board symmetries
_AUG_TRANSFORMS: List[Tuple[bool, int]] = [
    (False, 0), (False, 1), (False, 2), (False, 3),
    (True,  0), (True,  1), (True,  2), (True,  3),
]

# Input plane layout:
#   0  own stones     1  opp stones
#   2  current-player flag (1=Black, 0=White)
#   3  own 1-lib      4  own 2-lib
#   5  opp 1-lib      6  opp 2-lib
#   7  legal moves    8  own eyes   9  opp eyes
_COLOR_SWAP = [1, 0, None, 5, 6, 3, 4, 7, 9, 8]  # None → flip plane 2


def _color_flip(planes: np.ndarray, policy: np.ndarray, value: float
                ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Create a color-swapped twin of one example.

    Swaps Black↔White stone planes, inverts the current-player flag, swaps
    own/opponent liberty and eye planes, keeps policy coordinates (the same
    board cell is strategically equivalent for the mirrored current player),
    and negates value (a win for one colour is a loss for the other).
    """
    p = np.empty_like(planes)
    for dst, src in enumerate(_COLOR_SWAP):
        if src is None:          # plane 2: current-player flag
            p[dst] = 1.0 - planes[dst]
        else:
            p[dst] = planes[src]
    # Plane 7 (legal moves): empty cells ≈ legal for either player
    p[7] = np.clip(1.0 - planes[0] - planes[1], 0.0, 1.0)
    return p, policy.copy(), -value


def _augment_examples(examples: List[Example]) -> List[Example]:
    """Expand each example 16× via rotations, flips, and color swaps."""
    out: List[Example] = []
    for planes, policy, value in examples:
        policy_2d = policy.reshape(BOARD_SIZE, BOARD_SIZE)
        for flip, k in _AUG_TRANSFORMS:
            p   = np.rot90(planes,    k, axes=(1, 2))
            pol = np.rot90(policy_2d, k)
            if flip:
                p   = np.flip(p,   axis=2).copy()
                pol = np.flip(pol, axis=1).copy()
            else:
                p   = p.copy()
                pol = pol.copy()
            pol_flat = pol.flatten()
            out.append((p, pol_flat, value))
            # Color-swapped twin: forces value network to learn board features,
            # not which colour is to move.
            out.append(_color_flip(p, pol_flat, value))
    return out


# ---------------------------------------------------------------------------
# Self-play data generation
# ---------------------------------------------------------------------------

def generate_self_play_game(
    ai: MCTS,
    temp_cutoff: int = 12,
    max_moves: int = 200,
) -> List[Example]:
    """
    Play one game, collecting (state, policy, outcome) per move.

    Temperature schedule:
      - moves 0..temp_cutoff-1 : temperature=1  (explore, stochastic)
      - moves temp_cutoff+      : temperature=0  (exploit, near-deterministic)
    """
    engine = GoEngine()
    history: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for move_num in range(max_moves):
        if engine.game_over:
            break
        if not engine.get_legal_moves():
            engine.pass_move()
            break

        temperature = 1.0 if move_num < temp_cutoff else 0.0
        probs = ai.get_move_probabilities(engine, temperature=temperature)

        policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        for (r, c), p in probs.items():
            policy[r * BOARD_SIZE + c] = p

        history.append((encode_board(engine).numpy(), policy, engine.current_player))

        moves   = list(probs.keys())
        weights = [probs[m] for m in moves]
        engine.play(*random.choices(moves, weights=weights, k=1)[0])

    score      = engine.get_score()
    score_diff = score["black_score"] - score["white_score"]   # + = Black ahead

    return [
        (planes, policy, float(np.clip(
            (1.0 if player == BLACK else -1.0) * score_diff / _SCORE_NORM,
            -1.0, 1.0,
        )))
        for planes, policy, player in history
    ]


# ---------------------------------------------------------------------------
# Parallel game generation worker (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _worker_generate_game(args: Tuple) -> List[Example]:
    """Subprocess entry: generate one self-play game with a fresh model copy."""
    state_dict, sims, temp_cutoff = args
    model = GoNet()
    model.load_state_dict(state_dict)
    ai = MCTS(model, num_simulations=sims, device="cpu", training=True)
    return generate_self_play_game(ai, temp_cutoff=temp_cutoff)


# ---------------------------------------------------------------------------
# Self-play training loop  (replay buffer version)
# ---------------------------------------------------------------------------

def run_self_play(
    num_games: int,
    sims_per_move: int,
    model_path: Optional[str],
    output_path: str,
    final_epochs: int,
    batch_size: int,
    lr: float,
    buffer_size: int,
    deep_every: int,
    temp_cutoff: int,
    save_every: int = 50,
    workers: int = 1,
    steps_per_game: int = 4,
    augment: bool = True,
    use_compile: bool = False,
) -> None:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(
        f"Device: {device}  |  buffer={buffer_size}  sims={sims_per_move}  "
        f"workers={workers}  steps/game={steps_per_game}  "
        f"augment={'8x' if augment else 'off'}  "
        f"temp_cutoff={temp_cutoff}  deep_every={deep_every}  save_every={save_every}"
    )

    model = GoNet()
    if model_path:
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"  Loaded checkpoint: {model_path}")

    if use_compile:
        try:
            model = torch.compile(model)
            print("  torch.compile: active")
        except Exception as e:
            print(f"  torch.compile: skipped — {e}")

    model = model.to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)

    # Single-process AI (used when workers=1)
    ai = MCTS(model, num_simulations=sims_per_move, device=device, training=True)

    # Replay buffer
    buf: Deque[Example]  = collections.deque(maxlen=buffer_size)
    loss_window: Deque[float] = collections.deque(maxlen=20)

    # Use spawn context so workers are safe on macOS and with PyTorch
    ctx = mp.get_context("spawn")

    game_idx = 0
    while game_idx < num_games:
        n_this = min(workers, num_games - game_idx)
        t0 = time.time()

        if n_this > 1:
            # Parallel generation: snapshot model weights to CPU for workers
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            args = [(cpu_state, sims_per_move, temp_cutoff)] * n_this
            with ctx.Pool(n_this) as pool:
                game_batch = pool.map(_worker_generate_game, args)
        else:
            game_batch = [generate_self_play_game(ai, temp_cutoff=temp_cutoff)]

        elapsed = time.time() - t0

        total_moves = 0
        for examples in game_batch:
            if augment:
                examples = _augment_examples(examples)
            total_moves += len(examples)
            buf.extend(examples)

        # Multiple gradient steps per game batch
        step_loss = 0.0
        n_steps = steps_per_game * n_this
        for _ in range(n_steps):
            if len(buf) >= batch_size:
                step_loss = _train_batch(model, opt, random.sample(list(buf), batch_size), device)
                loss_window.append(step_loss)

        game_idx += n_this
        avg = sum(loss_window) / len(loss_window) if loss_window else 0.0
        print(
            f"Game {game_idx:4d}/{num_games}  "
            f"examples={total_moves:4d}  "
            f"buf={len(buf):6d}  "
            f"time={elapsed:.1f}s  "
            f"loss={step_loss:.4f}  avg20={avg:.4f}"
        )

        # Periodic deep pass
        if game_idx % deep_every < n_this and len(buf) >= batch_size:
            print(f"  >> Deep pass ({len(buf)} examples, 3 epochs, lr={lr*0.5:.2e})")
            opt_deep = optim.Adam(model.parameters(), lr=lr * 0.5)
            _train_on_examples(model, opt_deep, list(buf), epochs=3,
                               batch_size=batch_size, device=device)
            for g in opt.param_groups:
                g["lr"] = lr
            # Refresh single-process AI after deep pass (model weights changed)
            ai = MCTS(model, num_simulations=sims_per_move, device=device, training=True)

        # Periodic checkpoint
        if save_every > 0 and game_idx % save_every < n_this:
            ckpt = output_path.replace(".pt", f"_ckpt{game_idx}.pt")
            torch.save(_unwrap_state_dict(model), ckpt)
            print(f"  >> Checkpoint saved → {ckpt}")

    # Final training pass
    print(f"\nFinal training pass ({final_epochs} epochs, lr={lr*0.3:.2e}) ...")
    opt_final = optim.Adam(model.parameters(), lr=lr * 0.3)
    _train_on_examples(model, opt_final, list(buf), epochs=final_epochs,
                       batch_size=batch_size, device=device)

    torch.save(_unwrap_state_dict(model), output_path)
    print(f"Model saved → {output_path}")


def _unwrap_state_dict(model: nn.Module) -> dict:
    """Return state_dict, unwrapping torch.compile wrapper if present."""
    m = getattr(model, "_orig_mod", model)
    return m.state_dict()


# ---------------------------------------------------------------------------
# SGF imitation learning  (useful for cold-start before self-play)
# ---------------------------------------------------------------------------

def _parse_sgf_moves(text: str) -> List[Tuple[int, int, int]]:
    moves = []
    for m in re.finditer(r'([BW])\[([a-i]{2})\]', text):
        color_char, coord = m.group(1), m.group(2)
        col = ord(coord[0]) - ord('a')
        row = ord(coord[1]) - ord('a')
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            moves.append((BLACK if color_char == 'B' else WHITE, row, col))
    return moves


def load_sgf_examples(sgf_dir: str, augment: bool = True) -> List[Example]:
    examples: List[Example] = []
    files = list(Path(sgf_dir).glob("**/*.sgf"))
    print(f"Found {len(files)} SGF files in {sgf_dir}")

    for path in files:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        m = re.search(r'RE\[([BW])', text, re.IGNORECASE)
        if not m:
            continue
        sgf_winner = BLACK if m.group(1).upper() == 'B' else WHITE

        moves = _parse_sgf_moves(text)
        if not moves:
            continue

        engine = GoEngine()
        for color, row, col in moves:
            if engine.current_player != color or not engine.is_legal(row, col):
                break
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            policy[row * BOARD_SIZE + col] = 1.0
            examples.append((
                encode_board(engine).numpy(),
                policy,
                1.0 if color == sgf_winner else -1.0,
            ))
            engine.play(row, col)

    print(f"Loaded {len(examples)} examples")
    if augment and examples:
        examples = _augment_examples(examples)
        print(f"After 8-fold augmentation: {len(examples)} examples")
    return examples


def run_sgf_training(
    sgf_dir: str,
    model_path: Optional[str],
    output_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    augment: bool = True,
    use_compile: bool = False,
) -> None:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    model = GoNet()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded base model from {model_path}")

    if use_compile:
        try:
            model = torch.compile(model)
            print("  torch.compile: active")
        except Exception as e:
            print(f"  torch.compile: skipped — {e}")

    examples = load_sgf_examples(sgf_dir, augment=augment)
    if not examples:
        print("No examples found.")
        return
    opt = optim.Adam(model.parameters(), lr=lr)
    _train_on_examples(model, opt, examples, epochs=epochs,
                       batch_size=batch_size, device=device)
    torch.save(_unwrap_state_dict(model), output_path)
    print(f"Model saved → {output_path}")


# ---------------------------------------------------------------------------
# Training primitives
# ---------------------------------------------------------------------------

def _train_batch(
    model: nn.Module,
    opt: optim.Optimizer,
    batch: List[Example],
    device: str,
) -> float:
    model.train(True)
    planes = torch.from_numpy(np.stack([e[0] for e in batch])).to(device)
    policy = torch.from_numpy(np.stack([e[1] for e in batch])).to(device)
    value  = torch.tensor([e[2] for e in batch], dtype=torch.float32,
                          device=device).unsqueeze(1)

    log_p, v = model(planes)
    p_loss = -(policy * log_p).sum(dim=1).mean()
    v_loss = nn.functional.mse_loss(v, value)
    loss   = p_loss + v_loss

    opt.zero_grad()
    loss.backward()
    opt.step()
    model.train(False)
    return loss.item()


def _train_on_examples(
    model: nn.Module,
    opt: optim.Optimizer,
    examples: List[Example],
    epochs: int,
    batch_size: int,
    device: str,
) -> None:
    model.to(device)
    model.train(True)

    ds = TensorDataset(
        torch.from_numpy(np.stack([e[0] for e in examples])),
        torch.from_numpy(np.stack([e[1] for e in examples])),
        torch.tensor([e[2] for e in examples], dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(1, epochs + 1):
        tot = p_tot = v_tot = 0.0
        for x, p_target, v_target in loader:
            x, p_target = x.to(device), p_target.to(device)
            v_target = v_target.to(device).unsqueeze(1)
            log_p, v = model(x)
            p_loss = -(p_target * log_p).sum(dim=1).mean()
            v_loss = nn.functional.mse_loss(v, v_target)
            loss   = p_loss + v_loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); p_tot += p_loss.item(); v_tot += v_loss.item()
        n = max(len(loader), 1)
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"loss={tot/n:.4f}  p={p_tot/n:.4f}  v={v_tot/n:.4f}")

    model.train(False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GoNet")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("selfplay")
    sp.add_argument("--games",          type=int,   default=500)
    sp.add_argument("--sims",           type=int,   default=100,
                    help="MCTS sims/move (higher = better signal, slower)")
    sp.add_argument("--epochs",         type=int,   default=5,
                    help="Final training epochs after all games")
    sp.add_argument("--batch",          type=int,   default=256)
    sp.add_argument("--lr",             type=float, default=1e-3)
    sp.add_argument("--buffer",         type=int,   default=20_000,
                    help="Replay buffer size")
    sp.add_argument("--deep-every",     type=int,   default=50,
                    help="3-epoch deep pass every N games")
    sp.add_argument("--temp-cutoff",    type=int,   default=12,
                    help="temperature=1 for first N moves, then 0")
    sp.add_argument("--save-every",     type=int,   default=50,
                    help="Checkpoint every N games (0 = disable)")
    sp.add_argument("--workers",        type=int,   default=1,
                    help="Parallel game-generation workers (uses CPU)")
    sp.add_argument("--steps-per-game", type=int,   default=4,
                    help="Gradient steps per game (or per worker-batch)")
    sp.add_argument("--no-augment",     action="store_true",
                    help="Disable 8-fold board symmetry augmentation")
    sp.add_argument("--compile",        action="store_true",
                    help="torch.compile the model (PyTorch 2+ only)")
    sp.add_argument("--model",          type=str,   default=None)
    sp.add_argument("--out",            type=str,   default="model.pt")

    sg = sub.add_parser("sgf")
    sg.add_argument("--sgf-dir",    type=str,   required=True)
    sg.add_argument("--epochs",     type=int,   default=10)
    sg.add_argument("--batch",      type=int,   default=256)
    sg.add_argument("--lr",         type=float, default=1e-3)
    sg.add_argument("--no-augment", action="store_true",
                    help="Disable 8-fold board symmetry augmentation")
    sg.add_argument("--compile",    action="store_true")
    sg.add_argument("--model",      type=str,   default=None)
    sg.add_argument("--out",        type=str,   default="model.pt")

    args = parser.parse_args()

    if args.cmd == "selfplay":
        run_self_play(
            num_games=args.games,
            sims_per_move=args.sims,
            model_path=args.model,
            output_path=args.out,
            final_epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            buffer_size=args.buffer,
            deep_every=args.deep_every,
            temp_cutoff=args.temp_cutoff,
            save_every=args.save_every,
            workers=args.workers,
            steps_per_game=args.steps_per_game,
            augment=not args.no_augment,
            use_compile=args.compile,
        )
    elif args.cmd == "sgf":
        run_sgf_training(
            sgf_dir=args.sgf_dir,
            model_path=args.model,
            output_path=args.out,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            augment=not args.no_augment,
            use_compile=args.compile,
        )


if __name__ == "__main__":
    main()
