# main.py — Entry point for the 9x9 Go engine

"""
Modes
-----
python main.py              → launch GUI (human=Black vs AI=White)
python main.py --selfplay   → AI vs AI on the console (no GUI)
python main.py --perft N    → run N games of AI self-play, print win stats
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from board import BLACK, BOARD_SIZE, WHITE, GoEngine, opponent
from ai import create_ai, MCTS


# ---------------------------------------------------------------------------
# Console self-play helpers
# ---------------------------------------------------------------------------

_COL_LETTERS = "ABCDEFGHJ"


def _move_str(move: Optional[tuple]) -> str:
    if move is None:
        return "PASS (concede)"
    r, c = move
    return f"{_COL_LETTERS[c]}{BOARD_SIZE - r}"


def _print_board(engine: GoEngine) -> None:
    header = "   " + " ".join(_COL_LETTERS[:BOARD_SIZE])
    print(header)
    for r in range(BOARD_SIZE):
        row_num = str(BOARD_SIZE - r).rjust(2)
        cells = []
        for c in range(BOARD_SIZE):
            v = engine.board[r][c]
            if v == BLACK:
                cells.append("X")
            elif v == WHITE:
                cells.append("O")
            else:
                cells.append(".")
        mark = ""
        if engine.last_move and engine.last_move[0] == r:
            lc = engine.last_move[1]
            cells[lc] = "x" if engine.board[r][lc] == BLACK else "o"
        print(f"{row_num} " + " ".join(cells))
    print()


def _play_game(
    ai_black: MCTS,
    ai_white: MCTS,
    verbose: bool = True,
    max_moves: int = BOARD_SIZE * BOARD_SIZE * 2,
) -> int:
    """Play one complete game.  Returns winner (BLACK or WHITE)."""
    engine = GoEngine()
    ais = {BLACK: ai_black, WHITE: ai_white}
    player_name = {BLACK: "Black(X)", WHITE: "White(O)"}
    move_number = 0

    while not engine.game_over and move_number < max_moves:
        ai = ais[engine.current_player]
        move = ai.select_move(engine)
        move_number += 1

        if verbose:
            print(f"Move {move_number:3d}  {player_name[engine.current_player]}: "
                  f"{_move_str(move)}")

        if ai.check_resign(engine):
            engine.resign()
        elif move is None:
            engine.pass_move()
        else:
            engine.play(*move)

        if verbose and move_number % 10 == 0:
            _print_board(engine)

    if engine.game_over:
        # Ended by pass/concede.
        winner = engine.winner
    else:
        # Max-move cap: score the position.
        score = engine.get_score()
        winner = score["winner"]

    if verbose:
        _print_board(engine)
        if engine.game_over:
            loser_name = player_name[opponent(winner)]
            print(f"Game over — {loser_name} conceded.")
        score = engine.get_score()
        print(
            f"Score  Black: {score['black_score']:.1f}  "
            f"White: {score['white_score']:.1f}  "
            f"(komi {score['komi']})"
        )
        print(f"Winner: {player_name[winner]}")

    return winner


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

def run_gui(sims: int, model_path: Optional[str] = None) -> None:
    try:
        import gui
        gui.run(num_simulations=sims, model_path=model_path)
    except ImportError as e:
        print(f"GUI unavailable: {e}")
        print("Install pygame:  pip install pygame")
        sys.exit(1)


def run_selfplay(sims: int, model_path: Optional[str] = None) -> None:
    label = model_path or "random-weight network"
    print(f"Self-play: MCTS {sims} sims/move, {label}")
    print("=" * 50)
    # training=True adds Dirichlet noise at root for diverse, non-deterministic play
    ai = create_ai(num_simulations=sims, model_path=model_path, training=True)
    _play_game(ai, ai, verbose=True)


def run_perft(sims: int, n_games: int, model_path: Optional[str] = None) -> None:
    print(f"Perft: {n_games} games, {sims} sims/move")
    # training=True adds Dirichlet noise at root so games are non-deterministic
    ai = create_ai(num_simulations=sims, model_path=model_path, training=True)
    wins = {BLACK: 0, WHITE: 0}
    t0 = time.time()

    for i in range(n_games):
        winner = _play_game(ai, ai, verbose=False)
        wins[winner] += 1
        elapsed = time.time() - t0
        print(f"  Game {i+1:3d}  winner={'Black' if winner==BLACK else 'White'}"
              f"  B:{wins[BLACK]}  W:{wins[WHITE]}"
              f"  elapsed={elapsed:.1f}s")

    total = wins[BLACK] + wins[WHITE]
    print(f"\nResults over {total} games:")
    print(f"  Black wins: {wins[BLACK]} ({100*wins[BLACK]/total:.1f}%)")
    print(f"  White wins: {wins[WHITE]} ({100*wins[WHITE]/total:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="9x9 Go engine with MCTS+NN AI"
    )
    parser.add_argument(
        "--selfplay", action="store_true",
        help="Run one AI vs AI game on the console"
    )
    parser.add_argument(
        "--perft", type=int, metavar="N", default=0,
        help="Run N AI self-play games and print win statistics"
    )
    parser.add_argument(
        "--sims", type=int, default=200,
        help="MCTS simulations per move (default 200)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model weights (e.g. model_v3.pt)"
    )
    args = parser.parse_args()

    if args.perft > 0:
        run_perft(args.sims, args.perft, args.model)
    elif args.selfplay:
        run_selfplay(args.sims, args.model)
    else:
        run_gui(args.sims, args.model)


if __name__ == "__main__":
    main()
