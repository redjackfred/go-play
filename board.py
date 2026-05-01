# board.py
# ==============================================================================
# 9×9 Go Engine — Core Rules and Public API
# ==============================================================================
#
# (a) AI Strategy: Why MCTS + Neural Network?
#
#     Pure MCTS explores the game tree by running random rollouts from each
#     candidate node and using accumulated win rates to bias further search.
#     Its key strength: no hand-crafted evaluation function is needed — only
#     the game rules — making it domain-agnostic and surprisingly strong.
#
#     The weakness of plain MCTS is that random rollouts are noisy guides to
#     Go's positional nuance.  Adding a lightweight CNN (Policy Head + Value
#     Head, in the AlphaZero style) fixes both problems at once:
#       • Policy head → prior probability over moves, steering PUCT expansion
#         toward promising children without exhaustive rollouts.
#       • Value head → scalar board evaluation replacing full random games,
#         drastically reducing variance per node visit.
#     Even a small network (3–4 residual blocks on a 9×9 board) provides a
#     meaningful signal that elevates play well above random-rollout MCTS.
#     PyTorch makes the forward pass fast enough to stay within a time budget.
#
# (b) Ko and Liberties — Design Decisions & Challenges
#
#     Liberties: A stone group is the maximal 4-connected component of same-
#     colored stones.  BFS from any member collects the whole group; empty
#     orthogonal neighbors are counted as liberties.  We recompute on every
#     call (O(N²) worst case) which is fine for 9×9.  A production engine
#     would maintain incremental counts via a union-find structure.
#
#     Ko (simple ko, matching Chinese rules): Before each move we snapshot the
#     board state.  In is_legal() we simulate the move — place the stone,
#     capture any opponent groups that reach zero liberties — then compare the
#     resulting board against the snapshot.  Identity means ko: rejected.
#     The snapshot is taken BEFORE the move in play(), so it represents the
#     board two half-moves ago from the perspective of the player about to
#     move, which is exactly the position that must not be recreated.
#     Challenge: the simulate-and-restore pattern inside is_legal() must
#     perfectly undo both the placed stone and all captured opponent stones,
#     otherwise incremental board state corrupts.  We track every captured
#     cell explicitly and restore them one by one.
#
# (c) Manual Testing
#
#     1. Self-test suite:
#           python board.py
#        Covers: capture, suicide rejection, ko rejection, scoring structure,
#        legal-move enumeration on an empty board.
#
#     2. Interactive Python shell:
#           from board import GoEngine, BLACK, WHITE
#           g = GoEngine()
#           g.play(3, 3)          # Black at D4
#           g.play(3, 4)          # White at E4
#           print(g.get_board())
#           print(g.get_legal_moves())
#           print(g.get_score())
#
#     3. GUI:
#           python gui.py         # click intersections to place stones
#
#     4. AI self-play:
#           python main.py --selfplay
#        Prints moves to console; watch MCTS agents compete.
#
# ==============================================================================

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

EMPTY = 0
BLACK = 1
WHITE = 2
BOARD_SIZE = 9
KOMI = 2.5


def opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


class GoEngine:
    """
    9×9 Go engine — Chinese area scoring.
    Two consecutive passes end the game (area scoring decides winner).
    """

    def __init__(self) -> None:
        self.new_game()

    # ------------------------------------------------------------------ #
    #  Public API (required by test harness)                              #
    # ------------------------------------------------------------------ #

    def new_game(self) -> None:
        """Reset to a fresh 9×9 board. Black moves first."""
        self.board: List[List[int]] = [
            [EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)
        ]
        self.current_player: int = BLACK
        self.previous_board: Optional[List[List[int]]] = None  # for ko detection
        self.captured: Dict[int, int] = {BLACK: 0, WHITE: 0}  # stones captured BY each color
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.last_move: Optional[Tuple[int, int]] = None
        self.consecutive_passes: int = 0

    def play(self, row: int, col: int) -> bool:
        """
        Place a stone for the current player at (row, col).
        Returns True on success, False if illegal or game already over.
        """
        if not self.is_legal(row, col):
            return False

        opp = opponent(self.current_player)

        # Snapshot current board BEFORE the move (used for ko on the next move).
        self.previous_board = [r[:] for r in self.board]

        # Place stone.
        self.board[row][col] = self.current_player

        # Capture every adjacent opponent group that now has zero liberties.
        checked: Set[Tuple[int, int]] = set()
        for nr, nc in self._neighbors(row, col):
            if self.board[nr][nc] == opp and (nr, nc) not in checked:
                group, liberties = self._get_group(nr, nc)
                checked |= group
                if not liberties:
                    self._remove_group(group)
                    self.captured[self.current_player] += len(group)

        self.consecutive_passes = 0
        self.last_move = (row, col)
        self.current_player = opp
        return True

    def is_legal(self, row: int, col: int) -> bool:
        """Return True iff placing a stone at (row, col) is legal for the current player."""
        if self.game_over:
            return False
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        if self.board[row][col] != EMPTY:
            return False

        opp = opponent(self.current_player)

        # --- Simulate the move on the live board, then restore. ---
        self.board[row][col] = self.current_player

        # Identify and temporarily remove opponent groups at zero liberties.
        captured: Set[Tuple[int, int]] = set()
        checked: Set[Tuple[int, int]] = set()
        for nr, nc in self._neighbors(row, col):
            if self.board[nr][nc] == opp and (nr, nc) not in checked:
                group, liberties = self._get_group(nr, nc)
                checked |= group
                if not liberties:
                    captured |= group
        for r, c in captured:
            self.board[r][c] = EMPTY

        # Suicide check: own group must have at least one liberty after all captures.
        _, own_liberties = self._get_group(row, col)
        is_suicide = not own_liberties

        # Ko check: resulting position must not equal the board before the last move.
        is_ko = (
            self.previous_board is not None
            and [r[:] for r in self.board] == self.previous_board
        )

        # Restore board to original state.
        self.board[row][col] = EMPTY
        for r, c in captured:
            self.board[r][c] = opp

        return not is_suicide and not is_ko

    def get_board(self) -> List[List[int]]:
        """Return a copy of the board (0 = empty, 1 = black, 2 = white)."""
        return [r[:] for r in self.board]

    def pass_move(self) -> None:
        """Current player passes.  Two consecutive passes end the game (area scoring)."""
        self.consecutive_passes += 1
        self.previous_board = None   # pass clears the ko restriction
        self.last_move = None
        if self.consecutive_passes >= 2:
            self.game_over = True
            self.winner = self.get_score()["winner"]
        self.current_player = opponent(self.current_player)  # always switch

    def decline_pass(self) -> None:
        """Current player rejects the opponent's pending pass.

        Cancels the most recent pass and returns the turn to the player
        who passed, so they can make a different move.
        """
        if self.game_over or self.consecutive_passes <= 0:
            return
        self.consecutive_passes -= 1
        self.current_player = opponent(self.current_player)
        # The ko restriction was already cleared by pass_move and stays cleared.

    def resign(self) -> None:
        """Current player resigns — opponent wins immediately."""
        self.game_over = True
        self.winner = opponent(self.current_player)

    def get_score(self) -> Dict:
        """
        Chinese area scoring: stones + surrounded empty intersections.
        White receives +KOMI (6.5).
        Returns a dict with full breakdown and winner.
        """
        black_stones = sum(
            self.board[r][c] == BLACK
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
        )
        white_stones = sum(
            self.board[r][c] == WHITE
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
        )

        black_territory, white_territory = self._count_territory()

        black_score = float(black_stones + black_territory)
        white_score = float(white_stones + white_territory) + KOMI

        winner = BLACK if black_score > white_score else WHITE

        return {
            "black_score": black_score,
            "white_score": white_score,
            "black_stones": black_stones,
            "white_stones": white_stones,
            "black_territory": black_territory,
            "white_territory": white_territory,
            "komi": KOMI,
            "winner": winner,
            "winner_name": "Black" if winner == BLACK else "White",
            "margin": abs(black_score - white_score),
        }

    # ------------------------------------------------------------------ #
    #  Helpers (used by GUI and AI)                                        #
    # ------------------------------------------------------------------ #

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """All legal (row, col) moves for the current player."""
        return [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if self.is_legal(r, c)
        ]

    def clone(self) -> GoEngine:
        """Fast deep copy of engine state — used by MCTS rollouts."""
        g = GoEngine.__new__(GoEngine)
        g.board = [r[:] for r in self.board]
        g.current_player = self.current_player
        g.previous_board = (
            [r[:] for r in self.previous_board] if self.previous_board else None
        )
        g.captured = dict(self.captured)
        g.game_over = self.game_over
        g.winner = self.winner
        g.last_move = self.last_move
        g.consecutive_passes = self.consecutive_passes
        return g

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _neighbors(self, row: int, col: int):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                yield r, c

    def _get_group(
        self, row: int, col: int
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """BFS: returns (group_cells, liberty_cells) for the stone at (row, col)."""
        color = self.board[row][col]
        if color == EMPTY:
            return set(), set()

        group: Set[Tuple[int, int]] = set()
        liberties: Set[Tuple[int, int]] = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            group.add((r, c))
            for nr, nc in self._neighbors(r, c):
                cell = self.board[nr][nc]
                if cell == color and (nr, nc) not in group:
                    stack.append((nr, nc))
                elif cell == EMPTY:
                    liberties.add((nr, nc))

        return group, liberties

    def _remove_group(self, group: Set[Tuple[int, int]]) -> None:
        for r, c in group:
            self.board[r][c] = EMPTY

    def _flood_fill_empty(
        self, row: int, col: int
    ) -> Tuple[Set[Tuple[int, int]], Set[int]]:
        """BFS from an empty cell. Returns (region, set_of_bordering_colors)."""
        region: Set[Tuple[int, int]] = set()
        border_colors: Set[int] = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in region:
                continue
            region.add((r, c))
            for nr, nc in self._neighbors(r, c):
                cell = self.board[nr][nc]
                if cell == EMPTY and (nr, nc) not in region:
                    stack.append((nr, nc))
                elif cell != EMPTY:
                    border_colors.add(cell)

        return region, border_colors

    def _count_territory(self) -> Tuple[int, int]:
        """Flood-fill all empty regions; attribute to Black, White, or neither."""
        visited: Set[Tuple[int, int]] = set()
        black_t = white_t = 0

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY and (r, c) not in visited:
                    region, colors = self._flood_fill_empty(r, c)
                    visited |= region
                    if colors == {BLACK}:
                        black_t += len(region)
                    elif colors == {WHITE}:
                        white_t += len(region)
                    # Region touching both colors: neutral, counts for neither.

        return black_t, white_t


# ------------------------------------------------------------------ #
#  Self-test suite                                                     #
# ------------------------------------------------------------------ #

def _run_tests() -> None:
    print("Running board.py self-tests...")

    # Test 1: basic placement and turn switching
    g = GoEngine()
    assert g.play(0, 0), "Black should play at (0,0)"
    assert g.board[0][0] == BLACK
    assert g.current_player == WHITE
    assert g.play(1, 1), "White should play at (1,1)"
    assert g.board[1][1] == WHITE
    assert g.current_player == BLACK
    print("  PASS: basic placement + turn switching")

    # Test 2: capture on the corner
    # Black at (0,0), White fills (0,1) and (1,0) → Black captured
    g = GoEngine()
    g.play(0, 0)   # Black
    g.play(0, 1)   # White
    g.play(4, 4)   # Black (elsewhere, to give White a turn)
    g.play(1, 0)   # White — captures Black at (0,0)
    assert g.board[0][0] == EMPTY, "Black at (0,0) should be captured"
    assert g.captured[WHITE] == 1
    print("  PASS: capture")

    # Test 3: suicide rejection
    # White walls off (0,0) with stones at (0,1) and (1,0); Black's turn
    g = GoEngine()
    g.board[0][1] = WHITE
    g.board[1][0] = WHITE
    # (0,0) has zero liberties for any placed stone → suicide
    assert not g.is_legal(0, 0), "Suicide at (0,0) must be illegal"
    print("  PASS: suicide rejection")

    # Test 4: suicide-that-is-actually-a-capture is legal
    # Black surrounds a single White stone at (0,0) on three sides already
    # (via board injection), then plays the final liberty — that captures White,
    # giving Black's new stone liberties.
    g = GoEngine()
    g.board[0][1] = BLACK  # Black
    g.board[1][0] = BLACK  # Black
    g.board[0][0] = WHITE  # lone White at corner
    # Black at (0,0)'s group would have 0 liberties BUT it first captures White
    # Result: (0,0) empty → Black's new stone at... wait, this is placing Black at (0,0)
    # After placing Black at (0,0): White group {(0,0)} loses its last... no.
    # Let's set up properly: White at a different spot.
    g = GoEngine()
    # White is at (1,1), surrounded by Black on three sides; Black fills last liberty.
    # White at (1,1), liberty only at (0,1).
    g.board[1][0] = BLACK
    g.board[1][2] = BLACK
    g.board[2][1] = BLACK
    g.board[1][1] = WHITE
    # Black plays (0,1): neighbors of (0,1) include (1,1)=White which will be captured.
    # After capture, (0,1) has liberties from (0,0),(0,2) → legal.
    assert g.is_legal(0, 1), "Capturing move that looks like suicide must be legal"
    print("  PASS: capture-not-suicide")

    # Test 5: real ko rejection
    # Classic interior ko shape:
    # row/col  0  1  2  3
    #   0      .  B  W  .
    #   1      B  W  .  W    <- (1,2) is the ko point
    #   2      .  B  W  .
    #
    # W at (1,1): neighbors (0,1)=B,(1,0)=B,(1,2)=empty,(2,1)=B → 1 liberty at (1,2)
    # Black plays (1,2) → W at (1,1) captured.
    # White immediately wants to recapture at (1,1) → would recreate the board → KO.
    g = GoEngine()
    g.board[0][1] = BLACK
    g.board[0][2] = WHITE
    g.board[1][0] = BLACK
    g.board[1][1] = WHITE
    g.board[1][3] = WHITE
    g.board[2][1] = BLACK
    g.board[2][2] = WHITE
    # Black plays the ko capture at (1,2)
    assert g.is_legal(1, 2), "Black should be able to play the ko capture at (1,2)"
    g.play(1, 2)
    assert g.board[1][1] == EMPTY, "White at (1,1) should be captured"
    assert g.board[1][2] == BLACK
    # White tries immediate recapture → ko violation
    assert not g.is_legal(1, 1), "Ko: White must not immediately recapture at (1,1)"
    print("  PASS: ko rejection")

    # Test 6: two consecutive passes end the game
    g = GoEngine()
    g.pass_move()                       # Black passes
    assert not g.game_over,             "single pass must not end game"
    assert g.current_player == WHITE,   "White to move after Black passes"
    assert g.consecutive_passes == 1
    g.pass_move()                       # White passes
    assert g.game_over,                 "two passes must end game"
    assert g.winner == g.get_score()["winner"]
    print("  PASS: two-pass ends game")

    # Test 6b: a real move resets the consecutive-pass counter
    g = GoEngine()
    g.pass_move()           # Black passes (consecutive_passes = 1)
    g.play(4, 4)            # White plays → resets counter
    assert g.consecutive_passes == 0
    g.pass_move()           # Black passes again (consecutive_passes = 1)
    assert not g.game_over, "counter should have reset after White's play"
    print("  PASS: pass counter resets on real move")

    # Test 6c: resign ends game immediately
    g = GoEngine()
    g.resign()
    assert g.game_over
    assert g.winner == WHITE  # Black resigned
    print("  PASS: resign")

    # Test 7: scoring structure and komi
    g = GoEngine()
    score = g.get_score()
    assert "black_score" in score and "white_score" in score
    # On empty board all territory is neutral → both scores equal their stone counts + komi
    assert score["white_score"] == score["white_stones"] + score["white_territory"] + KOMI
    assert score["winner"] in (BLACK, WHITE)
    print("  PASS: scoring structure + komi")

    # Test 8: legal moves on empty board
    g = GoEngine()
    moves = g.get_legal_moves()
    assert len(moves) == BOARD_SIZE * BOARD_SIZE
    print("  PASS: legal moves on empty board")

    # Test 9: clone does not share state
    g = GoEngine()
    g.play(4, 4)
    g2 = g.clone()
    g2.play(0, 0)
    assert g.board[0][0] == EMPTY, "Original must be unaffected by clone's moves"
    print("  PASS: clone isolation")

    print("\nAll tests passed.")


if __name__ == "__main__":
    _run_tests()
