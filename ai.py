# ai.py — MCTS + Neural Network AI for 9x9 Go
#
# Input planes (10 total):
#   0  own stones
#   1  opponent stones
#   2  current-player flag (1=Black, 0=White)
#   3  own groups with 1 liberty  (in atari — must escape)
#   4  own groups with 2 liberties
#   5  opponent groups with 1 liberty  (can be captured immediately)
#   6  opponent groups with 2 liberties
#   7  legal moves for current player
#   8  potential own eyes  (empty cell, all orthogonal nbrs = own stones)
#   9  potential opponent eyes
#
# ---------------------------------------------------------------------------
# Architecture overview
# ---------------------------------------------------------------------------
# This file implements an AlphaZero-style Go AI with three layers:
#
#   1. Board encoding (encode_board): turns a GoEngine position into a
#      (10, 9, 9) tensor combining stones, liberties, legal moves, and
#      eye-shape hints — features the network would otherwise have to
#      learn from scratch.
#
#   2. Neural network (GoNet): a small residual policy+value network.
#      The policy head outputs a log-prob over 81 intersections; the
#      value head outputs a tanh score in [-1, +1] from the side-to-move's
#      perspective.
#
#   3. MCTS search (MCTS / MCTSNode): AlphaZero-style PUCT search with
#      batched leaf evaluation, virtual loss for parallel rollouts,
#      Dirichlet exploration noise at the root during training, and
#      multiplicative prior shaping (spatial / eye / capture / escape /
#      connection-penalty) layered on top of the network policy.
#
# Move selection in `select_move`:
#   All moves go through MCTS (num_simulations PUCT rollouts).  Tactical
#   knowledge (captures, atari threats, escapes, eye formation, etc.) is
#   injected via multiplicative prior biases in `_create_children` so the
#   search naturally prefers strong moves without bypassing the tree.
#   The pass move is a special child handled separately — see `_create_children`
#   for the training-vs-gameplay difference.

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import BLACK, BOARD_SIZE, EMPTY, WHITE, GoEngine, opponent

# ---------------------------------------------------------------------------
# Spatial prior heatmap  (金角銀邊草肚皮 + 3rd/4th line preference)
# ---------------------------------------------------------------------------

_LINE_SCORES = [0.35, 0.75, 1.45, 1.80, 0.60]


def _make_spatial_prior(size: int) -> np.ndarray:
    half = size // 2
    grid = np.empty((size, size), dtype=np.float32)
    for r in range(size):
        for c in range(size):
            dr = min(r, size - 1 - r)
            dc = min(c, size - 1 - c)
            grid[r, c] = _LINE_SCORES[min(dr, half)] * _LINE_SCORES[min(dc, half)]
    grid /= grid.mean()
    return grid


_SPATIAL_PRIOR: np.ndarray = _make_spatial_prior(BOARD_SIZE)

# Opening priorities: tengen (天元) first, then the four 4-4 star points (星位).
# 4-4 (0-indexed: 3,3) radiates more influence than 3-3 and is the standard
# high-Chinese opening on 9x9.  3-4 (小目) points (2,3)/(3,2) also rank well
# via the spatial prior (4th-line score > 3rd-line score).
_TENGEN: Tuple[int, int] = (4, 4)
_OPENING_CORNERS: List[Tuple[int, int]] = [(3, 3), (5, 5), (3, 5), (5, 3)]
_OPENING_PRIORITIES: List[Tuple[int, int]] = [_TENGEN] + _OPENING_CORNERS

# ---------------------------------------------------------------------------
# Valid in-bounds neighbor count per cell — precomputed once for eye detection
# ---------------------------------------------------------------------------


def _make_valid_nbr_count(size: int) -> np.ndarray:
    cnt = np.full((size, size), 4, dtype=np.float32)
    cnt[0] -= 1;  cnt[-1] -= 1
    cnt[:, 0] -= 1;  cnt[:, -1] -= 1
    return cnt


_VALID_NBR_COUNT: np.ndarray = _make_valid_nbr_count(BOARD_SIZE)

# ---------------------------------------------------------------------------
# Board encoding  (10 feature planes)
# ---------------------------------------------------------------------------

NUM_PLANES = 10


def _count_neighbors(arr: np.ndarray) -> np.ndarray:
    """Sum of arr values at the 4 orthogonal in-bounds neighbors of each cell."""
    out = np.zeros_like(arr, dtype=np.float32)
    out[1:]    += arr[:-1]
    out[:-1]   += arr[1:]
    out[:, 1:] += arr[:, :-1]
    out[:, :-1] += arr[:, 1:]
    return out


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _is_own_eye(
    board: List[List[int]], r: int, c: int, color: int
) -> bool:
    """Return True if (r,c) is a potential eye for `color`.

    Condition: cell is empty (checked by caller) AND every in-bounds
    orthogonal neighbour is occupied by `color`.  Board edges count as
    neutral (no neighbour), so a corner only needs 2 friendly neighbours.
    """
    for dr, dc in _NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            if board[nr][nc] != color:
                return False
    return True


def _is_real_eye(board: List[List[int]], r: int, c: int, color: int) -> bool:
    """True if (r,c) is a real (true) eye (真眼) for `color`.

    Extends _is_own_eye with the diagonal condition:
      · Interior cell (all 4 orthogonal in-bounds): at most 1 bad diagonal allowed.
      · Edge/corner cell: 0 bad diagonals allowed.
    This separates false eyes (虛眼) from real eyes (真眼) that actually
    contribute to group life.
    """
    if not _is_own_eye(board, r, c, color):
        return False
    opp = WHITE if color == BLACK else BLACK
    interior = 0 < r < BOARD_SIZE - 1 and 0 < c < BOARD_SIZE - 1
    bad = sum(
        1 for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))
        if (0 <= r + dr < BOARD_SIZE and 0 <= c + dc < BOARD_SIZE
            and board[r + dr][c + dc] == opp)
    )
    return bad <= (1 if interior else 0)


def _count_group_real_eyes(
    board: List[List[int]], liberties: set, color: int
) -> int:
    """Count enclosed eye spaces (真眼空間) for a group.

    Step 1 — find "interior" liberties: cells where EVERY in-bounds orthogonal
      neighbor is either an own stone or another liberty of this group.
      Exterior liberties (adjacent to open empty cells) are excluded.

    Step 2 — find connected components of interior liberties.  Each component
      is one enclosed eye space:
        · Single-cell component: applies the diagonal real-eye condition.
        · Multi-cell component: always counts as 1 eye (大眼).

    This correctly handles single-cell eyes, large enclosed areas, and avoids
    false positives from exterior liberties that open to the rest of the board.
    """
    opp = WHITE if color == BLACK else BLACK

    interior: set = set()
    for r, c in liberties:
        enclosed = True
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue  # board edge is a wall — OK
            nb = board[nr][nc]
            if nb == EMPTY and (nr, nc) not in liberties:
                enclosed = False  # open empty cell outside group → not enclosed
                break
            if nb == opp:
                enclosed = False  # opponent stone borders this cell → not an eye
                break
        if enclosed:
            interior.add((r, c))

    eye_count = 0
    visited: set = set()
    for start in interior:
        if start in visited:
            continue
        region: set = set()
        stack = [start]
        while stack:
            cell = stack.pop()
            if cell in region:
                continue
            region.add(cell)
            r, c = cell
            for dr, dc in _NEIGHBORS:
                nb_cell = (r + dr, c + dc)
                if nb_cell in interior and nb_cell not in region:
                    stack.append(nb_cell)
        visited |= region
        if len(region) == 1:
            r, c = next(iter(region))
            if _is_real_eye(board, r, c, color):
                eye_count += 1
        else:
            eye_count += 1  # multi-cell enclosed region = 1 eye space
    return eye_count


def _get_life_urgency_moves(
    board: List[List[int]], color: int
) -> Dict[Tuple[int, int], float]:
    """Return {move: multiplier} boosting moves that help own groups with < 2 real eyes.

    For each own group with fewer than 2 real eyes (真眼), boost:
      · its liberties (direct expansion / natural eye-forming spots)
      · empty cells adjacent to those liberties (wider eye-space growth)
    Groups with 0 eyes get a stronger multiplier (2.0) than groups with 1 eye (1.5).
    """
    result: Dict[Tuple[int, int], float] = {}
    for stone, _group, liberties in _iter_groups(board):
        if stone != color:
            continue
        n_eyes = _count_group_real_eyes(board, liberties, color)
        if n_eyes >= 2:
            continue
        mult = 2.0 if n_eyes == 0 else 1.5
        for lib in liberties:
            if result.get(lib, 0.0) < mult:
                result[lib] = mult
        for r, c in liberties:
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                        and board[nr][nc] == EMPTY and (nr, nc) not in liberties):
                    edge_mult = mult * 0.65
                    if result.get((nr, nc), 0.0) < edge_mult:
                        result[(nr, nc)] = edge_mult
    return result


def _get_eye_attack_moves(
    board: List[List[int]], color: int
) -> Dict[Tuple[int, int], float]:
    """Return {move: multiplier} for moves that attack opponent groups with < 2 real eyes.

    This is the offensive mirror of _get_life_urgency_moves: for each opponent
    group that hasn't yet secured two real eyes, we identify the moves that
    invade or constrain their eye-forming space, potentially making them dead.

    Priority tiers:
      · Direct liberty invasion (佔眼位): opponent's liberties that are also
        candidate eye spaces.  Highest multiplier — playing here directly
        reduces the eye space available to the opponent.
      · Adjacent pressure (縮眼空): empty cells next to those liberties.
        Secondary multiplier — narrows the region, making two-eye life harder.

    Opponent groups with 0 eyes are the most urgent targets (mult=2.0);
    groups with 1 eye are still valuable to attack before they stabilise (mult=1.5).
    """
    opp = WHITE if color == BLACK else BLACK
    result: Dict[Tuple[int, int], float] = {}

    for stone, _group, liberties in _iter_groups(board):
        if stone != opp:
            continue
        n_eyes = _count_group_real_eyes(board, liberties, opp)
        if n_eyes >= 2:
            continue  # already alive — not worth attacking here

        mult = 2.0 if n_eyes == 0 else 1.5

        # Direct invasion: occupy the opponent's potential eye spaces.
        for lib in liberties:
            if result.get(lib, 0.0) < mult:
                result[lib] = mult

        # Adjacent pressure: shrink the eye-forming region from the outside.
        for r, c in liberties:
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                        and board[nr][nc] == EMPTY and (nr, nc) not in liberties):
                    edge_mult = mult * 0.65
                    if result.get((nr, nc), 0.0) < edge_mult:
                        result[(nr, nc)] = edge_mult

    return result


def _iter_groups(board: List[List[int]]):
    """Yield (color, group_cells, liberty_cells) for each connected stone group.

    Single shared BFS used by every routine that needs per-group data
    (capture detection, atari-escape detection, etc.) so we don't repeat
    flood-fill code in multiple places.
    """
    visited: set[Tuple[int, int]] = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            stone = board[r][c]
            if stone == EMPTY or (r, c) in visited:
                continue
            group: set[Tuple[int, int]] = set()
            liberties: set[Tuple[int, int]] = set()
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                if (rr, cc) in group:
                    continue
                group.add((rr, cc))
                for dr, dc in _NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr][nc] == stone and (nr, nc) not in group:
                            stack.append((nr, nc))
                        elif board[nr][nc] == EMPTY:
                            liberties.add((nr, nc))
            visited |= group
            yield stone, group, liberties


def _find_dead_zone_cells(
    board: List[List[int]], color: int, max_size: int = 4
) -> set[Tuple[int, int]]:
    """Return empty cells in regions enclosed by opponent that are too small to form two eyes.

    A region qualifies when every bordering stone is an opponent stone
    (no friendly stones touch it, so we cannot connect out) AND the
    region has at most `max_size` empty cells (not enough space for two
    eyes).  Note: this iterates EMPTY regions, unlike `_iter_groups`
    which iterates STONE groups, so it cannot share that helper.
    """
    opp = WHITE if color == BLACK else BLACK
    visited: set[Tuple[int, int]] = set()
    dead: set[Tuple[int, int]] = set()

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY or (r, c) in visited:
                continue
            region: set[Tuple[int, int]] = set()
            border_colors: set[int] = set()
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                if (rr, cc) in region:
                    continue
                region.add((rr, cc))
                for dr, dc in _NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr][nc] == EMPTY and (nr, nc) not in region:
                            stack.append((nr, nc))
                        elif board[nr][nc] != EMPTY:
                            border_colors.add(board[nr][nc])
            visited |= region
            if border_colors == {opp} and len(region) <= max_size:
                dead |= region

    return dead


def _is_wasteful_connection(
    board: List[List[int]], r: int, c: int, color: int
) -> bool:
    """Return True if placing `color` at (r,c) merely connects already-strong own groups.

    A connection is wasteful only when ALL adjacent own groups have > 3 liberties
    (already strong, no need for reinforcement) AND no opponent pressure nearby.
    Connecting two weak groups is legitimate — they share liberties and become
    harder to kill (連接只能為了讓兩個有弱點的棋連起來變強).
    """
    opp = WHITE if color == BLACK else BLACK
    own_adj_starts: List[Tuple[int, int]] = []
    opp_adj = 0
    for dr, dc in _NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            if board[nr][nc] == color:
                own_adj_starts.append((nr, nc))
            elif board[nr][nc] == opp:
                opp_adj += 1
    if len(own_adj_starts) < 2 or opp_adj > 0:
        return False
    # Check each adjacent own group's liberty count — exempt if any group is weak (≤ 3 libs)
    visited: set = set()
    for start in own_adj_starts:
        if start in visited:
            continue
        group: set = set()
        libs: set = set()
        stack = [start]
        while stack:
            rr, cc = stack.pop()
            if (rr, cc) in group:
                continue
            group.add((rr, cc))
            for ddr, ddc in _NEIGHBORS:
                nr, nc = rr + ddr, cc + ddc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    nb = board[nr][nc]
                    if nb == color and (nr, nc) not in group:
                        stack.append((nr, nc))
                    elif nb == EMPTY:
                        libs.add((nr, nc))
        visited |= group
        if len(libs) <= 3:
            return False  # weak group — connecting is justified
    return True  # all adjacent groups are strong — connection is wasteful


_STAR_POSITIONS: frozenset[Tuple[int, int]] = frozenset(
    [(4, 4), (3, 3), (5, 5), (3, 5), (5, 3)]
)


def _get_star_defense_moves(
    engine: GoEngine, color: int
) -> List[Tuple[int, int]]:
    """Return defensive moves for own star-position stones under threat.

    A star-position stone is "threatened" when our group containing it has
    ≤ 3 liberties AND at least one opponent stone is orthogonally adjacent
    to any stone in that group.  We return the group's liberties as candidate
    defense moves, sorted by how many liberties playing there would add
    (most-liberating first).
    """
    board = engine.board
    opp = WHITE if color == BLACK else BLACK
    candidates: List[Tuple[int, int]] = []
    seen_groups: set[Tuple[int, int]] = set()

    for star_r, star_c in _STAR_POSITIONS:
        if board[star_r][star_c] != color:
            continue
        if (star_r, star_c) in seen_groups:
            continue

        group: set[Tuple[int, int]] = set()
        liberties: set[Tuple[int, int]] = set()
        stack = [(star_r, star_c)]
        while stack:
            rr, cc = stack.pop()
            if (rr, cc) in group:
                continue
            group.add((rr, cc))
            for dr, dc in _NEIGHBORS:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr][nc] == color and (nr, nc) not in group:
                        stack.append((nr, nc))
                    elif board[nr][nc] == EMPTY:
                        liberties.add((nr, nc))
        seen_groups |= group

        if len(liberties) > 3:
            continue  # group still comfortable — not threatened

        opp_adjacent = any(
            board[rr + dr][cc + dc] == opp
            for rr, cc in group
            for dr, dc in _NEIGHBORS
            if 0 <= rr + dr < BOARD_SIZE and 0 <= cc + dc < BOARD_SIZE
        )
        if not opp_adjacent:
            continue  # no direct opponent pressure yet

        legal_libs = [m for m in liberties if engine.is_legal(*m)]
        if not legal_libs:
            continue

        # Sort: prefer liberty moves that give the group more breathing room.
        def _lib_gain(move: Tuple[int, int]) -> int:
            sim = engine.clone()
            if not sim.play(*move):
                return 0
            _, new_libs = sim._get_group(*move)
            return len(new_libs)

        legal_libs.sort(key=_lib_gain, reverse=True)
        candidates.extend(legal_libs)

    return candidates


def _escape_is_futile(engine: GoEngine, move: Tuple[int, int]) -> bool:
    """One-step lookahead: return True if escaping to `move` still leaves the group
    in atari and the opponent can immediately recapture.

    If this returns True the AI should not waste a move on this escape and should
    instead play tenuki or look for a better response.
    """
    sim = engine.clone()
    if not sim.play(*move):
        return True  # move turned out illegal in simulation (safety fallback)
    if sim.game_over:
        return False
    _, liberties = sim._get_group(move[0], move[1])
    if len(liberties) != 1:
        return False  # gained ≥2 liberties — escape has real value
    lib = next(iter(liberties))
    return sim.is_legal(*lib)  # opponent can legally snap back → futile


def _atari_threat_is_safe(engine: GoEngine, move: Tuple[int, int]) -> bool:
    """Return True if playing an atari threat at `move` doesn't endanger own groups.

    Uses a full one-step simulation (so captures are handled correctly) and
    checks two conditions:

      1. The placed stone's group has ≥ 2 liberties after the move.
         (Catches self-atari even when a capture opens temporary liberties.)

      2. No own group that was previously safe (≥ 2 liberties) drops to
         ≤ 1 liberty — i.e., the move doesn't indirectly weaken another
         own group that the opponent could then capture immediately.
    """
    color = engine.current_player
    r, c = move

    # Record which own cells belonged to safe groups before the move.
    safe_cells_before: set[Tuple[int, int]] = {
        cell
        for stone, group, libs in _iter_groups(engine.board)
        if stone == color and len(libs) >= 2
        for cell in group
    }

    # Simulate (captures resolved by engine).
    sim = engine.clone()
    if not sim.play(r, c):
        return False

    # Condition 1: placed stone's group must have ≥ 2 liberties.
    if sim.board[r][c] == color:            # stone wasn't captured itself
        _, placed_libs = sim._get_group(r, c)
        if len(placed_libs) <= 1:
            return False

    # Condition 2: no previously-safe group may now be in atari.
    for stone, group, libs in _iter_groups(sim.board):
        if stone != color or len(libs) > 1:
            continue
        if any(cell in safe_cells_before for cell in group):
            return False

    return True


def _get_tactical_moves(
    board: List[List[int]], color: int
) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:
    """Return (capture_moves, atari_escape_moves) for `color`.

    capture_moves: empty cells that immediately capture an opponent group in atari.
    atari_escape_moves: empty cells that are the single liberty of an own group in atari.
    """
    opp = WHITE if color == BLACK else BLACK
    capture_moves: set[Tuple[int, int]] = set()
    atari_escape_moves: set[Tuple[int, int]] = set()

    for stone, _group, liberties in _iter_groups(board):
        if len(liberties) != 1:
            continue
        lib = next(iter(liberties))
        if stone == opp:
            capture_moves.add(lib)
        elif stone == color:
            atari_escape_moves.add(lib)

    return capture_moves, atari_escape_moves


def _get_double_atari_moves(
    board: List[List[int]], color: int
) -> set[Tuple[int, int]]:
    """Return empty cells that simultaneously put ≥2 opponent groups in atari.

    A cell qualifies when it is a shared liberty of ≥2 distinct opponent
    groups that currently have exactly 2 liberties.  Placing there reduces
    both groups to 1 liberty in a single move.
    """
    opp = WHITE if color == BLACK else BLACK
    cell_count: Dict[Tuple[int, int], int] = {}
    for stone, _group, liberties in _iter_groups(board):
        if stone != opp or len(liberties) != 2:
            continue
        for lib in liberties:
            cell_count[lib] = cell_count.get(lib, 0) + 1
    return {cell for cell, cnt in cell_count.items() if cnt >= 2}


def _creates_empty_triangle(
    board: List[List[int]], r: int, c: int, color: int
) -> bool:
    """Return True if placing color at (r,c) would form an empty triangle.

    An empty triangle: three same-color stones in an L-shape where the inner
    corner cell is empty.  Bad shape — fewer effective liberties than a line.
    Detection: (r,c) plus one vertical and one horizontal same-color neighbor
    form the L; if the diagonal cell completing the rectangle is empty, it's
    an empty triangle.
    """
    for dr in (-1, 1):
        nr = r + dr
        if not (0 <= nr < BOARD_SIZE) or board[nr][c] != color:
            continue
        for dc in (-1, 1):
            nc = c + dc
            if not (0 <= nc < BOARD_SIZE) or board[r][nc] != color:
                continue
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == EMPTY:
                return True
    return False


def _get_capture_move_sizes(
    board: List[List[int]], color: int
) -> Dict[Tuple[int, int], int]:
    """Return {liberty: total_capturable_stones} for each opponent group in atari.

    When one liberty is shared by multiple groups, their sizes are summed so
    that a move capturing several groups at once gets the full credit.
    """
    opp = WHITE if color == BLACK else BLACK
    sizes: Dict[Tuple[int, int], int] = {}
    for stone, group, liberties in _iter_groups(board):
        if stone != opp or len(liberties) != 1:
            continue
        lib = next(iter(liberties))
        sizes[lib] = sizes.get(lib, 0) + len(group)
    return sizes


def _get_atari_threat_moves(
    board: List[List[int]], color: int
) -> Dict[Tuple[int, int], int]:
    """Return {move: group_size} for moves that put an opponent group in atari (叫吃).

    Only groups with exactly 2 liberties qualify — playing at either liberty
    reduces the group to 1 liberty (atari).  When a liberty is shared by
    multiple 2-liberty groups the sizes are summed, rewarding moves that
    simultaneously threaten several groups.
    """
    opp = WHITE if color == BLACK else BLACK
    threats: Dict[Tuple[int, int], int] = {}
    for stone, group, liberties in _iter_groups(board):
        if stone != opp or len(liberties) != 2:
            continue
        for lib in liberties:
            threats[lib] = threats.get(lib, 0) + len(group)
    return threats


def _nakade_vital_points(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells that are vital nakade points inside opponent-enclosed regions.

    A region qualifies when it is enclosed entirely by `color` stones and has
    3–5 empty cells.  The vital point is any cell whose removal splits the
    remaining space into components all of size ≤ 1 — no room left for two
    eyes.  Playing there kills the opponent's eyeless group.
    """
    vital: set = set()
    visited: set = set()

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY or (r, c) in visited:
                continue
            region: set = set()
            border_colors: set = set()
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                if (rr, cc) in region:
                    continue
                region.add((rr, cc))
                for dr, dc in _NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr][nc] == EMPTY and (nr, nc) not in region:
                            stack.append((nr, nc))
                        elif board[nr][nc] != EMPTY:
                            border_colors.add(board[nr][nc])
            visited |= region
            if border_colors != {color} or not (3 <= len(region) <= 5):
                continue
            for cell in region:
                remaining = region - {cell}
                seen: set = set()
                max_comp = 0
                for start in remaining:
                    if start in seen:
                        continue
                    comp: set = set()
                    q = [start]
                    while q:
                        cur = q.pop()
                        if cur in comp:
                            continue
                        comp.add(cur)
                        for dr, dc in _NEIGHBORS:
                            nb = (cur[0] + dr, cur[1] + dc)
                            if nb in remaining and nb not in comp:
                                q.append(nb)
                    seen |= comp
                    max_comp = max(max_comp, len(comp))
                if max_comp <= 1:
                    vital.add(cell)
    return vital


def _is_losing_ladder(
    engine: GoEngine, escape_move: Tuple[int, int], depth: int = 14
) -> bool:
    """Return True if escape_move leads into a losing ladder for engine.current_player.

    Simulates the forced continuation: attacker reduces the chased group to
    ≤ 1 liberty each step; group keeps running until it escapes (≥ 3 libs →
    False) or is captured (0 libs → True).  Depth-limited to avoid slowdown.
    """
    color = engine.current_player
    opp   = WHITE if color == BLACK else BLACK

    sim = engine.clone()
    if not sim.play(*escape_move):
        return True

    anchor = escape_move

    for _ in range(depth):
        if sim.game_over:
            return False
        if sim.board[anchor[0]][anchor[1]] != color:
            return True  # group was captured

        _, libs = sim._get_group(*anchor)
        n = len(libs)
        if n == 0:
            return True
        if n >= 3:
            return False  # escaped the chase

        if sim.current_player == opp:
            # Attacker tries the liberty that reduces our group to ≤ 1 lib.
            continued = False
            for lib in list(libs):
                if not sim.is_legal(*lib):
                    continue
                sim2 = sim.clone()
                sim2.play(*lib)
                if sim2.board[anchor[0]][anchor[1]] != color:
                    return True  # directly captured
                _, l2 = sim2._get_group(*anchor)
                if len(l2) <= 1:
                    sim = sim2
                    continued = True
                    break
            if not continued:
                return False  # attacker cannot maintain the chase
        else:
            if n == 1:
                only = next(iter(libs))
                if not sim.play(*only):
                    return True
                anchor = only
            else:
                return False  # 2 libs on our own turn → not a forced ladder

    return False


def _eye_score(
    board: List[List[int]], r: int, c: int, color: int
) -> Tuple[int, int]:
    """Return (own_eyes_created, opponent_eyes_denied) for placing `color` at (r,c).

    own_eyes_created:   empty neighbours of (r,c) that would become own eyes
                        once the stone is placed (all their other in-bounds
                        orthogonal neighbours already belong to `color`).

    opponent_eyes_denied: empty neighbours of (r,c) where (r,c) is the sole
                          missing stone that would complete an opponent eye —
                          by occupying it we permanently block that eye.
    """
    opp = WHITE if color == BLACK else BLACK
    own_created = 0
    opp_denied  = 0

    for dr, dc in _NEIGHBORS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
            continue
        if board[nr][nc] != EMPTY:
            continue

        # (nr, nc) is an empty orthogonal neighbour of (r, c).
        own_ok    = True   # would (nr,nc) become own eye after we place at (r,c)?
        opp_close = True   # is (r,c) the only non-opp stone blocking opp eye at (nr,nc)?

        for dr2, dc2 in _NEIGHBORS:
            nr2, nc2 = nr + dr2, nc + dc2
            if not (0 <= nr2 < BOARD_SIZE and 0 <= nc2 < BOARD_SIZE):
                continue  # board edge counts as "friendly" for both checks
            if (nr2, nc2) == (r, c):
                # The cell we're about to place — counts as own for own_ok,
                # and as the contested cell for opp_close (skip it).
                continue
            if board[nr2][nc2] != color:
                own_ok = False
            if board[nr2][nc2] != opp:
                opp_close = False

        if own_ok:
            own_created += 1
        if opp_close:
            opp_denied += 1

    return own_created, opp_denied


def _get_purposeless_moves(
    board: List[List[int]], color: int, legal: List[Tuple[int, int]],
    radius: int = 3,
) -> set[Tuple[int, int]]:
    """Return moves that accomplish nothing locally (無意義手).

    A move is purposeless when ALL of the following hold:
      1. No opponent stone within Manhattan distance `radius`.
      2. No own group with ≤ 3 liberties within `radius`.
      3. Placing there creates no eye for us (own_eye_score == 0).
      4. No adjacent empty cell sits on the boundary between own and opponent
         influence (so the move doesn't contest any future territory).

    Such moves don't attack, don't defend, don't build eyes, and don't claim
    territory — they waste the initiative entirely.
    """
    opp = WHITE if color == BLACK else BLACK

    # Precompute which cells belong to own weak groups (≤ 3 liberties).
    weak_own: set[Tuple[int, int]] = set()
    for stone, group, liberties in _iter_groups(board):
        if stone == color and len(liberties) <= 3:
            weak_own |= group

    purposeless: set[Tuple[int, int]] = set()
    for r, c in legal:
        # --- Conditions 1 & 2: neighbourhood scan ---
        has_opp = False
        has_weak_own = False
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) > radius:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if board[nr][nc] == opp:
                    has_opp = True
                if (nr, nc) in weak_own:
                    has_weak_own = True
            if has_opp and has_weak_own:
                break
        if has_opp or has_weak_own:
            continue

        # --- Condition 3: eye contribution ---
        own_eye, _ = _eye_score(board, r, c, color)
        if own_eye > 0:
            continue

        # --- Condition 4: territory contest ---
        # An adjacent empty cell that borders BOTH a own stone and an opponent
        # stone is a contested frontier — playing nearby has territorial meaning.
        on_frontier = False
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue
            if board[nr][nc] != EMPTY:
                continue
            adj_has_own = adj_has_opp = False
            for ddr, ddc in _NEIGHBORS:
                nnr, nnc = nr + ddr, nc + ddc
                if not (0 <= nnr < BOARD_SIZE and 0 <= nnc < BOARD_SIZE):
                    continue
                if board[nnr][nnc] == color:
                    adj_has_own = True
                elif board[nnr][nnc] == opp:
                    adj_has_opp = True
            if adj_has_own and adj_has_opp:
                on_frontier = True
                break
        if on_frontier:
            continue

        purposeless.add((r, c))

    return purposeless


def encode_board(
    engine: GoEngine,
    legal: Optional[List[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """Return a (10, 9, 9) float32 tensor encoding the current position.

    Pass pre-computed legal moves to avoid a redundant get_legal_moves() call.
    """
    planes    = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    board_arr = np.array(engine.board, dtype=np.int8)
    cur       = engine.current_player
    opp_color = opponent(cur)

    # --- Planes 0, 1: stone positions ---
    planes[0] = (board_arr == cur).astype(np.float32)
    planes[1] = (board_arr == opp_color).astype(np.float32)

    # --- Plane 2: current-player flag ---
    if cur == BLACK:
        planes[2] = 1.0

    # --- Planes 3-6: liberty counts per group (single BFS pass) ---
    lib_count: Dict[Tuple[int, int], int] = {}
    visited: set = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board_arr[r, c] != EMPTY and (r, c) not in visited:
                group, liberties = engine._get_group(r, c)
                n_lib = len(liberties)
                for cell in group:
                    lib_count[cell] = n_lib
                visited |= group

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            stone = board_arr[r, c]
            if stone == EMPTY:
                continue
            n    = lib_count.get((r, c), 0)
            base = 3 if stone == cur else 5
            if n == 1:
                planes[base,     r, c] = 1.0
            elif n == 2:
                planes[base + 1, r, c] = 1.0

    # --- Plane 7: legal moves (reuse caller's list to skip redundant call) ---
    if legal is None:
        legal = engine.get_legal_moves()
    if legal:
        idx_r, idx_c = zip(*legal)
        planes[7, list(idx_r), list(idx_c)] = 1.0

    # --- Planes 8, 9: potential eyes (vectorized) ---
    empty_mask = (board_arr == EMPTY).astype(np.float32)
    planes[8]  = empty_mask * (_count_neighbors(planes[0]) == _VALID_NBR_COUNT)
    planes[9]  = empty_mask * (_count_neighbors(planes[1]) == _VALID_NBR_COUNT)

    return torch.from_numpy(planes)


# ---------------------------------------------------------------------------
# Neural network  (128 ch, 5 res-blocks)
# ---------------------------------------------------------------------------


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class GoNet(nn.Module):
    """
    Residual policy+value network.
    Policy head  -> log-softmax over 81 intersections.
    Value head   -> tanh in [-1,+1], from current player's perspective.
    """

    def __init__(self, num_channels: int = 128, num_res_blocks: int = 5) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            *[_ResBlock(num_channels) for _ in range(num_res_blocks)]
        )
        self.p_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.p_bn   = nn.BatchNorm2d(2)
        self.p_fc   = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        self.v_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.v_bn   = nn.BatchNorm2d(1)
        self.v_fc1  = nn.Linear(BOARD_SIZE * BOARD_SIZE, 128)
        self.v_fc2  = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(self.stem(x))
        p = F.relu(self.p_bn(self.p_conv(x))).view(x.size(0), -1)
        log_policy = F.log_softmax(self.p_fc(p), dim=1)
        v = F.relu(self.v_bn(self.v_conv(x))).view(x.size(0), -1)
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v))
        return log_policy, value

    def set_inference_mode(self) -> None:
        self.train(False)


# ---------------------------------------------------------------------------
# Additional tactical helpers
# ---------------------------------------------------------------------------


def _get_cut_moves(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells that are cutting points against opponent groups.

    Two patterns detected:
    (a) Adjacency cut: cell touches ≥ 2 distinct opponent groups — our stone
        there prevents those groups from connecting.
    (b) Diagonal crosscut: two opponent stones are diagonally adjacent with
        both bridging cells empty — either bridging cell separates them.
    """
    opp = WHITE if color == BLACK else BLACK
    group_id: Dict[Tuple[int, int], int] = {}
    visited: set = set()
    gid = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != opp or (r, c) in visited:
                continue
            group: set = set()
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                if (rr, cc) in group:
                    continue
                group.add((rr, cc))
                for dr, dc in _NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr][nc] == opp and (nr, nc) not in group:
                            stack.append((nr, nc))
            for cell in group:
                group_id[cell] = gid
            visited |= group
            gid += 1

    cuts: set = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                continue
            adj_groups: set = set()
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and (nr, nc) in group_id:
                    adj_groups.add(group_id[(nr, nc)])
            if len(adj_groups) >= 2:
                cuts.add((r, c))

    for r in range(BOARD_SIZE - 1):
        for c in range(BOARD_SIZE - 1):
            if board[r][c] == opp and board[r + 1][c + 1] == opp:
                if board[r][c + 1] == EMPTY:
                    cuts.add((r, c + 1))
                if board[r + 1][c] == EMPTY:
                    cuts.add((r + 1, c))
            if board[r][c + 1] == opp and board[r + 1][c] == opp:
                if board[r][c] == EMPTY:
                    cuts.add((r, c))
                if board[r + 1][c + 1] == EMPTY:
                    cuts.add((r + 1, c + 1))
    return cuts


def _count_tiger_mouths_created(
    board: List[List[int]], r: int, c: int, color: int
) -> int:
    """Count empty neighbor cells that would become a 虎口 (tiger's mouth) for color.

    A tiger's mouth is an empty cell with ≥ 3 in-bounds orthogonal neighbors of
    the same color — strong shape that the opponent cannot safely enter.
    """
    count = 0
    for dr, dc in _NEIGHBORS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE) or board[nr][nc] != EMPTY:
            continue
        color_nbrs = 0
        in_bounds = 0
        for dr2, dc2 in _NEIGHBORS:
            nr2, nc2 = nr + dr2, nc + dc2
            if not (0 <= nr2 < BOARD_SIZE and 0 <= nc2 < BOARD_SIZE):
                continue
            in_bounds += 1
            if (nr2, nc2) == (r, c) or board[nr2][nc2] == color:
                color_nbrs += 1
        if in_bounds >= 3 and color_nbrs >= 3:
            count += 1
    return count


def _get_weak_group_pressure_moves(
    board: List[List[int]], color: int, max_liberties: int = 3
) -> set:
    """Return the liberties of opponent groups with few liberties (2–max_liberties).

    These empty cells are adjacent to groups that are already under pressure.
    Playing there tightens the net one step before issuing a formal atari —
    making follow-up captures much easier.
    """
    opp = WHITE if color == BLACK else BLACK
    pressure: set = set()
    for stone, _group, liberties in _iter_groups(board):
        if stone == opp and 2 <= len(liberties) <= max_liberties:
            pressure |= liberties
    return pressure


def _get_semeai_moves(
    board: List[List[int]], color: int
) -> set:
    """In liberty races (攻殺/semeai), return the opponent's exclusive liberties to fill.

    When one of our groups is adjacent to an opponent group and both have ≤ 5
    liberties, the side that fills external liberties first wins.  Returns the
    opponent's exclusive liberties (those not shared with our group) so MCTS
    explores the correct 'outside attack' moves.
    """
    opp = WHITE if color == BLACK else BLACK

    all_groups: Dict[int, Tuple] = {}
    cell_to_gid: Dict[Tuple[int, int], int] = {}
    gid = 0
    for stone, cells, libs in _iter_groups(board):
        all_groups[gid] = (stone, cells, libs)
        for cell in cells:
            cell_to_gid[cell] = gid
        gid += 1

    result: set = set()
    for our_id, (stone_a, cells_a, libs_a) in all_groups.items():
        if stone_a != color or len(libs_a) > 5:
            continue
        adj_opp: set = set()
        for r, c in cells_a:
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                        and board[nr][nc] == opp and (nr, nc) in cell_to_gid):
                    adj_opp.add(cell_to_gid[(nr, nc)])
        for opp_id in adj_opp:
            _, _cells_b, libs_b = all_groups[opp_id]
            if len(libs_b) <= 5:
                result |= libs_b - libs_a  # opponent's exclusive liberties
    return result


def _get_hane_at_head_moves(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells at the 'head' of opponent stone lines (頭頂).

    When 2+ opponent stones are in a straight line, playing at either end
    applies direct pressure ('hane at the head') — a classic shape that
    restricts the opponent's development and often sets up atari.
    """
    opp = WHITE if color == BLACK else BLACK
    moves: set = set()

    for dr, dc in ((0, 1), (1, 0)):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                pr, pc = r - dr, c - dc
                if (0 <= pr < BOARD_SIZE and 0 <= pc < BOARD_SIZE
                        and board[pr][pc] == opp):
                    continue  # not the start of the run
                if board[r][c] != opp:
                    continue
                end_r, end_c = r, c
                while True:
                    nr, nc = end_r + dr, end_c + dc
                    if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                            and board[nr][nc] == opp):
                        end_r, end_c = nr, nc
                    else:
                        break
                run_len = max(abs(end_r - r), abs(end_c - c)) + 1
                if run_len < 2:
                    continue
                for hr, hc in (
                    (r - dr, c - dc),
                    (end_r + dr, end_c + dc),
                ):
                    if (0 <= hr < BOARD_SIZE and 0 <= hc < BOARD_SIZE
                            and board[hr][hc] == EMPTY):
                        moves.add((hr, hc))
    return moves


def _get_vulnerable_connection_moves(
    board: List[List[int]], color: int
) -> set:
    """Return our connection points (bamboo joints / bridges) threatened by opponent.

    An empty cell is a vulnerable connection when it is adjacent to ≥ 2 of our
    own stones (so playing there connects groups) AND ≥ 1 opponent stone
    (so the opponent threatens to cut through that cell).  Playing there
    preemptively protects the connection before a cut can happen.
    """
    opp = WHITE if color == BLACK else BLACK
    result: set = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                continue
            own_adj = opp_adj = 0
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if board[nr][nc] == color:
                    own_adj += 1
                elif board[nr][nc] == opp:
                    opp_adj += 1
            if own_adj >= 2 and opp_adj >= 1:
                result.add((r, c))
    return result


def _is_ko_active(
    board: List[List[int]], previous_board: Optional[List[List[int]]]
) -> bool:
    """Return True when there is an active ko fight.

    A ko exists when the current board differs from the previous state by exactly
    2 cells (one stone placed, one stone captured via a single-stone removal).
    """
    if previous_board is None:
        return False
    diff = sum(
        1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
        if board[r][c] != previous_board[r][c]
    )
    return diff == 2


def _get_dual_purpose_attack_moves(
    board: List[List[int]], color: int
) -> set:
    """Empty cells that both pressure a weak opponent group AND extend own position.

    Qualifies when the cell is a liberty of an opponent group with 2–3 liberties
    (under pressure but not yet in atari — that case is already handled by
    capture_bias / atari_escape_bias) AND is orthogonally adjacent to at least
    one own stone.  Playing there tightens the net while physically connecting
    to / extending our own formation — 攻守兼備.
    """
    opp = WHITE if color == BLACK else BLACK
    attack_cells: set = set()
    for stone, _group, libs in _iter_groups(board):
        if stone == opp and 2 <= len(libs) <= 3:
            attack_cells |= libs

    result: set = set()
    for r, c in attack_cells:
        if board[r][c] != EMPTY:
            continue
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                    and board[nr][nc] == color):
                result.add((r, c))
                break
    return result


def _creates_self_atari(
    board: List[List[int]], r: int, c: int, color: int
) -> bool:
    """Return True if placing color at (r,c) leaves the resulting group with exactly 1 liberty.

    Uses a flood-fill from (r,c) treating it as already placed.  Callers should
    exempt capture moves, since capturing an opponent group in atari is legal and
    desirable even if the new stone temporarily has 1 liberty.
    """
    liberties: set = set()
    visited: set = {(r, c)}
    stack = [(r, c)]
    while stack:
        rr, cc = stack.pop()
        for dr, dc in _NEIGHBORS:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue
            if (nr, nc) in visited:
                continue
            nb = board[nr][nc]
            if nb == EMPTY:
                liberties.add((nr, nc))
            elif nb == color:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return len(liberties) == 1


def _get_corner_consolidation_moves(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells in corner regions where `color` already has stones.

    Corner region = 4×4 quadrant at each board corner.  When we have a
    foothold there we should deepen it to secure two eyes before the opponent
    can invade.  Only corners that already contain at least one own stone are
    returned — we are consolidating, not invading from scratch.
    """
    CORNER_RADIUS = 4
    result: set = set()
    starts = [0, BOARD_SIZE - CORNER_RADIUS]
    for r0 in starts:
        for c0 in starts:
            rows = range(r0, r0 + CORNER_RADIUS)
            cols = range(c0, c0 + CORNER_RADIUS)
            if not any(board[r][c] == color for r in rows for c in cols):
                continue
            for r in rows:
                for c in cols:
                    if board[r][c] == EMPTY:
                        result.add((r, c))
    return result


def _get_jump_connection_moves(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells that are efficient jump (一間跳) bridges between own stones.

    A one-point jump target T qualifies when:
      · T is two orthogonal steps from an own stone S with the intermediate
        cell empty (classic 一間跳 shape), AND
      · T has at least one other own stone as an orthogonal or diagonal
        neighbour (making T a bridge between two separate stone clusters).

    Jump connections are harder for the opponent to cut than solid chains and
    more efficient than passing — prefer them when linking own positions.
    """
    result: set = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != color:
                continue
            for dr, dc in _NEIGHBORS:
                mr, mc = r + dr,      c + dc       # midpoint
                tr, tc = r + 2 * dr, c + 2 * dc   # jump target
                if not (0 <= mr < BOARD_SIZE and 0 <= mc < BOARD_SIZE):
                    continue
                if not (0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE):
                    continue
                if board[mr][mc] != EMPTY or board[tr][tc] != EMPTY:
                    continue
                # Check if target is near another own stone (orth or diag)
                bridged = False
                for dr2, dc2 in _NEIGHBORS:
                    nr, nc = tr + dr2, tc + dc2
                    if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                            and board[nr][nc] == color and (nr, nc) != (r, c)):
                        bridged = True
                        break
                if not bridged:
                    for dr2, dc2 in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                        nr, nc = tr + dr2, tc + dc2
                        if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                                and board[nr][nc] == color and (nr, nc) != (r, c)):
                            bridged = True
                            break
                if bridged:
                    result.add((tr, tc))
    return result


def _get_nobi_moves(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells that extend own stone chains by one step (長/nobi).

    A nobi qualifies when the cell is adjacent to exactly one own stone and
    no opponent stones — pure directional extension, not a connection.
    Moves adjacent to opponent stones are already covered by tactical biases.
    """
    opp = WHITE if color == BLACK else BLACK
    result: set = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                continue
            own_adj = opp_adj = 0
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr][nc] == color:
                        own_adj += 1
                    elif board[nr][nc] == opp:
                        opp_adj += 1
            if own_adj == 1 and opp_adj == 0:
                result.add((r, c))
    return result


def _get_boundary_moves(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells on the border between own and opponent territory.

    Each boundary cell is adjacent to both our stones and opponent stones.
    Playing first gains roughly 1 point; these are the key endgame moves.
    """
    opp = WHITE if color == BLACK else BLACK
    boundary: set = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                continue
            has_own = has_opp = False
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if board[nr][nc] == color:
                    has_own = True
                elif board[nr][nc] == opp:
                    has_opp = True
            if has_own and has_opp:
                boundary.add((r, c))
    return boundary


def _find_snapback_escapes(
    board: List[List[int]], color: int
) -> set:
    """Return our escape moves we should skip because not escaping creates a snapback.

    If our group is in atari at liberty L and letting the opponent capture us would
    leave their capturing group with exactly 1 liberty (which we can immediately
    recapture), AND their group is larger than ours, we gain net material by NOT
    escaping — the snapback recapture is more valuable than saving our own stones.
    """
    opp = WHITE if color == BLACK else BLACK
    skip: set = set()

    for stone, group, libs in _iter_groups(board):
        if stone != color or len(libs) != 1:
            continue
        lib = next(iter(libs))

        temp = [row[:] for row in board]
        for r, c in group:
            temp[r][c] = EMPTY
        temp[lib[0]][lib[1]] = opp

        opp_grp: set = set()
        opp_libs: set = set()
        stack = [lib]
        while stack:
            rr, cc = stack.pop()
            if (rr, cc) in opp_grp or temp[rr][cc] != opp:
                continue
            opp_grp.add((rr, cc))
            for dr, dc in _NEIGHBORS:
                nr, nc = rr + dr, cc + dc
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if temp[nr][nc] == opp and (nr, nc) not in opp_grp:
                    stack.append((nr, nc))
                elif temp[nr][nc] == EMPTY:
                    opp_libs.add((nr, nc))

        if len(opp_libs) == 1 and len(opp_grp) > len(group):
            skip.add(lib)
    return skip


def _get_last_move(
    board: List[List[int]],
    previous_board: Optional[List[List[int]]],
    opp_color: int,
) -> Optional[Tuple[int, int]]:
    """Return the cell where opp_color just played, or None."""
    if previous_board is None:
        return None
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == opp_color and previous_board[r][c] != opp_color:
                return (r, c)
    return None


def _get_pattern_response_moves(
    board: List[List[int]],
    color: int,
    last_opp_move: Optional[Tuple[int, int]],
) -> Dict[Tuple[int, int], float]:
    """Return {move: multiplier} for pattern-based responses to opponent's last move.

    Pattern A — Tsuke (contact): opponent attached to our stone.
      · Hane: perpendicular turning move adjacent to our stone (×1.8).
      · Nobi: extend our stone in the contact direction away from opponent (×1.5).

    Pattern B — Peep-block: opponent is adjacent to our stone and a second of our
      stones flanks the contact, threatening a diagonal cut.
      Boost the cell that keeps us connected (×1.7).
    """
    if last_opp_move is None:
        return {}

    lr, lc = last_opp_move
    responses: Dict[Tuple[int, int], float] = {}

    for dr, dc in _NEIGHBORS:
        our_r, our_c = lr + dr, lc + dc
        if not (0 <= our_r < BOARD_SIZE and 0 <= our_c < BOARD_SIZE):
            continue
        if board[our_r][our_c] != color:
            continue
        # --- Pattern A ---
        # Nobi: extend away from opponent in the same direction
        ext_r, ext_c = our_r + dr, our_c + dc
        if (0 <= ext_r < BOARD_SIZE and 0 <= ext_c < BOARD_SIZE
                and board[ext_r][ext_c] == EMPTY):
            responses[(ext_r, ext_c)] = responses.get((ext_r, ext_c), 1.0) * 1.5
        # Hane: perpendicular to the contact axis, adjacent to our stone
        for h_dr, h_dc in ((-dc, dr), (dc, -dr)):
            h_r, h_c = our_r + h_dr, our_c + h_dc
            if (0 <= h_r < BOARD_SIZE and 0 <= h_c < BOARD_SIZE
                    and board[h_r][h_c] == EMPTY):
                responses[(h_r, h_c)] = responses.get((h_r, h_c), 1.0) * 1.8

        # --- Pattern B ---
        for perp_dr, perp_dc in ((-dc, dr), (dc, -dr)):
            sec_r, sec_c = our_r + perp_dr, our_c + perp_dc
            if not (0 <= sec_r < BOARD_SIZE and 0 <= sec_c < BOARD_SIZE):
                continue
            if board[sec_r][sec_c] != color:
                continue
            blk_r, blk_c = lr + perp_dr, lc + perp_dc
            if (0 <= blk_r < BOARD_SIZE and 0 <= blk_c < BOARD_SIZE
                    and board[blk_r][blk_c] == EMPTY):
                responses[(blk_r, blk_c)] = responses.get((blk_r, blk_c), 1.0) * 1.7

    return responses


def _get_kosumi_moves(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells that are diagonal extensions from own stones (小飛/Kosumi).

    A kosumi qualifies when the target cell is diagonally adjacent to an own
    stone.  We return ALL diagonal extensions (including cuttable ones) because:
      · On 9x9 the board is small and thick shape is always valuable.
      · Cuttable kosumi is still a key territory-claiming and connecting tool.

    The caller (_create_children) will apply kosumi_bias as the multiplier.
    """
    diagonals = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    result: set = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                continue
            for dr, dc in diagonals:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if board[nr][nc] == color:
                    result.add((r, c))
                    break
    return result


def _get_keima_moves(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells a knight's move from own stones (小飛/Keima).

    Keima (1×2 knight's move) is faster than kosumi for enclosure and blocking
    escape routes.  More aggressive than unarmed diagonal steps but cuttable —
    used primarily for chasing stones and sealing territory on 9x9.
    """
    result: set = set()
    keima_offsets = ((-2, -1), (-2, 1), (-1, -2), (-1, 2),
                     (1, -2),  (1, 2),  (2, -1),  (2, 1))
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                continue
            for dr, dc in keima_offsets:
                nr, nc = r + dr, c + dc
                if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                        and board[nr][nc] == color):
                    result.add((r, c))
                    break
    return result


def _get_ikken_tobi_moves(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells reachable by a one-point jump (一間跳) from any own stone.

    A one-point jump: target T is two orthogonal steps from own stone S with
    the intermediate cell empty.  This covers BOTH:
      · Pure extension: T has no other own neighbour (expanding into new territory).
      · Bridge: T connects two separate own clusters (already handled by
        _get_jump_connection_moves but included here for the unified bias).

    一間跳 is the fastest safe extension — harder to cut than nobi and more
    efficient than staying adjacent.  It is the primary tool for racing across
    the board and building influence.
    """
    result: set = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != color:
                continue
            for dr, dc in _NEIGHBORS:
                mr, mc = r + dr,      c + dc       # intermediate cell
                tr, tc = r + 2 * dr, c + 2 * dc   # jump target
                if not (0 <= mr < BOARD_SIZE and 0 <= mc < BOARD_SIZE):
                    continue
                if not (0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE):
                    continue
                if board[mr][mc] == EMPTY and board[tr][tc] == EMPTY:
                    result.add((tr, tc))
    return result


def _get_tsuke_moves(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells adjacent to opponent stones for contact play (碰/靠/Tsuke).

    Attaching directly to an opponent stone forces an immediate response — the
    strongest way to invade, settle, or ignite a fight in the confined 9x9 space.
    Only pure contact cells (adjacent to opponent, not already adjacent to own
    stones) are returned; mixed adjacency is already covered by tactical biases.
    """
    opp = WHITE if color == BLACK else BLACK
    result: set = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                continue
            has_opp = has_own = False
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr][nc] == opp:
                        has_opp = True
                    elif board[nr][nc] == color:
                        has_own = True
            if has_opp and not has_own:
                result.add((r, c))
    return result


def _get_clamp_moves(
    board: List[List[int]], color: int
) -> set:
    """Return cells that clamp opponent groups against the board edge (夾/Clamp).

    On the compact 9x9 board the edge is always close.  A clamp uses the boundary
    and our stone together to strip away the opponent's escape routes and base —
    particularly lethal against edge groups (1st–2nd line stones).
    """
    opp = WHITE if color == BLACK else BLACK
    result: set = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                continue
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if board[nr][nc] != opp:
                    continue
                edge_dist = min(nr, BOARD_SIZE - 1 - nr, nc, BOARD_SIZE - 1 - nc)
                if edge_dist <= 1:  # opponent stone on 1st or 2nd line
                    result.add((r, c))
                    break
    return result


def _get_ogeima_moves(
    board: List[List[int]], color: int
) -> set:
    """Return empty cells at large-knight (大飛/ogeima) distance from own stones.

    Ogeima = 1×3 or 3×1 jump.  Wider and faster than keima for building
    long-range frameworks and sliding along the edge (邊上大飛).
    """
    result: set = set()
    ogeima_offsets = (
        (-1, -3), (-1, 3), (1, -3), (1, 3),
        (-3, -1), (-3, 1), (3, -1), (3, 1),
    )
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != color:
                continue
            for dr, dc in ogeima_offsets:
                tr, tc = r + dr, c + dc
                if (0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE
                        and board[tr][tc] == EMPTY):
                    result.add((tr, tc))
    return result


def _get_territory_bridge_moves(
    board: List[List[int]], color: int
) -> set:
    """Empty cells reachable via shape move from TWO+ distinct own clusters.

    A cell qualifies when it can be reached by kosumi / keima / ogeima /
    one-point-jump from stones in at least two different connected components.
    Playing there bridges the two groups — essential for connecting own
    territories (連接兩個自己的陣地) and for edge protection (保護邊).
    """
    # Label every own stone with its connected-component id.
    comp_id: Dict[Tuple[int, int], int] = {}
    visited: set = set()
    cid = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != color or (r, c) in visited:
                continue
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                if (rr, cc) in visited:
                    continue
                visited.add((rr, cc))
                comp_id[(rr, cc)] = cid
                for dr, dc in _NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                            and board[nr][nc] == color and (nr, nc) not in visited):
                        stack.append((nr, nc))
            cid += 1

    if cid < 2:
        return set()  # only one cluster — nothing to bridge

    # Offsets for all recognised shape moves.
    _SHAPE_OFFSETS = (
        (-1, -1), (-1, 1), (1, -1), (1, 1),              # kosumi (尖)
        (-2, -1), (-2, 1), (2, -1), (2, 1),
        (-1, -2), (-1, 2), (1, -2), (1, 2),              # keima (小飛)
        (-3, -1), (-3, 1), (3, -1), (3, 1),
        (-1, -3), (-1, 3), (1, -3), (1, 3),              # ogeima (大飛)
        (-2, 0), (2, 0), (0, -2), (0, 2),                # one-point jump (跳)
    )

    cell_comps: Dict[Tuple[int, int], set] = {}
    for (r, c), cid_here in comp_id.items():
        for dr, dc in _SHAPE_OFFSETS:
            tr, tc = r + dr, c + dc
            if (0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE
                    and board[tr][tc] == EMPTY):
                if (tr, tc) not in cell_comps:
                    cell_comps[(tr, tc)] = set()
                cell_comps[(tr, tc)].add(cid_here)

    return {cell for cell, comps in cell_comps.items() if len(comps) >= 2}


def _get_moyo_invasion_moves(
    board: List[List[int]], color: int, radius: int = 3
) -> set:
    """Empty open cells where opponent has more nearby influence than us.

    A cell qualifies when it has no adjacent stones (open space, not a contact
    play) AND there are at least 2 opponent stones within `radius` steps but
    more opponent stones than own stones in that window.  These are the natural
    invasion points inside the opponent's developing framework (侵入對方陣地).
    """
    opp = WHITE if color == BLACK else BLACK
    result: set = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                continue
            # Must be open — no immediate contact with any stone.
            if any(
                board[r + dr][c + dc] != EMPTY
                for dr, dc in _NEIGHBORS
                if 0 <= r + dr < BOARD_SIZE and 0 <= c + dc < BOARD_SIZE
            ):
                continue
            own_n = opp_n = 0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                        continue
                    if board[nr][nc] == color:
                        own_n += 1
                    elif board[nr][nc] == opp:
                        opp_n += 1
            if opp_n >= 2 and opp_n > own_n + 1:
                result.add((r, c))
    return result


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

# Virtual loss magnitude.  Applied as an assumed win for the traversed node's
# player: visit_count += VL, value_sum += VL → q_value → 1 → subsequent
# traversals in the same batch see the node as "claimed" and explore elsewhere.
_VL = 1

# Sentinel for the pass move inside the MCTS tree (never returned to callers;
# select_move converts it back to None).
_PASS: Tuple[int, int] = (-1, -1)


class MCTSNode:
    """
    MCTS tree node.  Q stored from the perspective of the player to move
    at this node; parent negates Q when computing PUCT.
    """

    __slots__ = ("prior", "visit_count", "value_sum", "children", "parent", "move")

    def __init__(
        self,
        prior: float,
        parent: Optional[MCTSNode] = None,
        move: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.prior       = prior
        self.visit_count = 0
        self.value_sum   = 0.0
        self.children: Dict[Tuple[int, int], MCTSNode] = {}
        self.parent      = parent
        self.move        = move

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def is_leaf(self) -> bool:
        return not self.children

    def puct_score(self, c_puct: float) -> float:
        # AlphaZero PUCT formula: combine an exploitation term (-Q from
        # the parent's perspective, since Q is stored from this node's
        # mover's perspective) with an exploration term U that fades as
        # the child gets visited:
        #     U = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        # High-prior unexplored moves get a large U; well-visited moves
        # rely mostly on their Q estimate.  c_puct trades the two off.
        parent_n = self.parent.visit_count if self.parent else 1
        u = c_puct * self.prior * math.sqrt(max(parent_n, 1)) / (1 + self.visit_count)
        return -self.q_value + u

    def best_child(self, c_puct: float) -> Tuple[Tuple[int, int], MCTSNode]:
        return max(self.children.items(),
                   key=lambda kv: kv[1].puct_score(c_puct))


class MCTS:
    """AlphaZero-style MCTS with batched leaf evaluation and virtual loss."""

    def __init__(
        self,
        model: GoNet,
        num_simulations: int = 400,
        c_puct: float = 1.5,
        device: str = "cpu",
        spatial_bias: float = 1.0,
        training: bool = True,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        eval_batch_size: int = 8,
        eye_bias: float = 4.0,
        anti_eye_bias: float = 1.5,
        pass_weight: float = 0.15,
        resign_threshold: Optional[float] = None,
        capture_bias: float = 5.0,
        atari_threat_bias: float = 4.0,
        atari_escape_bias: float = 4.0,
        connection_penalty: float = 0.35,
        double_atari_bias: float = 3.0,
        gameplay_temperature: float = 0.8,
        edge_penalty: float = 0.04,
        empty_triangle_penalty: float = 0.6,
        nakade_bias: float = 5.0,
        eye_attack_bias: float = 4.5,
        cut_bias: float = 2.5,
        tiger_mouth_bias: float = 1.5,
        ko_threat_multiplier: float = 1.5,
        boundary_bias: float = 2.0,
        weak_group_pressure_bias: float = 2.0,
        semeai_bias: float = 3.0,
        hane_bias: float = 2.0,
        connection_guard_bias: float = 2.5,
        pattern_response_bias: float = 1.0,
        self_atari_penalty: float = 0.10,
        purposeless_penalty: float = 0.05,
        dual_purpose_bias: float = 3.5,
        corner_consolidation_bias: float = 2.0,
        jump_connection_bias: float = 2.5,
        ikken_tobi_bias: float = 5.5,
        nobi_bias: float = 1.4,
        kosumi_bias: float = 6.0,
        keima_bias: float = 1.3,
        tsuke_bias: float = 1.5,
        clamp_bias: float = 2.0,
        edge_eye_bias: float = 3.5,
        ogeima_bias: float = 1.8,
        territory_bridge_bias: float = 3.5,
        invasion_bias: float = 2.0,
        life_urgency_bias: float = 5.5,
    ) -> None:
        self.model           = model.to(device)
        self.model.set_inference_mode()
        self.num_simulations = num_simulations
        self.c_puct          = c_puct
        self.device          = torch.device(device)
        self.spatial_bias    = spatial_bias
        self.training        = training
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps   = dirichlet_eps
        self.eval_batch_size = eval_batch_size
        # eye_bias: prior multiplier bonus per own eye a move creates
        # anti_eye_bias: prior multiplier bonus per opponent eye a move denies
        self.eye_bias         = eye_bias
        self.anti_eye_bias    = anti_eye_bias
        self.pass_weight      = pass_weight
        # resign when q_value drops below this during play (disabled in training)
        self.resign_threshold = resign_threshold
        # capture_bias: prior multiplier for moves that capture opponent in atari (提子)
        # atari_threat_bias: prior multiplier for moves that put opponent in atari (叫吃)
        # atari_escape_bias: prior multiplier for moves that escape own atari
        # connection_penalty: prior multiplier for moves that only connect safe own groups
        self.capture_bias         = capture_bias
        self.atari_threat_bias    = atari_threat_bias
        self.atari_escape_bias    = atari_escape_bias
        self.connection_penalty   = connection_penalty
        self.double_atari_bias    = double_atari_bias
        # temperature > 0: sample from visit-count distribution instead of argmax,
        # introducing game-to-game variety without sacrificing tactical accuracy
        self.gameplay_temperature = gameplay_temperature
        # edge_penalty: prior multiplier for first-line moves that have no
        # tactical justification (capture/escape/eye/double-atari exempt)
        self.edge_penalty             = edge_penalty
        # empty_triangle_penalty: prior multiplier for moves that form bad L-shape
        self.empty_triangle_penalty   = empty_triangle_penalty
        # nakade_bias: prior multiplier for vital points inside opponent eye-spaces
        self.nakade_bias              = nakade_bias
        self.eye_attack_bias          = eye_attack_bias
        # cut_bias: prior multiplier for moves that separate distinct opponent groups
        self.cut_bias                 = cut_bias
        # tiger_mouth_bias: prior multiplier per tiger's mouth shape created
        self.tiger_mouth_bias         = tiger_mouth_bias
        # ko_threat_multiplier: extra boost to capture/atari moves when a ko is active
        self.ko_threat_multiplier     = ko_threat_multiplier
        # boundary_bias: boost territory boundary moves in late game (≥50 stones on board)
        self.boundary_bias              = boundary_bias
        # weak_group_pressure_bias: boost liberties of opponent groups with 2-3 liberties
        self.weak_group_pressure_bias   = weak_group_pressure_bias
        # semeai_bias: boost opponent exclusive liberties in adjacent liberty races
        self.semeai_bias                = semeai_bias
        # hane_bias: boost cells at the head of opponent stone lines
        self.hane_bias                  = hane_bias
        # connection_guard_bias: boost our threatened connection points (bamboo joints)
        self.connection_guard_bias      = connection_guard_bias
        # pattern_response_bias: strength multiplier for pattern-based responses (tsuke/peep)
        self.pattern_response_bias      = pattern_response_bias
        self.self_atari_penalty         = self_atari_penalty
        self.purposeless_penalty        = purposeless_penalty
        self.dual_purpose_bias              = dual_purpose_bias
        self.corner_consolidation_bias      = corner_consolidation_bias
        self.jump_connection_bias           = jump_connection_bias
        self.ikken_tobi_bias                = ikken_tobi_bias
        self.nobi_bias                      = nobi_bias
        self.kosumi_bias                    = kosumi_bias
        self.keima_bias                     = keima_bias
        self.tsuke_bias                     = tsuke_bias
        self.clamp_bias                     = clamp_bias
        self.edge_eye_bias                  = edge_eye_bias
        self.ogeima_bias                    = ogeima_bias
        self.territory_bridge_bias          = territory_bridge_bias
        self.invasion_bias                  = invasion_bias
        self.life_urgency_bias              = life_urgency_bias
        self._eval_cache: Dict[bytes, Tuple[np.ndarray, float]] = {}
        # Shuffle the four corner star points randomly each game.
        self._opening_priorities: List[Tuple[int, int]] = (
            random.sample(_OPENING_CORNERS, len(_OPENING_CORNERS))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_game(self) -> None:
        """Reshuffle opening priorities and clear the eval cache for a new game."""
        self._opening_priorities = (
            random.sample(_OPENING_CORNERS, len(_OPENING_CORNERS))
        )
        self._eval_cache.clear()

    def check_resign(self, engine: GoEngine) -> bool:
        """Return True when a single NN eval says the position is hopeless.

        Fast pre-MCTS check — avoids wasting simulations when clearly losing.
        Never triggers during training (self.training=True).
        """
        if self.training or self.resign_threshold is None:
            return False
        with torch.inference_mode():
            t = encode_board(engine).unsqueeze(0).to(self.device)
            _, v = self.model(t)
        return float(v.item()) < self.resign_threshold

    def select_move(
        self, engine: GoEngine, allow_pass: bool = True
    ) -> Optional[Tuple[int, int]]:
        legal = engine.get_legal_moves()

        if not legal and self.pass_weight <= 0.0:
            return None  # no legal board moves and pass disabled → concede

        self._eval_cache.clear()
        root = MCTSNode(prior=1.0)

        sims_done = 0
        while sims_done < self.num_simulations:
            b = min(self.eval_batch_size, self.num_simulations - sims_done)
            self._run_batch(root, engine, b)
            sims_done += b

        if not root.children:
            if not legal:
                return None
            non_eye = self._filter_eyes(engine.board, engine.current_player, legal)
            if not non_eye:
                return None  # all remaining moves are own eyes → pass
            return random.choice(non_eye)

        # Move selection: temperature=0 → argmax (deterministic);
        # temperature>0 → sample proportionally to visit_count^(1/T) for variety.
        T = self.gameplay_temperature if not self.training else 0.0
        if T > 0.0:
            candidates = list(root.children.items())
            weights = [ch.visit_count ** (1.0 / T) for _, ch in candidates]
            total_w = sum(weights) or 1.0
            r_val = random.random() * total_w
            cumsum = 0.0
            best = candidates[0][0]
            for (move, _), w in zip(candidates, weights):
                cumsum += w
                if cumsum >= r_val:
                    best = move
                    break
        else:
            best = max(root.children, key=lambda m: root.children[m].visit_count)

        if not allow_pass and best == _PASS:
            # Human rejected our pass — must play a real board move.
            non_pass = {m: ch for m, ch in root.children.items() if m != _PASS}
            if non_pass:
                if T > 0.0:
                    cands = list(non_pass.items())
                    ws = [ch.visit_count ** (1.0 / T) for _, ch in cands]
                    tw = sum(ws) or 1.0
                    rv = random.random() * tw
                    cs = 0.0
                    best = cands[0][0]
                    for (mv, _), w in zip(cands, ws):
                        cs += w
                        if cs >= rv:
                            best = mv
                            break
                else:
                    best = max(non_pass, key=lambda m: non_pass[m].visit_count)
            else:
                return None  # truly no legal moves; accept the pass situation
        return None if best == _PASS else best

    def get_move_probabilities(
        self, engine: GoEngine, temperature: float = 1.0
    ) -> Dict[Tuple[int, int], float]:
        legal = engine.get_legal_moves()
        if not legal:
            return {}

        self._eval_cache.clear()
        root = MCTSNode(prior=1.0)

        sims_done = 0
        while sims_done < self.num_simulations:
            b = min(self.eval_batch_size, self.num_simulations - sims_done)
            self._run_batch(root, engine, b)
            sims_done += b

        visits = {m: ch.visit_count for m, ch in root.children.items()
                  if m != _PASS}
        if not visits:
            return {}

        total  = sum(visits.values()) or 1

        if temperature == 0:
            best = max(visits, key=visits.get)
            return {m: (1.0 if m == best else 0.0) for m in visits}

        raw     = {m: (v / total) ** (1.0 / temperature) for m, v in visits.items()}
        raw_sum = sum(raw.values()) or 1.0
        return {m: p / raw_sum for m, p in raw.items()}

    # ------------------------------------------------------------------
    # Core: batched simulation round
    # ------------------------------------------------------------------

    def _run_batch(self, root: MCTSNode, engine: GoEngine, b: int) -> None:
        """Run b MCTS simulations using virtual loss + one batched NN call.

        Virtual loss is the key trick that makes batching safe: each
        traversal in this batch tentatively pretends the path it took
        already won (visit_count += _VL, value_sum += _VL → q_value → 1).
        Because PUCT uses -q_value from the parent's perspective, the
        path now looks bad to the parent and the *next* traversal in the
        batch will pick a different child.  Phase 3 undoes the virtual
        loss before applying the real backed-up value.
        """

        # ---- Phase 1: Traverse b paths, applying virtual loss ----
        # path[i] = list of nodes from root to leaf (inclusive)
        paths: List[List[MCTSNode]] = []
        leaf_sims: List[GoEngine]   = []

        for _ in range(b):
            node = root
            sim  = engine.clone()
            path: List[MCTSNode] = []

            while not node.is_leaf():
                if sim.game_over:
                    break
                # Tentative "this path already won" — pushes other
                # traversals in this batch toward different children.
                node.visit_count += _VL
                node.value_sum   += _VL
                path.append(node)
                move, node = node.best_child(self.c_puct)
                if move == _PASS:
                    sim.pass_move()
                else:
                    sim.play(*move)

            node.visit_count += _VL
            node.value_sum   += _VL
            path.append(node)

            paths.append(path)
            leaf_sims.append(sim)

        # ---- Phase 2: Batch NN evaluation ----
        leaf_values: List[float] = [0.0] * b

        # Separate cache hits / misses / terminal positions
        to_eval: List[Tuple[int, MCTSNode, GoEngine, List, bytes]] = []

        for i in range(b):
            sim_i = leaf_sims[i]

            # Terminal: game ended (e.g. double pass) — value from winner vs current player
            if sim_i.game_over:
                leaf_values[i] = 1.0 if sim_i.winner == sim_i.current_player else -1.0
                continue

            leaf_legal = sim_i.get_legal_moves()
            if not leaf_legal:
                leaf_values[i] = 0.0   # no board moves — pass will be added as child
                continue

            # Filter moves for child creation.
            # encode_board still gets the full leaf_legal (plane 7 reflects rules).
            expand_legal = self._filter_eyes(
                sim_i.board, sim_i.current_player, leaf_legal
            )
            expand_legal = self._filter_dead_moves(
                sim_i.board, sim_i.current_player, expand_legal
            )

            board_i      = sim_i.board
            color_i      = sim_i.current_player
            cpasses_i    = sim_i.consecutive_passes
            ko_i         = _is_ko_active(board_i, sim_i.previous_board)
            last_move_i  = _get_last_move(board_i, sim_i.previous_board, opponent(color_i))
            key = self._cache_key(sim_i)
            if key in self._eval_cache:
                policy, value = self._eval_cache[key]
                self._create_children(
                    paths[i][-1], expand_legal, policy,
                    self.training and (paths[i][-1] is root),
                    board=board_i, color=color_i, value=value,
                    consecutive_passes=cpasses_i, ko_active=ko_i,
                    last_opp_move=last_move_i,
                )
                leaf_values[i] = value
            else:
                to_eval.append((i, paths[i][-1], sim_i, leaf_legal, expand_legal,
                                 board_i, color_i, cpasses_i, key, ko_i, last_move_i))

        if to_eval:
            # Stack boards into one batch tensor — one forward pass for all
            tensors = torch.stack([
                encode_board(sim, ll) for _, _, sim, ll, _, _, _, _, _, _, _ in to_eval
            ]).to(self.device)

            with torch.inference_mode():
                log_p, vt = self.model(tensors)
            policies   = log_p.exp().cpu().numpy()
            values_arr = vt.squeeze(1).cpu().numpy()

            for j, (i, leaf, sim, leaf_legal, expand_legal,
                    board_j, color_j, cpasses_j, key, ko_j, last_move_j) in enumerate(to_eval):
                policy = policies[j]
                value  = float(values_arr[j])
                self._eval_cache[key] = (policy, value)
                self._create_children(
                    leaf, expand_legal, policy,
                    self.training and (leaf is root),
                    board=board_j, color=color_j, value=value,
                    consecutive_passes=cpasses_j, ko_active=ko_j,
                    last_opp_move=last_move_j,
                )
                leaf_values[i] = value

        # ---- Phase 3: Undo virtual loss, then apply real backprop ----
        for i in range(b):
            value = leaf_values[i]
            for node in reversed(paths[i]):
                node.visit_count -= _VL   # undo virtual loss
                node.value_sum   -= _VL
                node.visit_count += 1
                node.value_sum   += value
                value             = -value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_eyes(
        self, board: List[List[int]], color: int, legal: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Remove moves that fill own true eyes.  Never falls back to including eye moves —
        if all remaining legal moves are own eyes, the caller should pass instead."""
        return [m for m in legal if not _is_own_eye(board, m[0], m[1], color)]

    def _filter_dead_moves(
        self, board: List[List[int]], color: int, legal: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Remove moves into opponent-enclosed regions too small to form two eyes.

        Regions of ≤ 4 empty cells bordered only by opponent stones cannot support
        a living group regardless of how we play.  Exception: a move that is also
        a capture point is kept — capturing opens liberties and may allow survival.
        """
        cap_moves, _ = _get_tactical_moves(board, color)
        dead_cells = _find_dead_zone_cells(board, color, max_size=4)
        filtered = [
            m for m in legal
            if m not in dead_cells or m in cap_moves
        ]
        return filtered if filtered else legal

    def _cache_key(self, engine: GoEngine) -> bytes:
        """Board hash including ko state so legal-move sets always match."""
        board_bytes = np.array(engine.board, dtype=np.int8).tobytes()
        prev_bytes  = (b'' if engine.previous_board is None
                       else np.array(engine.previous_board, dtype=np.int8).tobytes())
        return bytes([engine.current_player]) + board_bytes + prev_bytes

    def _create_children(
        self,
        node: MCTSNode,
        legal: List[Tuple[int, int]],
        policy: np.ndarray,
        add_noise: bool,
        board: Optional[List[List[int]]] = None,
        color: Optional[int] = None,
        value: float = 0.0,
        consecutive_passes: int = 0,
        ko_active: bool = False,
        last_opp_move: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Populate node.children from policy array.  No-op if already expanded."""
        if node.children:
            return

        raw = {(r, c): float(policy[r * BOARD_SIZE + c]) for r, c in legal}

        # --- Spatial bias ---
        # The 9x9 spatial heatmap embeds the Go proverb "金角銀邊草肚皮"
        # (corners are gold, sides silver, centre straw) along with a
        # 3rd/4th-line preference.  Multiplying the network policy by it
        # nudges the AI toward proven good shape without overriding the
        # NN's own opinions — moves the NN really likes still win.
        if self.spatial_bias > 0.0:
            for move in raw:
                r, c = move
                raw[move] *= max(0.01, 1.0 + self.spatial_bias * (_SPATIAL_PRIOR[r, c] - 1.0))

        # --- Eye bias / anti-eye bias ---
        # Eyes are what keeps groups alive, so we boost moves that create
        # own eyes or destroy opponent eye shape.  Applied multiplicatively
        # to the NN prior so the network policy still dominates — this is
        # a nudge, not an override.  The point is to give a freshly-trained
        # network sensible behavior on eye shape from move 1.
        if board is not None and color is not None:
            if self.eye_bias > 0.0 or self.anti_eye_bias > 0.0:
                for move in raw:
                    r, c = move
                    own, denied = _eye_score(board, r, c, color)
                    bonus = self.eye_bias * own + self.anti_eye_bias * denied
                    if bonus > 0.0:
                        raw[move] *= 1.0 + bonus

            # --- Edge eye bias ---
            # Board edges act as extra walls — a 1st/2nd-line stone needs fewer
            # neighbours to enclose an eye (邊做眼).  Give an additional multiplier
            # on top of eye_bias so the AI learns to use the boundary for life/death.
            if self.edge_eye_bias > 1.0:
                for move in raw:
                    if move == _PASS:
                        continue
                    r, c = move
                    if min(r, BOARD_SIZE - 1 - r, c, BOARD_SIZE - 1 - c) > 1:
                        continue  # not on 1st or 2nd line
                    own_e, _ = _eye_score(board, r, c, color)
                    if own_e > 0:
                        raw[move] *= self.edge_eye_bias

        # --- Gameplay-only prior shaping (all heuristics disabled during training) ---
        # Tactical sets are computed once here and reused by every heuristic below
        # to avoid redundant board scans.
        if not self.training and board is not None and color is not None:
            capture_sizes = _get_capture_move_sizes(board, color)  # {lib: stones_captured}
            cap_moves     = set(capture_sizes)
            _, esc_moves  = _get_tactical_moves(board, color)
            da_moves = (
                _get_double_atari_moves(board, color)
                if self.double_atari_bias > 1.0 or self.edge_penalty < 1.0
                else set()
            )

            # Capture bias (提子): scale by log(group_size) so capturing large groups
            # gets proportionally more exploration than capturing lone stones.
            if self.capture_bias > 1.0:
                for move, size in capture_sizes.items():
                    if move in raw:
                        raw[move] *= self.capture_bias * (1.0 + 0.5 * math.log1p(size - 1))

            # Atari-threat bias (叫吃): boost moves that reduce an opponent group
            # from 2 liberties to 1.  Scaled by log(group_size) so threatening a
            # large group earns stronger exploration than threatening a single stone.
            # Capture moves are excluded — they already receive capture_bias above.
            # Computed unconditionally so purposeless_penalty can use it as an exemption.
            atari_threats = _get_atari_threat_moves(board, color)
            if self.atari_threat_bias > 1.0:
                for move, size in atari_threats.items():
                    if move in raw and move not in cap_moves:
                        raw[move] *= self.atari_threat_bias * (1.0 + 0.3 * math.log1p(size - 1))

            # Atari-escape bias: amplify priors for moves that save own groups.
            if self.atari_escape_bias > 1.0:
                for move in raw:
                    if move in esc_moves:
                        raw[move] *= self.atari_escape_bias

            # Connection penalty: discount moves that only connect already-safe groups.
            if self.connection_penalty < 1.0:
                for move in raw:
                    if move != _PASS:
                        r, c = move
                        if _is_wasteful_connection(board, r, c, color):
                            raw[move] *= self.connection_penalty

            # Double atari bias: boost moves that threaten two opponent groups at once.
            if self.double_atari_bias > 1.0:
                for move in raw:
                    if move in da_moves:
                        raw[move] *= self.double_atari_bias

            # Cut bias: boost moves that separate two or more distinct opponent groups.
            if self.cut_bias > 1.0:
                cut_moves = _get_cut_moves(board, color)
                for move in raw:
                    if move in cut_moves:
                        raw[move] *= self.cut_bias

            # Tiger's mouth bias: boost moves that create 虎口 (3-sided enclosed) shapes.
            # Stack the multiplier for each tiger's mouth created (usually 0 or 1).
            if self.tiger_mouth_bias > 1.0:
                for move in raw:
                    if move == _PASS:
                        continue
                    r, c = move
                    mouths = _count_tiger_mouths_created(board, r, c, color)
                    if mouths > 0:
                        raw[move] *= self.tiger_mouth_bias ** mouths

            # Ko threat multiplier: when a ko fight is active, extra-boost captures
            # and atari-escape moves so MCTS finds good ko threats faster.
            if ko_active and self.ko_threat_multiplier > 1.0:
                for move in raw:
                    if move in cap_moves or move in esc_moves:
                        raw[move] *= self.ko_threat_multiplier

            # Boundary bias: in the late game (≥ 50 stones), boost moves on the
            # boundary between own and opponent territory — each is worth ~1 point.
            if self.boundary_bias > 1.0:
                stone_count = sum(
                    1 for rr in range(BOARD_SIZE) for cc in range(BOARD_SIZE)
                    if board[rr][cc] != EMPTY
                )
                if stone_count >= 50:
                    boundary = _get_boundary_moves(board, color)
                    for move in raw:
                        if move in boundary:
                            raw[move] *= self.boundary_bias

            # Empty triangle penalty: discourage moves that form bad L-shape.
            # Exempt captures and escapes which may require bad-shape moves.
            if self.empty_triangle_penalty < 1.0:
                for move in raw:
                    if move == _PASS or move in cap_moves or move in esc_moves:
                        continue
                    r, c = move
                    if _creates_empty_triangle(board, r, c, color):
                        raw[move] *= self.empty_triangle_penalty

            # Nakade bias: boost vital points inside opponent enclosed eyespaces
            # to encourage killing groups rather than letting them live with two eyes.
            if self.nakade_bias > 1.0:
                vital = _nakade_vital_points(board, color)
                for move in raw:
                    if move in vital:
                        raw[move] *= self.nakade_bias

            # Eye-attack bias (打眼/殺眼): boost moves that invade or constrain the
            # eye-forming space of opponent groups that haven't yet secured two real
            # eyes.  Direct liberty invasion gets full multiplier; adjacent pressure
            # cells get 65% (matching the life_urgency edge scaling).  This is the
            # offensive complement to life_urgency — where life_urgency helps our own
            # groups survive, eye_attack actively kills opponent groups.
            if self.eye_attack_bias > 1.0:
                eye_atk = _get_eye_attack_moves(board, color)
                for move, mult in eye_atk.items():
                    if move in raw:
                        raw[move] *= self.eye_attack_bias * mult

            # Weak group pressure: tighten the net around opponent groups with 2-3 liberties.
            if self.weak_group_pressure_bias > 1.0:
                pressure = _get_weak_group_pressure_moves(board, color)
                for move in raw:
                    if move in pressure:
                        raw[move] *= self.weak_group_pressure_bias

            # Semeai bias: in adjacent liberty races, fill opponent's exclusive liberties first.
            if self.semeai_bias > 1.0:
                semeai = _get_semeai_moves(board, color)
                for move in raw:
                    if move in semeai:
                        raw[move] *= self.semeai_bias

            # Hane at the head: apply pressure to the end of opponent stone lines.
            if self.hane_bias > 1.0:
                hane = _get_hane_at_head_moves(board, color)
                for move in raw:
                    if move in hane:
                        raw[move] *= self.hane_bias

            # Connection guard: boost threatened bamboo joints / bridge points.
            if self.connection_guard_bias > 1.0:
                guard = _get_vulnerable_connection_moves(board, color)
                for move in raw:
                    if move in guard:
                        raw[move] *= self.connection_guard_bias

            # Dual-purpose attack: boost moves that simultaneously pressure a weak
            # opponent group (2–3 liberties) AND are adjacent to own stones.
            # These moves attack while building our own position — 攻守兼備.
            if self.dual_purpose_bias > 1.0:
                dual = _get_dual_purpose_attack_moves(board, color)
                for move in raw:
                    if move in dual:
                        raw[move] *= self.dual_purpose_bias

            # Corner consolidation: when we already have stones in a corner, boost
            # moves that deepen that presence — corners are the easiest place to
            # form two eyes (金角) so securing them before the opponent invades
            # is almost always the right priority.
            if self.corner_consolidation_bias > 1.0:
                corner_moves = _get_corner_consolidation_moves(board, color)
                for move in raw:
                    if move in corner_moves:
                        raw[move] *= self.corner_consolidation_bias

            # Jump connection: boost one-point-jump cells that bridge two own
            # stone clusters.  Jumps are harder to cut than solid chains and
            # more efficient than direct connections — preferred when linking
            # two separate positions (兩個陣地之間用跳連接).
            if self.jump_connection_bias > 1.0:
                jump_moves = _get_jump_connection_moves(board, color)
                for move in raw:
                    if move in jump_moves:
                        raw[move] *= self.jump_connection_bias

            # Ikken tobi (一間跳): primary extension/connection tool.
            # Boosts ALL one-point jumps (2 orthogonal steps, empty middle)
            # from any own stone — both pure extension into new territory and
            # bridges between clusters.  This is the fastest safe way to expand
            # influence and connect groups across the board.
            if self.ikken_tobi_bias > 1.0:
                ikken_moves = _get_ikken_tobi_moves(board, color)
                for move in raw:
                    if move in ikken_moves:
                        raw[move] *= self.ikken_tobi_bias

            # Nobi (extension/長): boost moves that extend own stone chains by
            # one step with no opponent pressure — pure directional development.
            # Exempt moves already captured by connection penalty (own_adj >= 2).
            if self.nobi_bias > 1.0:
                nobi_moves = _get_nobi_moves(board, color)
                for move in raw:
                    if move in nobi_moves:
                        raw[move] *= self.nobi_bias

            # Kosumi (小飛/尖): primary diagonal connection and territory tool.
            # Boosts ALL diagonal extensions from own stones.  Together with
            # 一間跳 these two are the main means of connecting and expanding.
            if self.kosumi_bias > 1.0:
                kosumi_moves = _get_kosumi_moves(board, color)
                for move in raw:
                    if move in kosumi_moves:
                        raw[move] *= self.kosumi_bias

            # Keima (小飛): boost knight's-move cells for fast enclosure.
            # Faster than kosumi for sealing escape routes and building framework,
            # though cuttable — useful for chasing and blocking on the small board.
            if self.keima_bias > 1.0:
                keima_moves = _get_keima_moves(board, color)
                for move in raw:
                    if move in keima_moves:
                        raw[move] *= self.keima_bias

            # Tsuke (碰/靠): boost pure contact plays against opponent stones.
            # Direct attachment forces an immediate response — the fastest way
            # to invade, settle, or start a fight in 9x9's compressed space.
            if self.tsuke_bias > 1.0:
                tsuke_moves = _get_tsuke_moves(board, color)
                for move in raw:
                    if move in tsuke_moves:
                        raw[move] *= self.tsuke_bias

            # Clamp (夾): boost moves that squeeze opponent edge groups using
            # the board boundary.  The 9x9 edge is always nearby — a clamp
            # can instantly seize the opponent's base (根據地).
            if self.clamp_bias > 1.0:
                clamp_moves = _get_clamp_moves(board, color)
                for move in raw:
                    if move in clamp_moves:
                        raw[move] *= self.clamp_bias

            # Ogeima (大飛): large knight's move — wider than keima for fast
            # framework building and edge slides (邊上大飛).
            if self.ogeima_bias > 1.0:
                ogeima_moves = _get_ogeima_moves(board, color)
                for move in raw:
                    if move in ogeima_moves:
                        raw[move] *= self.ogeima_bias

            # Territory bridge: cells reachable via shape move (尖/小飛/大飛/跳)
            # from TWO+ own clusters — plays there connect groups and protect the edge.
            if self.territory_bridge_bias > 1.0:
                bridge_moves = _get_territory_bridge_moves(board, color)
                for move in raw:
                    if move in bridge_moves:
                        raw[move] *= self.territory_bridge_bias

            # Invasion: open cells inside opponent's developing framework where
            # we have less nearby influence — entering before the moyo closes.
            if self.invasion_bias > 1.0:
                invasion_moves = _get_moyo_invasion_moves(board, color)
                for move in raw:
                    if move in invasion_moves:
                        raw[move] *= self.invasion_bias

            # Life urgency: boost moves that help own groups with < 2 real eyes
            # (真眼) survive.  Groups with 0 eyes (mult=2.0) receive a stronger
            # boost than groups with 1 eye (mult=1.5), scaled by life_urgency_bias.
            # This prevents the AI from passively letting its own territory die (死棋).
            if self.life_urgency_bias > 1.0:
                urgency = _get_life_urgency_moves(board, color)
                for move, mult in urgency.items():
                    if move in raw:
                        raw[move] *= self.life_urgency_bias * mult

            # Pattern response: boost tsuke (contact) and peep-block responses.
            if self.pattern_response_bias > 0.0 and last_opp_move is not None:
                pattern_resp = _get_pattern_response_moves(board, color, last_opp_move)
                for move, mult in pattern_resp.items():
                    if move in raw:
                        raw[move] *= 1.0 + (mult - 1.0) * self.pattern_response_bias

            # Self-atari penalty: heavily discount moves that leave the placed group
            # with exactly 1 liberty.  Captures and escapes are exempt — capturing an
            # opponent group in atari may look like self-atari but is correct play.
            if self.self_atari_penalty < 1.0:
                for move in raw:
                    if move == _PASS or move in cap_moves or move in esc_moves:
                        continue
                    r, c = move
                    if _creates_self_atari(board, r, c, color):
                        raw[move] *= self.self_atari_penalty

            # Purposeless penalty (無意義手): heavily discount moves that accomplish
            # nothing — no nearby opponent, no own group in need, no eye creation,
            # no territory contest.  Exempt captures, escapes, and atari threats.
            if self.purposeless_penalty < 1.0:
                exempt = cap_moves | esc_moves | set(atari_threats)
                purposeless = _get_purposeless_moves(board, color, list(raw), radius=3)
                for move in purposeless:
                    if move in raw and move not in exempt and move != _PASS:
                        raw[move] *= self.purposeless_penalty

            # Edge penalty: heavily discount first-line moves with no tactical purpose.
            # Exemptions: capture, escape, double-atari, nakade vital points, semeai fills,
            # making/blocking eyes — all of these can legitimately occur on the first line.
            if self.edge_penalty < 1.0:
                tactical = (cap_moves | esc_moves | da_moves
                            | _nakade_vital_points(board, color)
                            | _get_semeai_moves(board, color))
                for move in raw:
                    if move == _PASS:
                        continue
                    r, c = move
                    if min(r, BOARD_SIZE - 1 - r, c, BOARD_SIZE - 1 - c) > 0:
                        continue  # not a first-line cell
                    if move in tactical:
                        continue
                    own_e, opp_e = _eye_score(board, r, c, color)
                    if own_e > 0 or opp_e > 0:
                        continue
                    raw[move] *= self.edge_penalty

        # --- Pass move ---
        # Pass is treated as a special child with prior derived from the
        # value head (only worth passing when we're already winning).
        # Training vs gameplay differ deliberately:
        #   - Training: passing is allowed any time so self-play games
        #     terminate cleanly via double-pass.  After one pass we
        #     boost pass_weight 5× to encourage a matching pass.
        #   - Gameplay: we never *initiate* a pass — passing is only
        #     considered after the opponent has just passed (so we can
        #     accept and end the game when winning).  This keeps the AI
        #     from prematurely conceding territory it could still play.
        if self.pass_weight > 0.0:
            if self.training:
                effective_pw = self.pass_weight * (5.0 if consecutive_passes >= 1 else 1.0)
                pass_raw = max(0.0, value) * effective_pw
            else:
                pass_raw = (max(0.0, value) * self.pass_weight * 5.0
                            if consecutive_passes >= 1 else 0.0)
            if pass_raw > 0.0:
                raw[_PASS] = pass_raw

        if add_noise:
            board_moves = [m for m in raw if m != _PASS]
            if board_moves:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(board_moves))
                for k, move in enumerate(board_moves):
                    raw[move] = ((1.0 - self.dirichlet_eps) * raw[move]
                                 + self.dirichlet_eps * float(noise[k]))

        total = sum(raw.values()) or 1.0
        for move, prior in raw.items():
            node.children[move] = MCTSNode(
                prior=prior / total, parent=node, move=move
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_ai(
    num_simulations: int = 400,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    spatial_bias: float = 1.0,
    training: bool = False,
    eval_batch_size: Optional[int] = None,
    eye_bias: float = 4.0,
    anti_eye_bias: float = 1.5,
    pass_weight: float = 0.15,
    resign_threshold: Optional[float] = None,
    capture_bias: float = 5.0,
    atari_threat_bias: float = 4.0,
    atari_escape_bias: float = 4.0,
    connection_penalty: float = 0.35,
    double_atari_bias: float = 3.0,
    gameplay_temperature: float = 0.8,
    edge_penalty: float = 0.04,
    empty_triangle_penalty: float = 0.6,
    nakade_bias: float = 5.0,
    eye_attack_bias: float = 4.5,
    cut_bias: float = 2.5,
    tiger_mouth_bias: float = 1.5,
    ko_threat_multiplier: float = 1.5,
    boundary_bias: float = 2.0,
    weak_group_pressure_bias: float = 2.0,
    semeai_bias: float = 3.0,
    hane_bias: float = 2.0,
    connection_guard_bias: float = 2.5,
    pattern_response_bias: float = 1.0,
    self_atari_penalty: float = 0.10,
    purposeless_penalty: float = 0.05,
    dual_purpose_bias: float = 2.0,
    corner_consolidation_bias: float = 2.0,
    jump_connection_bias: float = 2.5,
    ikken_tobi_bias: float = 5.5,
    nobi_bias: float = 1.4,
    kosumi_bias: float = 6.0,
    keima_bias: float = 1.3,
    tsuke_bias: float = 1.5,
    clamp_bias: float = 2.0,
    edge_eye_bias: float = 3.5,
    ogeima_bias: float = 1.8,
    territory_bridge_bias: float = 3.5,
    invasion_bias: float = 2.0,
    life_urgency_bias: float = 4.0,
) -> MCTS:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if eval_batch_size is None:
        # CPU: batch=8 still cuts NN-call overhead 8×; MPS/CUDA: even more benefit
        eval_batch_size = 8
    model = GoNet()
    if model_path:
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    return MCTS(model, num_simulations=num_simulations, device=device,
                spatial_bias=spatial_bias, training=training,
                eval_batch_size=eval_batch_size,
                eye_bias=eye_bias, anti_eye_bias=anti_eye_bias,
                pass_weight=pass_weight, resign_threshold=resign_threshold,
                capture_bias=capture_bias, atari_threat_bias=atari_threat_bias,
                atari_escape_bias=atari_escape_bias,
                connection_penalty=connection_penalty,
                double_atari_bias=double_atari_bias,
                gameplay_temperature=gameplay_temperature,
                edge_penalty=edge_penalty,
                empty_triangle_penalty=empty_triangle_penalty,
                nakade_bias=nakade_bias,
                eye_attack_bias=eye_attack_bias,
                cut_bias=cut_bias,
                tiger_mouth_bias=tiger_mouth_bias,
                ko_threat_multiplier=ko_threat_multiplier,
                boundary_bias=boundary_bias,
                weak_group_pressure_bias=weak_group_pressure_bias,
                semeai_bias=semeai_bias,
                hane_bias=hane_bias,
                connection_guard_bias=connection_guard_bias,
                pattern_response_bias=pattern_response_bias,
                self_atari_penalty=self_atari_penalty,
                purposeless_penalty=purposeless_penalty,
                dual_purpose_bias=dual_purpose_bias,
                corner_consolidation_bias=corner_consolidation_bias,
                jump_connection_bias=jump_connection_bias,
                ikken_tobi_bias=ikken_tobi_bias,
                nobi_bias=nobi_bias,
                kosumi_bias=kosumi_bias,
                keima_bias=keima_bias,
                tsuke_bias=tsuke_bias,
                clamp_bias=clamp_bias,
                edge_eye_bias=edge_eye_bias,
                ogeima_bias=ogeima_bias,
                territory_bridge_bias=territory_bridge_bias,
                invasion_bias=invasion_bias,
                life_urgency_bias=life_urgency_bias)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import cProfile, io, pstats, time

    print("Sanity check: 10-plane encoding + 128ch/5block network...")
    g  = GoEngine()
    t  = encode_board(g)
    assert t.shape == (NUM_PLANES, BOARD_SIZE, BOARD_SIZE), t.shape

    g3 = GoEngine()
    g3.board[3][4] = WHITE; g3.board[5][4] = WHITE; g3.board[4][3] = WHITE
    g3.board[4][4] = BLACK
    t3 = encode_board(g3)
    assert t3[3, 4, 4] == 1.0, "Own-atari plane should be 1 at (4,4)"
    print("  PASS: atari plane correct")

    g4 = GoEngine()
    g4.board[4][3] = BLACK; g4.board[4][5] = BLACK
    g4.board[3][4] = BLACK; g4.board[5][4] = BLACK
    t4 = encode_board(g4)
    assert t4[8, 4, 4] == 1.0, "Eye plane should be 1 at (4,4)"
    print("  PASS: eye plane correct")

    g5 = GoEngine()
    g5.board[0][1] = BLACK; g5.board[1][0] = BLACK
    t5 = encode_board(g5)
    assert t5[8, 0, 0] == 1.0, "Corner eye plane should be 1 at (0,0)"
    print("  PASS: corner eye correct")

    ai = create_ai(num_simulations=20)
    g6 = GoEngine()
    m  = ai.select_move(g6)
    print(f"  AI plays: {m}")

    total_params = sum(p.numel() for p in ai.model.parameters())
    print(f"  Network params: {total_params:,}")
    print("ai.py OK\n")

    # --- Speed benchmark ---
    print("Speed benchmark: 200 sims, batch=1 vs batch=8 ...")

    def bench(batch_size: int, n_sims: int = 200) -> float:
        a = create_ai(num_simulations=n_sims, eval_batch_size=batch_size)
        gb = GoEngine(); gb.play(4, 4)
        t0 = time.perf_counter()
        a.select_move(gb)
        return time.perf_counter() - t0

    t1 = bench(1)
    t8 = bench(8)
    print(f"  batch=1 : {t1*1000:.1f} ms")
    print(f"  batch=8 : {t8*1000:.1f} ms  ({t1/t8:.1f}× faster)")
