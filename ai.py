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
# Move-selection priority order in `select_move` (gameplay mode only;
# in training mode steps 1–3 are skipped so the search distribution
# stays unbiased):
#
#   1. Capture: if any legal move captures an opponent group in atari,
#      take the smallest such move immediately.
#   2. Forced atari escape: if exactly one of our groups in atari can be
#      saved by exactly one move, AND that escape is not futile (a
#      one-step lookahead shows it gains real liberties), play it.
#   3. Opening 3-3 corner points: while we have fewer than four stones
#      on the board, claim 3-3 corners in a fixed order, skipping any
#      that would fill an own eye.
#   4. MCTS: run num_simulations PUCT rollouts and pick the child with
#      the most visits (the AlphaZero standard). The pass move is a
#      special child handled separately — see `_create_children` for
#      the training-vs-gameplay difference.

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

_LINE_SCORES = [0.35, 0.75, 1.70, 1.40, 0.55]


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

# Opening priorities: claim all four 3-3 corner points before switching to MCTS.
# Order: top-left → bottom-right (diagonal) → top-right → bottom-left.
_OPENING_PRIORITIES: List[Tuple[int, int]] = [(2, 2), (6, 6), (2, 6), (6, 2)]

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
    """Return True if placing `color` at (r,c) merely connects already-adjacent own stones
    with no opponent pressure nearby.

    Heuristic: ≥2 orthogonal own-stone neighbors AND 0 opponent neighbors.
    Connections under pressure (any opponent neighbor) are NOT penalised.
    """
    opp = WHITE if color == BLACK else BLACK
    own_adj = opp_adj = 0
    for dr, dc in _NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            if board[nr][nc] == color:
                own_adj += 1
            elif board[nr][nc] == opp:
                opp_adj += 1
    return own_adj >= 2 and opp_adj == 0


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
        eye_bias: float = 1.0,
        anti_eye_bias: float = 0.8,
        pass_weight: float = 0.15,
        resign_threshold: Optional[float] = None,
        capture_bias: float = 5.0,
        atari_escape_bias: float = 4.0,
        connection_penalty: float = 0.35,
        double_atari_bias: float = 3.0,
        gameplay_temperature: float = 0.8,
        edge_penalty: float = 0.04,
        empty_triangle_penalty: float = 0.6,
        nakade_bias: float = 3.0,
        cut_bias: float = 2.5,
        tiger_mouth_bias: float = 1.5,
        ko_threat_multiplier: float = 1.5,
        boundary_bias: float = 2.0,
        weak_group_pressure_bias: float = 2.0,
        semeai_bias: float = 3.0,
        hane_bias: float = 2.0,
        connection_guard_bias: float = 2.5,
        pattern_response_bias: float = 1.0,
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
        # capture_bias: prior multiplier for moves that capture opponent in atari
        # atari_escape_bias: prior multiplier for moves that escape own atari
        # connection_penalty: prior multiplier for moves that only connect safe own groups
        self.capture_bias         = capture_bias
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
        self._eval_cache: Dict[bytes, Tuple[np.ndarray, float]] = {}
        self._opening_priorities  = random.sample(_OPENING_PRIORITIES, len(_OPENING_PRIORITIES))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_game(self) -> None:
        """Reshuffle opening priorities and clear the eval cache for a new game."""
        self._opening_priorities = random.sample(_OPENING_PRIORITIES, len(_OPENING_PRIORITIES))
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

        if not self.training:
            legal_set = set(legal)
            cap_moves, esc_moves = _get_tactical_moves(engine.board, engine.current_player)

            # 1. Captures: always take immediately (highest priority).
            legal_caps = sorted(cap_moves & legal_set)
            if legal_caps:
                return legal_caps[0]

            # 2. Escape own atari: force when there is exactly one escape candidate
            #    AND the escape gains real liberties AND it does not lead into a ladder
            #    AND not escaping would NOT produce a beneficial snapback.
            #    Multiple escapes, futile escapes, ladder escapes, or snapbacks → MCTS.
            legal_escs = sorted(esc_moves & legal_set)
            if len(legal_escs) == 1:
                esc = legal_escs[0]
                snapback_skips = _find_snapback_escapes(engine.board, engine.current_player)
                if (esc not in snapback_skips
                        and not _escape_is_futile(engine, esc)
                        and not _is_losing_ladder(engine, esc)):
                    return esc

            # 3. Opening corners (only when no tactical urgency, and not filling own eye).
            own_stones = sum(
                engine.board[r][c] == engine.current_player
                for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
            )
            if own_stones < len(self._opening_priorities):
                cur = engine.current_player
                for move in self._opening_priorities:
                    r, c = move
                    if engine.is_legal(r, c) and not _is_own_eye(engine.board, r, c, cur):
                        return move

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

            # Capture bias: scale by log(group_size) so capturing large groups
            # gets proportionally more exploration than capturing lone stones.
            if self.capture_bias > 1.0:
                for move, size in capture_sizes.items():
                    if move in raw:
                        raw[move] *= self.capture_bias * (1.0 + 0.5 * math.log1p(size - 1))

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

            # Pattern response: boost tsuke (contact) and peep-block responses.
            if self.pattern_response_bias > 0.0 and last_opp_move is not None:
                pattern_resp = _get_pattern_response_moves(board, color, last_opp_move)
                for move, mult in pattern_resp.items():
                    if move in raw:
                        raw[move] *= 1.0 + (mult - 1.0) * self.pattern_response_bias

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
    eye_bias: float = 1.0,
    anti_eye_bias: float = 0.8,
    pass_weight: float = 0.15,
    resign_threshold: Optional[float] = None,
    capture_bias: float = 5.0,
    atari_escape_bias: float = 4.0,
    connection_penalty: float = 0.35,
    double_atari_bias: float = 3.0,
    gameplay_temperature: float = 0.8,
    edge_penalty: float = 0.04,
    empty_triangle_penalty: float = 0.6,
    nakade_bias: float = 3.0,
    cut_bias: float = 2.5,
    tiger_mouth_bias: float = 1.5,
    ko_threat_multiplier: float = 1.5,
    boundary_bias: float = 2.0,
    weak_group_pressure_bias: float = 2.0,
    semeai_bias: float = 3.0,
    hane_bias: float = 2.0,
    connection_guard_bias: float = 2.5,
    pattern_response_bias: float = 1.0,
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
                capture_bias=capture_bias, atari_escape_bias=atari_escape_bias,
                connection_penalty=connection_penalty,
                double_atari_bias=double_atari_bias,
                gameplay_temperature=gameplay_temperature,
                edge_penalty=edge_penalty,
                empty_triangle_penalty=empty_triangle_penalty,
                nakade_bias=nakade_bias,
                cut_bias=cut_bias,
                tiger_mouth_bias=tiger_mouth_bias,
                ko_threat_multiplier=ko_threat_multiplier,
                boundary_bias=boundary_bias,
                weak_group_pressure_bias=weak_group_pressure_bias,
                semeai_bias=semeai_bias,
                hane_bias=hane_bias,
                connection_guard_bias=connection_guard_bias,
                pattern_response_bias=pattern_response_bias)


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
