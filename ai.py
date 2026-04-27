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
        self.capture_bias       = capture_bias
        self.atari_escape_bias  = atari_escape_bias
        self.connection_penalty = connection_penalty
        self.double_atari_bias  = double_atari_bias
        self._eval_cache: Dict[bytes, Tuple[np.ndarray, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            #    AND the escape actually gains liberties (not a futile one-step chase).
            #    Multiple escapes, or a futile single escape → MCTS with atari_escape_bias.
            legal_escs = sorted(esc_moves & legal_set)
            if len(legal_escs) == 1 and not _escape_is_futile(engine, legal_escs[0]):
                return legal_escs[0]

            # 3. Opening corners (only when no tactical urgency, and not filling own eye).
            own_stones = sum(
                engine.board[r][c] == engine.current_player
                for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
            )
            if own_stones < len(_OPENING_PRIORITIES):
                cur = engine.current_player
                for move in _OPENING_PRIORITIES:
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

        best = max(root.children, key=lambda m: root.children[m].visit_count)
        if not allow_pass and best == _PASS:
            # Human rejected our pass — must play a real board move.
            non_pass = {m: ch for m, ch in root.children.items() if m != _PASS}
            if non_pass:
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

            board_i  = sim_i.board
            color_i  = sim_i.current_player
            cpasses_i = sim_i.consecutive_passes
            key = self._cache_key(sim_i)
            if key in self._eval_cache:
                policy, value = self._eval_cache[key]
                self._create_children(
                    paths[i][-1], expand_legal, policy,
                    self.training and (paths[i][-1] is root),
                    board=board_i, color=color_i, value=value,
                    consecutive_passes=cpasses_i,
                )
                leaf_values[i] = value
            else:
                to_eval.append((i, paths[i][-1], sim_i, leaf_legal, expand_legal,
                                 board_i, color_i, cpasses_i, key))

        if to_eval:
            # Stack boards into one batch tensor — one forward pass for all
            tensors = torch.stack([
                encode_board(sim, ll) for _, _, sim, ll, _, _, _, _, _ in to_eval
            ]).to(self.device)

            with torch.inference_mode():
                log_p, vt = self.model(tensors)
            policies   = log_p.exp().cpu().numpy()
            values_arr = vt.squeeze(1).cpu().numpy()

            for j, (i, leaf, sim, leaf_legal, expand_legal,
                    board_j, color_j, cpasses_j, key) in enumerate(to_eval):
                policy = policies[j]
                value  = float(values_arr[j])
                self._eval_cache[key] = (policy, value)
                self._create_children(
                    leaf, expand_legal, policy,
                    self.training and (leaf is root),
                    board=board_j, color=color_j, value=value,
                    consecutive_passes=cpasses_j,
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

        # --- Capture / atari-escape bias (gameplay only) ---
        # During gameplay we hugely amplify priors for moves that capture
        # an opponent group in atari or save one of our own groups in
        # atari.  These are nearly always urgent in real play.
        # Disabled during self-play training so the move distribution
        # stays natural and the network learns tactics from game outcomes
        # rather than from hand-tuned priors.
        if not self.training and board is not None and color is not None:
            if self.capture_bias > 1.0 or self.atari_escape_bias > 1.0:
                cap_moves, esc_moves = _get_tactical_moves(board, color)
                for move in raw:
                    if move in cap_moves:
                        raw[move] *= self.capture_bias
                    if move in esc_moves:
                        raw[move] *= self.atari_escape_bias

            # --- Connection penalty (gameplay only) ---
            # Beginners' nets often play solid but pointless connection
            # moves between groups that are already safe.  We multiply
            # the prior of any such move by connection_penalty (<1.0) so
            # MCTS shifts visits to higher-leverage moves.  Connections
            # under genuine pressure (any opponent neighbour) are exempt.
            if self.connection_penalty < 1.0:
                for move in raw:
                    if move != _PASS:
                        r, c = move
                        if _is_wasteful_connection(board, r, c, color):
                            raw[move] *= self.connection_penalty

            # --- Double atari bias (gameplay only) ---
            # A move that simultaneously threatens two opponent groups
            # (each currently at 2 liberties) is very strong — the
            # opponent can only save one.  Boost its prior so MCTS
            # explores it heavily without hard-overriding the NN.
            if self.double_atari_bias > 1.0:
                da_moves = _get_double_atari_moves(board, color)
                for move in raw:
                    if move in da_moves:
                        raw[move] *= self.double_atari_bias

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
                double_atari_bias=double_atari_bias)


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
