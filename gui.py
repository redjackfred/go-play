# gui.py — pygame GUI for 9x9 Go  (4 game modes + colour selection)

from __future__ import annotations

import sys
import threading
from enum import Enum, auto
from typing import Optional, Tuple

import pygame

from board import BLACK, BOARD_SIZE, EMPTY, WHITE, GoEngine, opponent
from ai import create_ai, MCTS

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

CELL      = 56
MARGIN    = 44
BOARD_PX  = CELL * (BOARD_SIZE - 1)
PANEL_W   = 210
WIN_W     = MARGIN * 2 + BOARD_PX + PANEL_W + 24
WIN_H     = MARGIN * 2 + BOARD_PX + 64
STONE_R   = CELL // 2 - 3
STAR_PTS  = [(2,2),(2,6),(4,4),(6,2),(6,6)]

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

C_BOARD   = (210, 175, 115)
C_LINE    = (55,  35,  15)
C_BLACK   = (22,  22,  22)
C_WHITE   = (248, 248, 242)
C_BSTONE_H= (70,  70,  70)
C_WSTONE_S= (180, 180, 175)
C_PANEL   = (235, 210, 165)
C_TEXT    = (40,  20,  0)
C_BTN     = (175, 135, 75)
C_BTN_H   = (215, 165, 85)
C_BTN_SEL = (120, 180, 100)   # selected/active button
C_BTN_DIS = (150, 135, 115)   # disabled button (not human turn)
C_WIN     = (55,  130, 55)
C_STATUS  = (30,  30,  120)
C_OVERLAY = (0,   0,   0,  160)   # semi-transparent overlay for menu

# ---------------------------------------------------------------------------
# Game mode enum
# ---------------------------------------------------------------------------

class Mode(Enum):
    HUMAN_VS_AI  = auto()   # human=current_color,  AI=other
    AI_VS_HUMAN  = auto()   # AI=current_color,      human=other  (alias, same logic)
    HUMAN_VS_HUMAN = auto()
    AI_VS_AI     = auto()


MODE_LABELS = {
    Mode.HUMAN_VS_AI:    "Human  vs  AI",
    Mode.AI_VS_HUMAN:    "AI  vs  Human",
    Mode.HUMAN_VS_HUMAN: "Human  vs  Human",
    Mode.AI_VS_AI:       "AI  vs  AI",
}

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _cell_to_px(row: int, col: int) -> Tuple[int, int]:
    return MARGIN + col * CELL, MARGIN + row * CELL


def _px_to_cell(x: int, y: int) -> Optional[Tuple[int, int]]:
    col = (x - MARGIN + CELL // 2) // CELL
    row = (y - MARGIN + CELL // 2) // CELL
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        cx, cy = _cell_to_px(row, col)
        if abs(x - cx) < CELL // 2 and abs(y - cy) < CELL // 2:
            return row, col
    return None

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_board(surf: pygame.Surface) -> None:
    rect = pygame.Rect(
        MARGIN - CELL // 2, MARGIN - CELL // 2,
        BOARD_PX + CELL, BOARD_PX + CELL,
    )
    pygame.draw.rect(surf, C_BOARD, rect, border_radius=6)
    for i in range(BOARD_SIZE):
        x0, y0 = _cell_to_px(0, i);      x1, y1 = _cell_to_px(BOARD_SIZE-1, i)
        pygame.draw.line(surf, C_LINE, (x0,y0), (x1,y1), 1)
        x0, y0 = _cell_to_px(i, 0);      x1, y1 = _cell_to_px(i, BOARD_SIZE-1)
        pygame.draw.line(surf, C_LINE, (x0,y0), (x1,y1), 1)
    for r, c in STAR_PTS:
        pygame.draw.circle(surf, C_LINE, _cell_to_px(r,c), 4)


def draw_stone(surf, row, col, color, last=False):
    cx, cy = _cell_to_px(row, col)
    r = STONE_R
    if color == BLACK:
        pygame.draw.circle(surf, C_BLACK, (cx,cy), r)
        pygame.draw.circle(surf, C_BSTONE_H, (cx-r//3,cy-r//3), r//4)
    else:
        pygame.draw.circle(surf, C_WSTONE_S, (cx+2,cy+2), r)
        pygame.draw.circle(surf, C_WHITE, (cx,cy), r)
        pygame.draw.circle(surf, C_LINE, (cx,cy), r, 1)
    if last:
        dot_c = (200,200,200) if color == BLACK else (80,80,80)
        pygame.draw.circle(surf, dot_c, (cx,cy), max(3, r//4))


def draw_coord_labels(surf, font):
    letters = "ABCDEFGHI"
    for i in range(BOARD_SIZE):
        cx, cy = _cell_to_px(0, i)
        l = font.render(letters[i], True, C_TEXT)
        surf.blit(l, (cx - l.get_width()//2, MARGIN - CELL//2 - l.get_height() - 2))
        _, ry = _cell_to_px(i, 0)
        r = font.render(str(i), True, C_TEXT)
        surf.blit(r, (MARGIN - CELL//2 - r.get_width() - 4, ry - r.get_height()//2))


def draw_panel(surf, engine, ai_thinking, mode, human_color, fonts,
               btn_new_rect, btn_menu_rect, new_hover, menu_hover,
               btn_pass_rect, btn_decline_rect, btn_resign_rect,
               pass_hover, decline_hover, resign_hover,
               human_can_act):
    px = MARGIN * 2 + BOARD_PX + 12
    pygame.draw.rect(surf, C_PANEL, pygame.Rect(px-6, 0, PANEL_W, WIN_H))

    y = 20
    big, med, sml = fonts["big"], fonts["med"], fonts["sml"]

    def blit(text, font, color, indent=0):
        nonlocal y
        lbl = font.render(text, True, color)
        surf.blit(lbl, (px + indent, y))
        y += lbl.get_height() + 4

    blit("9x9 Go", big, C_TEXT)
    y += 6

    mode_str = MODE_LABELS[mode]
    col_str  = "Black" if human_color == BLACK else "White"
    if mode in (Mode.HUMAN_VS_AI, Mode.AI_VS_HUMAN):
        blit(f"{mode_str}", sml, C_TEXT)
        blit(f"You = {col_str}", sml, C_TEXT)
    else:
        blit(f"{mode_str}", sml, C_TEXT)
    y += 6

    if engine.game_over:
        name = "Black" if engine.winner == BLACK else "White"
        blit(f"{name} wins!", med, C_WIN)
        score = engine.get_score()
        blit(f"B: {score['black_score']:.1f}", sml, C_TEXT, 4)
        blit(f"W: {score['white_score']:.1f}  (+{score['komi']})", sml, C_TEXT, 4)
    else:
        cur = "Black" if engine.current_player == BLACK else "White"
        blit(f"Turn: {cur}", med, C_TEXT)
        if ai_thinking:
            blit("AI thinking...", sml, C_STATUS)

    y += 10
    blit("Captured by:", med, C_TEXT)
    blit(f"  Black: {engine.captured[BLACK]}", sml, C_TEXT)
    blit(f"  White: {engine.captured[WHITE]}", sml, C_TEXT)

    # Pass / Decline-Pass / Resign buttons
    pass_pending = engine.consecutive_passes >= 1 and not engine.game_over
    pass_label   = "Accept Pass" if pass_pending else "Pass"

    for rect, label, hover, always_show in (
        (btn_pass_rect,    pass_label,    pass_hover,    True),
        (btn_decline_rect, "不想Pass",    decline_hover, pass_pending),
        (btn_resign_rect,  "Resign",      resign_hover,  True),
    ):
        if not always_show:
            continue
        if human_can_act:
            bg = C_BTN_H if hover else C_BTN
            tc = C_TEXT
        else:
            bg = C_BTN_DIS
            tc = (120, 100, 80)
        pygame.draw.rect(surf, bg, rect, border_radius=6)
        pygame.draw.rect(surf, C_LINE, rect, 1, border_radius=6)
        lbl = med.render(label, True, tc)
        surf.blit(lbl, (rect.centerx - lbl.get_width()//2,
                        rect.centery - lbl.get_height()//2))

    # Hint under resign: only when no pass is pending (avoid clutter)
    if not pass_pending:
        hint = sml.render("(resign = concede)", True, (160, 50, 50))
        surf.blit(hint, (px, btn_resign_rect.bottom + 4))
    else:
        hint = sml.render("Opponent passed — accept or decline", True, C_STATUS)
        surf.blit(hint, (px, btn_decline_rect.bottom + 4))

    # New Game / Menu buttons
    for rect, label, hover in ((btn_new_rect, "New Game", new_hover),
                                (btn_menu_rect, "Menu", menu_hover)):
        pygame.draw.rect(surf, C_BTN_H if hover else C_BTN, rect, border_radius=6)
        pygame.draw.rect(surf, C_LINE, rect, 1, border_radius=6)
        lbl = med.render(label, True, C_TEXT)
        surf.blit(lbl, (rect.centerx - lbl.get_width()//2,
                        rect.centery - lbl.get_height()//2))

# ---------------------------------------------------------------------------
# Start / menu screen
# ---------------------------------------------------------------------------

class _Button:
    def __init__(self, rect: pygame.Rect, label: str):
        self.rect  = rect
        self.label = label

    def draw(self, surf, font, selected=False, hovered=False):
        c = C_BTN_SEL if selected else (C_BTN_H if hovered else C_BTN)
        pygame.draw.rect(surf, c, self.rect, border_radius=8)
        pygame.draw.rect(surf, C_LINE, self.rect, 2, border_radius=8)
        lbl = font.render(self.label, True, C_TEXT)
        surf.blit(lbl, (self.rect.centerx - lbl.get_width()//2,
                        self.rect.centery - lbl.get_height()//2))

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)


def run_menu(surf: pygame.Surface, clock, fonts) -> Tuple[Mode, int, int]:
    """
    Display a start menu.  Returns (mode, human_color, num_simulations).
    Blocks until the user clicks Start.
    """
    med, big, sml = fonts["med"], fonts["big"], fonts["sml"]
    W, H = surf.get_size()
    cx = W // 2

    mode_order = list(MODE_LABELS.keys())
    selected_mode = Mode.HUMAN_VS_AI
    selected_color = BLACK   # human colour (irrelevant for AI vs AI / H vs H)
    sims = 200

    bw = 260; bh = 42; gap = 10
    total_h = len(mode_order) * (bh + gap)
    start_y = H // 2 - total_h // 2

    mode_btns = [
        _Button(pygame.Rect(cx - bw//2, start_y + i*(bh+gap), bw, bh), MODE_LABELS[m])
        for i, m in enumerate(mode_order)
    ]

    color_y  = start_y + len(mode_order)*(bh+gap) + 24
    black_btn = _Button(pygame.Rect(cx - 130, color_y, 120, 38), "Play Black")
    white_btn = _Button(pygame.Rect(cx + 10,  color_y, 120, 38), "Play White")

    sim_y = color_y + 60
    less_btn  = _Button(pygame.Rect(cx - 90, sim_y, 40, 34), " - ")
    more_btn  = _Button(pygame.Rect(cx + 50, sim_y, 40, 34), " + ")

    start_btn = _Button(pygame.Rect(cx - 80, sim_y + 56, 160, 46), "Start Game")

    running = True
    while running:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, btn in enumerate(mode_btns):
                    if btn.hit(event.pos):
                        selected_mode = mode_order[i]
                if black_btn.hit(event.pos): selected_color = BLACK
                if white_btn.hit(event.pos): selected_color = WHITE
                if less_btn.hit(event.pos):  sims = max(50,  sims - 50)
                if more_btn.hit(event.pos):  sims = min(800, sims + 50)
                if start_btn.hit(event.pos): running = False

        surf.fill((160, 140, 100))

        title = big.render("9 x 9 Go", True, C_TEXT)
        surf.blit(title, (cx - title.get_width()//2, 30))
        sub = sml.render("Select game mode", True, C_TEXT)
        surf.blit(sub, (cx - sub.get_width()//2, 72))

        for i, btn in enumerate(mode_btns):
            btn.draw(surf, med,
                     selected=(mode_order[i] == selected_mode),
                     hovered=btn.hit((mx, my)))

        # Colour selection (only relevant when a human is playing)
        if selected_mode in (Mode.HUMAN_VS_AI, Mode.AI_VS_HUMAN, Mode.HUMAN_VS_HUMAN):
            lbl = sml.render("Your colour:", True, C_TEXT)
            surf.blit(lbl, (cx - lbl.get_width()//2, color_y - 22))
            black_btn.draw(surf, med, selected=(selected_color==BLACK), hovered=black_btn.hit((mx,my)))
            white_btn.draw(surf, med, selected=(selected_color==WHITE), hovered=white_btn.hit((mx,my)))
        else:
            lbl = sml.render("(AI vs AI — no colour choice)", True, (100,80,60))
            surf.blit(lbl, (cx - lbl.get_width()//2, color_y))

        # Simulations
        lbl = sml.render(f"AI strength (sims/move): {sims}", True, C_TEXT)
        surf.blit(lbl, (cx - lbl.get_width()//2, sim_y - 22))
        less_btn.draw(surf, med, hovered=less_btn.hit((mx,my)))
        more_btn.draw(surf, med, hovered=more_btn.hit((mx,my)))

        start_btn.draw(surf, fonts["big"], hovered=start_btn.hit((mx,my)))
        pygame.display.flip()
        clock.tick(60)

    return selected_mode, selected_color, sims

# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def run(num_simulations: int = 200, skip_menu: bool = False,
        initial_mode: Mode = Mode.HUMAN_VS_AI,
        initial_human_color: int = BLACK,
        model_path: Optional[str] = None) -> None:
    pygame.init()
    surf = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("9x9 Go")
    clock = pygame.time.Clock()

    fonts = {
        "big": pygame.font.SysFont("Arial", 22, bold=True),
        "med": pygame.font.SysFont("Arial", 17),
        "sml": pygame.font.SysFont("Arial", 14),
    }

    px_panel     = MARGIN * 2 + BOARD_PX + 12
    btn_pass     = pygame.Rect(px_panel + 8, WIN_H - 240, PANEL_W - 20, 36)
    btn_decline  = pygame.Rect(px_panel + 8, WIN_H - 196, PANEL_W - 20, 36)
    btn_resign   = pygame.Rect(px_panel + 8, WIN_H - 152, PANEL_W - 20, 36)
    btn_new      = pygame.Rect(px_panel + 8, WIN_H - 100, PANEL_W - 20, 36)
    btn_menu     = pygame.Rect(px_panel + 8, WIN_H - 56,  PANEL_W - 20, 36)

    def start_new_game():
        nonlocal engine, mode, human_color, ai, ai_thinking, ai_result
        mode, human_color, sims = run_menu(surf, clock, fonts)
        ai = create_ai(num_simulations=sims, model_path=model_path)
        engine = GoEngine()
        ai_thinking = False
        ai_result.clear()

    # ---------- first run ----------
    if skip_menu:
        mode, human_color, sims = initial_mode, initial_human_color, num_simulations
        ai = create_ai(num_simulations=sims, model_path=model_path)
    else:
        mode, human_color, sims = run_menu(surf, clock, fonts)
        ai = create_ai(num_simulations=sims, model_path=model_path)

    engine      = GoEngine()
    ai_thinking = False
    ai_result: list = []

    _AI_DECLINE  = "decline"  # AI chose to play → decline human's pending pass
    _AI_RESIGN   = "resign"   # AI's position is hopeless → resign
    _force_no_pass = [False]  # True after human rejects AI's pass: AI must play

    def launch_ai():
        if ai.check_resign(engine):
            ai_result.append(_AI_RESIGN)
            return
        force = _force_no_pass[0]
        _force_no_pass[0] = False
        move = ai.select_move(engine, allow_pass=not force)
        # If the human had a pending pass and AI has a real move, AI "declines":
        # return the turn to the human first; AI recalculates on its own next turn.
        if engine.consecutive_passes >= 1 and move is not None:
            ai_result.append(_AI_DECLINE)
        else:
            ai_result.append(move)

    def is_human_turn() -> bool:
        if mode == Mode.HUMAN_VS_HUMAN:
            return True
        if mode == Mode.AI_VS_AI:
            return False
        # HUMAN_VS_AI: human plays human_color
        # AI_VS_HUMAN: human plays human_color
        return engine.current_player == human_color

    def maybe_start_ai():
        nonlocal ai_thinking
        if not engine.game_over and not is_human_turn() and not ai_thinking:
            ai_thinking = True
            threading.Thread(target=launch_ai, daemon=True).start()

    # Trigger AI first move if AI goes first
    maybe_start_ai()

    running = True
    while running:
        mx, my = pygame.mouse.get_pos()
        new_hover     = btn_new.collidepoint(mx, my)
        menu_hover    = btn_menu.collidepoint(mx, my)
        pass_hover    = btn_pass.collidepoint(mx, my)
        decline_hover = btn_decline.collidepoint(mx, my)
        resign_hover  = btn_resign.collidepoint(mx, my)

        human_can_act = (not engine.game_over and not ai_thinking
                         and is_human_turn())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_new.collidepoint(event.pos):
                    engine = GoEngine()
                    ai.reset_game()
                    ai_result.clear(); ai_thinking = False
                    maybe_start_ai()
                elif btn_menu.collidepoint(event.pos):
                    start_new_game()
                    maybe_start_ai()
                elif human_can_act and btn_pass.collidepoint(event.pos):
                    engine.pass_move()
                    maybe_start_ai()
                elif (human_can_act and engine.consecutive_passes >= 1
                      and btn_decline.collidepoint(event.pos)):
                    engine.decline_pass()
                    _force_no_pass[0] = True  # AI must play a real move, not pass again
                    maybe_start_ai()   # may need to wake AI if turn returned to AI
                elif human_can_act and btn_resign.collidepoint(event.pos):
                    engine.resign()
                elif human_can_act:
                    cell = _px_to_cell(*event.pos)
                    if cell and engine.is_legal(*cell):
                        engine.play(*cell)
                        maybe_start_ai()

        # Consume AI result
        if ai_result:
            action = ai_result.pop(0)
            ai_thinking = False
            if action is _AI_RESIGN:
                engine.resign()          # AI concedes — game over, no AI turn needed
            elif action is _AI_DECLINE:
                engine.decline_pass()    # return turn to the human who passed
                # do NOT call maybe_start_ai() — it is now the human's turn
            elif action is None:
                engine.pass_move()
                maybe_start_ai()
            else:
                engine.play(*action)
                maybe_start_ai()   # chain: AI vs AI keeps going

        # ---- Render ----
        surf.fill((180, 160, 120))
        draw_board(surf)
        draw_coord_labels(surf, fonts["sml"])

        board = engine.get_board()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != EMPTY:
                    draw_stone(surf, r, c, board[r][c],
                               last=(engine.last_move == (r,c)))

        draw_panel(surf, engine, ai_thinking, mode, human_color, fonts,
                   btn_new, btn_menu, new_hover, menu_hover,
                   btn_pass, btn_decline, btn_resign,
                   pass_hover, decline_hover, resign_hover,
                   human_can_act)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sims",  type=int, default=200)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()
    run(num_simulations=args.sims, model_path=args.model)
