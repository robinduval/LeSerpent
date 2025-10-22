# groupe22/snake-algo.py
"""
Agent Snake optimisé (Groupe 22) — GUI seulement.

- A* tail-aware (heuristique Manhattan torique) + vérif de sécurité légère
- Fallback : A* vers la queue, puis Hamiltonien
- Tables pré-calculées : voisins, index hamiltonien
- 100% compatible avec serpent.py (Snake, Apple, constantes...)

Usage :
    python groupe22/snake-algo.py
"""

import heapq
import time
import random
import pygame
import os
import sys
from collections import deque

# === Import robuste du module 'serpent' ======================================
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import serpent as s
    _ = s.UP
except Exception:
    serpent_path = os.path.join(REPO_ROOT, 'serpent.py')
    if not os.path.exists(serpent_path):
        raise
    import importlib.util
    spec = importlib.util.spec_from_file_location('serpent_local', serpent_path)
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)

# === Environnement GUI (WSLg / Linux) ========================================
def _prepare_gui_env():
    """
    Sélectionne un driver vidéo SDL valide sous WSLg et coupe l'audio (ALSA).
    Essaie Wayland d'abord, puis X11. Ne touche pas à 'dummy' ici.
    """
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if os.path.exists("/mnt/wslg/runtime-dir/wayland-0") or os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "wayland")
        os.environ.setdefault("WAYLAND_DISPLAY", os.environ.get("WAYLAND_DISPLAY", "wayland-0"))
    else:
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")

# === Constantes & tables pré-calculées =======================================
DIRECTIONS = (s.UP, s.LEFT, s.RIGHT, s.DOWN)
GRID_SIZE = int(s.GRID_SIZE)

def _build_neighbors_table(n: int):
    tbl = {}
    for y in range(n):
        for x in range(n):
            tbl[(x, y)] = (
                ((x + s.UP[0]) % n, (y + s.UP[1]) % n),
                ((x + s.LEFT[0]) % n, (y + s.LEFT[1]) % n),
                ((x + s.RIGHT[0]) % n, (y + s.RIGHT[1]) % n),
                ((x + s.DOWN[0]) % n, (y + s.DOWN[1]) % n),
            )
    return tbl

NEIGHBORS = _build_neighbors_table(GRID_SIZE)

def toroidal_manhattan(a, b):
    dx = abs(a[0] - b[0]); dy = abs(a[1] - b[1])
    if dx > GRID_SIZE - dx: dx = GRID_SIZE - dx
    if dy > GRID_SIZE - dy: dy = GRID_SIZE - dy
    return dx + dy

# === Hamiltonien serpentin + index inverse ===================================
def _build_hamiltonian_serpentine(n: int):
    path = []
    for y in range(n):
        row = [(x, y) for x in range(n)]
        if y % 2 == 1:
            row.reverse()
        path.extend(row)
    return path

HAM_PATH = _build_hamiltonian_serpentine(GRID_SIZE)
HAM_INDEX = {cell: i for i, cell in enumerate(HAM_PATH)}
HAM_LEN = len(HAM_PATH)

def _dir_from_to(a, b):
    ax, ay = a
    for d in DIRECTIONS:
        nx = (ax + d[0]) % GRID_SIZE
        ny = (ay + d[1]) % GRID_SIZE
        if (nx, ny) == b:
            return d
    return None

def next_on_hamiltonian(head_pos):
    idx = HAM_INDEX.get((head_pos[0], head_pos[1]))
    if idx is None:
        best, bestd = None, None
        for c in HAM_PATH:
            d = toroidal_manhattan(head_pos, c)
            if bestd is None or d < bestd:
                best, bestd = c, d
        return _dir_from_to(head_pos, best) if best else random.choice(DIRECTIONS)
    nxt = HAM_PATH[(idx + 1) % HAM_LEN]
    return _dir_from_to(head_pos, nxt) or random.choice(DIRECTIONS)

# === BFS utilitaires =========================================================
def bfs_reachable(start, goal, blocked):
    if start == goal:
        return True
    q = deque([start]); seen = {start}
    while q:
        cur = q.popleft()
        for n in NEIGHBORS[cur]:
            if n == goal:
                return True
            if n in seen or n in blocked:
                continue
            seen.add(n); q.append(n)
    return False

# === A* tail-aware ===========================================================
def astar_tail_aware(start, goal, body_list):
    """
    A* tenant compte du déplacement de la queue :
    - Après t pas, la queue a libéré exactement t cases.
    - On bloque body_list[: L - t] (en autorisant 'goal').
    """
    if goal is None or start == goal:
        return [start, goal] if goal else None

    L = len(body_list)
    def blocked_at_time(t):
        k = max(0, L - t)  # libère t cases depuis la queue
        return set(body_list[:k])

    open_heap = []
    g = {start: 0}
    came = {}
    closed = set()
    heapq.heappush(open_heap, (toroidal_manhattan(start, goal), 0, start))

    while open_heap:
        f, cg, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            return list(reversed(path))
        closed.add(cur)

        next_t = cg + 1
        blocked = blocked_at_time(next_t)
        for n in NEIGHBORS[cur]:
            if n in blocked and n != goal:
                continue
            ng = next_t
            if ng < g.get(n, 1 << 30):
                g[n] = ng; came[n] = cur
                fscore = ng + toroidal_manhattan(n, goal)
                heapq.heappush(open_heap, (fscore, ng, n))
    return None

# === Heuristiques de mouvement ==============================================
def choose_safe_move(snake, apple):
    head = (snake.head_pos[0], snake.head_pos[1])
    body = snake.body
    blocked = {(p[0], p[1]) for p in body[:-1]}  # queue autorisée (elle bouge)
    safe_dirs = [d for d in DIRECTIONS
                 if ((head[0]+d[0])%GRID_SIZE, (head[1]+d[1])%GRID_SIZE) not in blocked]
    if not safe_dirs:
        return snake.direction
    if apple is None or apple.position is None:
        return random.choice(safe_dirs)

    goal = (apple.position[0], apple.position[1])
    best, best_dirs = None, []
    for d in safe_dirs:
        nx, ny = (head[0]+d[0])%GRID_SIZE, (head[1]+d[1])%GRID_SIZE
        dist = toroidal_manhattan((nx, ny), goal)
        if best is None or dist < best:
            best, best_dirs = dist, [d]
        elif dist == best:
            best_dirs.append(d)
    if len(best_dirs) == 1:
        return best_dirs[0]

    full_blocked = {(p[0], p[1]) for p in body}
    def area_from(pos):
        q = deque([pos]); seen = {pos}
        while q:
            cur = q.popleft()
            for n in NEIGHBORS[cur]:
                if n in seen or n in full_blocked:
                    continue
                seen.add(n); q.append(n)
        return len(seen)

    best_area, best_dir = -1, None
    for d in best_dirs:
        cand = ((head[0]+d[0])%GRID_SIZE, (head[1]+d[1])%GRID_SIZE)
        a = area_from(cand)
        if a > best_area:
            best_area, best_dir = a, d

    # micro tiebreaker : éviter demi-tour si possible
    cur = snake.direction
    for d in best_dirs:
        if (d[0] == -cur[0] and d[1] == -cur[1]) and len(best_dirs) > 1:
            best_dirs = [x for x in best_dirs if x != d] + [d]
            break
    return best_dir or random.choice(best_dirs)

def path_is_safe_follow_tail(path, snake, apple):
    """Suit `path`, puis vérifie que la tête peut rejoindre la queue simulée."""
    if not path or len(path) < 2:
        return False
    sim_body = [(p[0], p[1]) for p in snake.body]
    sim_head = (snake.head_pos[0], snake.head_pos[1])
    apple_pos = (apple.position[0], apple.position[1]) if apple and apple.position else None
    grew = False
    for step in path[1:]:
        sim_body.insert(0, step); sim_head = step
        if not grew and apple_pos and step == apple_pos:
            grew = True
        else:
            sim_body.pop()
    tail = sim_body[-1]
    blocked = set(sim_body[:-1])
    return bfs_reachable(sim_head, tail, blocked)

# === Boucle de jeu GUI uniquement ===========================================
def run_agent():
    """Lance une partie avec interface graphique (vitesse = s.GAME_SPEED)."""
    _prepare_gui_env()
    pygame.init()
    screen = pygame.display.set_mode((s.SCREEN_WIDTH, s.SCREEN_HEIGHT))
    pygame.display.set_caption("Snake - Agent optimisé (Groupe 22)")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    snake = s.Snake()
    apple = s.Apple(snake.body)
    running = True; game_over = victory = False
    start_time = time.time(); planned_path = None
    moves_per_sec = max(1, int(getattr(s, "GAME_SPEED", 10)))
    move_interval = 1.0 / float(moves_per_sec)
    last_move = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        now = time.time()
        while (now - last_move) >= move_interval and not game_over and not victory:
            head = (snake.head_pos[0], snake.head_pos[1])
            goal = (apple.position[0], apple.position[1]) if apple.position else None
            body_list = [(p[0], p[1]) for p in snake.body]

            path_to_apple = astar_tail_aware(head, goal, body_list) if goal else None
            planned_path = path_to_apple
            chosen = None
            if path_to_apple and len(path_to_apple) >= 2:
                if path_is_safe_follow_tail(path_to_apple, snake, apple):
                    next_pos = path_to_apple[1]
                    chosen = _dir_from_to(head, next_pos)
            if chosen is None:
                tail = body_list[-1]
                path_to_tail = astar_tail_aware(head, tail, body_list)
                if path_to_tail and len(path_to_tail) >= 2:
                    next_pos = path_to_tail[1]
                    chosen = _dir_from_to(head, next_pos)
            if chosen is None:
                chosen = next_on_hamiltonian(head) or choose_safe_move(snake, apple)

            snake.set_direction(chosen)
            snake.move()
            if snake.is_game_over():
                game_over = True
            if apple.position and snake.head_pos == list(apple.position):
                snake.grow()
                if not apple.relocate(snake.body):
                    victory = True; game_over = True
            last_move += move_interval; now = time.time()

        # Rendu
        screen.fill(s.GRIS_FOND)
        game_area_rect = pygame.Rect(0, s.SCORE_PANEL_HEIGHT, s.SCREEN_WIDTH, s.SCREEN_WIDTH)
        pygame.draw.rect(screen, s.NOIR, game_area_rect)
        s.draw_grid(screen)
        if planned_path:
            for cell in planned_path:
                if (snake.head_pos[0], snake.head_pos[1]) == cell:
                    continue
                rect = pygame.Rect(cell[0]*s.CELL_SIZE, cell[1]*s.CELL_SIZE + s.SCORE_PANEL_HEIGHT, s.CELL_SIZE, s.CELL_SIZE)
                inner = rect.inflate(-6, -6)
                pygame.draw.rect(screen, (50,120,220), inner, border_radius=6)
        apple.draw(screen); snake.draw(screen)
        s.display_info(screen, font_main, snake, start_time)
        if game_over:
            msg = "VICTOIRE !" if victory else "GAME OVER"
            col = s.VERT if victory else s.ROUGE
            s.display_message(screen, font_game_over, msg, col)
            s.display_message(screen, font_main, "Fermez la fenêtre pour quitter.", s.BLANC, y_offset=100)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

# === Entrée CLI ==============================================================
if __name__ == "__main__":
    run_agent()
