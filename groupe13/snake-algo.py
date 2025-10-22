"""
serpent-suite.py

Single-file tool that can run in two modes:
- Master (default): launches multiple worker instances and collects their scores, then shows a scoreboard window.
- Worker (--worker): runs a single pygame snake game with A* autoplayer and exits after game over or a timeout, printing "<id> <score>" to stdout.

Usage (master):
    python3 serpent-suite.py [N] [--timeout seconds]

Usage (worker): (managed by master automatically)
    python3 serpent-suite.py --worker --id 0 --timeout 30

Notes:
- Each worker opens its own pygame window. Running many windows may be heavy.
- Workers auto-close after the given timeout in seconds to ensure progress when stuck.
"""

import heapq
import math
import os
import random
import subprocess
import sys
import time
from collections import deque

# --- GAME CONSTANTS ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 5
SCORE_PANEL_HEIGHT = 80
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Colors
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Local FPS multiplier
LOCAL_FPS = max(30, GAME_SPEED * 12)

# --- GAME CLASSES ---
class Snake:
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos, [self.head_pos[0] - 1, self.head_pos[1]], [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        if (new_dir[0] * -1, new_dir[1] * -1) == self.direction:
            return
        # Prevent moving into body immediately unless it's the tail that moves away
        next_x = (self.head_pos[0] + new_dir[0]) % GRID_SIZE
        next_y = (self.head_pos[1] + new_dir[1]) % GRID_SIZE
        next_cell = [next_x, next_y]
        tail = self.body[-1]
        if next_cell in self.body and next_cell != tail:
            return
        self.direction = new_dir

    def move(self):
        new_head_x = (self.head_pos[0] + self.direction[0]) % GRID_SIZE
        new_head_y = (self.head_pos[1] + self.direction[1]) % GRID_SIZE
        new_head_pos = [new_head_x, new_head_y]
        self.body.insert(0, new_head_pos)
        self.head_pos = new_head_pos
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False

    def grow(self):
        self.grow_pending = True
        self.score += 1

    def check_self_collision(self):
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        return self.check_self_collision()

    def draw(self, surface, pygame):
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)
        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)

class Apple:
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(pos) not in occupied_positions]
        if not available_positions:
            return None
        return random.choice(available_positions)

    def relocate(self, snake_body):
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False

    def draw(self, surface, pygame):
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)

# --- DRAW HELPERS ---
def draw_grid(surface, pygame):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

def display_info(surface, font, snake, start_time, pygame):
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 20))
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_text = font.render(f"Temps: {minutes:02d}:{seconds:02d}", True, BLANC)
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 20))
    max_cells = GRID_SIZE * GRID_SIZE
    fill_rate = (len(snake.body) / max_cells) * 100
    fill_text = font.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
    surface.blit(fill_text, (SCREEN_WIDTH // 2 - fill_text.get_width() // 2, 20))

def display_message(surface, font, message, color, pygame, y_offset=0):
    text_surface = font.render(message, True, color)
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)

# --- PATH / AI HELPERS ---
def toroidal_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dx = min(dx, GRID_SIZE - dx)
    dy = min(dy, GRID_SIZE - dy)
    return dx + dy

def neighbors(pos):
    x, y = pos
    for d in DIRECTIONS:
        nx = (x + d[0]) % GRID_SIZE
        ny = (y + d[1]) % GRID_SIZE
        yield (nx, ny)

def astar(start, goal, occupied):
    if start == goal:
        return [start]
    open_heap = []
    heapq.heappush(open_heap, (toroidal_distance(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()
    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        for nb in neighbors(current):
            if nb in occupied:
                continue
            tentative_g = current_g + 1
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb] = tentative_g
                priority = tentative_g + toroidal_distance(nb, goal)
                heapq.heappush(open_heap, (priority, tentative_g, nb))
                came_from[nb] = current
    return None

def bfs_reachable(start, targets, occupied):
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        if cur in targets:
            return True
        for nb in neighbors(cur):
            if nb in seen or nb in occupied:
                continue
            seen.add(nb)
            q.append(nb)
    return False

def path_simulate_survivable(snake_body, path_to_food, apple_will_be_eaten):
    body = [tuple(seg) for seg in snake_body]
    grow = apple_will_be_eaten
    for step in path_to_food[1:]:
        body.insert(0, step)
        if not grow:
            body.pop()
        else:
            grow = False
    return body

def find_path_to_tail(snake_body):
    head = tuple(snake_body[0])
    tail = tuple(snake_body[-1])
    occupied = set(tuple(seg) for seg in snake_body)
    if tail in occupied:
        occupied.remove(tail)
    return astar(head, tail, occupied)

def next_direction_from_step(head, next_step):
    hx, hy = head
    for d in DIRECTIONS:
        if ((hx + d[0]) % GRID_SIZE, (hy + d[1]) % GRID_SIZE) == next_step:
            return d
    return None

def find_safe_direction(head, occupied):
    for d in DIRECTIONS:
        nx = (head[0] + d[0]) % GRID_SIZE
        ny = (head[1] + d[1]) % GRID_SIZE
        if (nx, ny) not in occupied:
            return d
    return None

# --- DRAW ARROWS ---
def draw_arrow(surface, from_cell, to_cell, pygame, color=(200,200,0), width=2):
    fx = from_cell[0] * CELL_SIZE + CELL_SIZE // 2
    fy = from_cell[1] * CELL_SIZE + SCORE_PANEL_HEIGHT + CELL_SIZE // 2
    tx = to_cell[0] * CELL_SIZE + CELL_SIZE // 2
    ty = to_cell[1] * CELL_SIZE + SCORE_PANEL_HEIGHT + CELL_SIZE // 2
    pygame.draw.line(surface, color, (fx, fy), (tx, ty), width)
    angle = math.atan2(ty - fy, tx - fx)
    head_length = CELL_SIZE // 4
    left = (tx - head_length * math.cos(angle - math.pi/6), ty - head_length * math.sin(angle - math.pi/6))
    right = (tx - head_length * math.cos(angle + math.pi/6), ty - head_length * math.sin(angle + math.pi/6))
    pygame.draw.polygon(surface, color, [(tx, ty), left, right])

def draw_path_arrows(surface, path, pygame, color=(255,255,0)):
    if not path or len(path) < 2:
        return
    for a,b in zip(path[:-1], path[1:]):
        draw_arrow(surface, a, b, pygame, color=color, width=2)

def draw_head_direction(surface, head_pos, direction, pygame, color=(255,120,0)):
    hx = head_pos[0] * CELL_SIZE + CELL_SIZE // 2
    hy = head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT + CELL_SIZE // 2
    dx = direction[0]
    dy = direction[1]
    tx = hx + dx * (CELL_SIZE // 2)
    ty = hy + dy * (CELL_SIZE // 2)
    pygame.draw.line(surface, color, (hx, hy), (tx, ty), 3)
    angle = math.atan2(ty - hy, tx - hx)
    head_length = CELL_SIZE // 5
    left = (tx - head_length * math.cos(angle - math.pi/6), ty - head_length * math.sin(angle - math.pi/6))
    right = (tx - head_length * math.cos(angle + math.pi/6), ty - head_length * math.sin(angle + math.pi/6))
    pygame.draw.polygon(surface, color, [(tx, ty), left, right])

# --- WORKER (single game instance) ---
def run_worker(instance_id=0, timeout=30, show_path=True):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Snake Worker {instance_id}")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 28)
    font_game_over = pygame.font.Font(None, 56)

    snake = Snake()
    apple = Apple(snake.body)

    running = True
    game_over = False
    victory = False

    start_time = time.time()
    move_counter = 0

    path = None

    while running:
        now = time.time()
        if timeout is not None and (now - start_time) >= timeout:
            # Timeout reached; stop the game and exit soon
            running = False
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and game_over:
                if event.key == pygame.K_SPACE:
                    return

        if not game_over and not victory:
            head = tuple(snake.head_pos)
            apple_pos = tuple(apple.position)
            occupied = set(tuple(seg) for seg in snake.body)
            tail = tuple(snake.body[-1])
            if not snake.grow_pending and tail in occupied:
                occupied.remove(tail)

            path = astar(head, apple_pos, occupied)
            chosen_path = None
            if path and len(path) >= 2:
                simulated_body = path_simulate_survivable(snake.body, path, apple_will_be_eaten=(path[-1] == apple_pos))
                sim_occupied = set(simulated_body)
                tail_after = simulated_body[-1]
                reachable = bfs_reachable(simulated_body[0], {tail_after}, set(sim_occupied) - {tail_after})
                if reachable:
                    chosen_path = path

            if not chosen_path:
                tail_path = find_path_to_tail(snake.body)
                if tail_path and len(tail_path) >= 2:
                    chosen_path = tail_path

            if chosen_path and len(chosen_path) >= 2:
                nd = next_direction_from_step(head, chosen_path[1])
                if nd:
                    snake.set_direction(nd)
            else:
                safe = find_safe_direction(head, occupied)
                if safe:
                    snake.set_direction(safe)

            move_counter += 1
            if move_counter >= GAME_SPEED // 10:
                snake.move()
                move_counter = 0
                if snake.is_game_over():
                    game_over = True
                if snake.head_pos == list(apple.position):
                    snake.grow()
                    if not apple.relocate(snake.body):
                        victory = True
                        game_over = True

        # Drawing
        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        draw_grid(screen, pygame)
        apple.draw(screen, pygame)
        snake.draw(screen, pygame)
        if show_path:
            try:
                draw_path_arrows(screen, path, pygame)
            except Exception:
                pass
        draw_head_direction(screen, tuple(snake.head_pos), snake.direction, pygame)
        display_info(screen, font_main, snake, start_time, pygame)
        if game_over:
            if victory:
                display_message(screen, font_game_over, "VICTOIRE !", VERT, pygame)
                display_message(screen, font_main, "ESPACE pour rejouer.", BLANC, pygame, y_offset=100)
            else:
                display_message(screen, font_game_over, "GAME OVER", ROUGE, pygame)
                display_message(screen, font_main, "ESPACE pour rejouer.", BLANC, pygame, y_offset=100)

        pygame.display.flip()
        clock.tick(LOCAL_FPS)

    # On exit, print result line for master to collect
    finish_ts = time.time()
    duration = int(finish_ts - start_time)
    # Print: <id> <score> <finish_ts> <duration>
    print(f"{instance_id} {snake.score} {finish_ts:.3f} {duration}", flush=True)
    pygame.quit()

# --- MASTER (launcher and scoreboard) ---
def run_master(n=10, timeout=30, show_path=True):
    script = os.path.abspath(__file__)
    procs = []
    results = {}
    # Launch workers
    for i in range(n):
        p = subprocess.Popen([sys.executable, script, '--worker', '--id', str(i), '--timeout', str(timeout), '--show-path', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append((i, p))
        time.sleep(0.05)

    # Collect outputs
    for i, p in procs:
        out, err = p.communicate()
        if err:
            print(f"Instance {i} stderr:\n{err}")
        if out:
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    # Expect: id score finish_ts duration
                    if len(parts) >= 4:
                        idx = int(parts[0])
                        score = int(parts[1])
                        finish_ts = float(parts[2])
                        duration = int(float(parts[3]))
                        results[idx] = (score, finish_ts, duration)
                        ft = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finish_ts))
                        print(f"Instance {idx} finished with score {score} at {ft} (duration {duration}s)")
                    else:
                        # Backwards-compatible: just id score
                        idx = int(parts[0])
                        score = int(parts[1])
                        results[idx] = (score, None, None)
                        print(f"Instance {idx} finished with score {score}")
                except Exception as e:
                    print(f"Couldn't parse line from instance {i}: {line} ({e})")

    # Fill missing instances with (score, finish_ts, duration)
    for i in range(n):
        results.setdefault(i, (0, None, None))

    # Show scoreboard in a pygame window
    try:
        import pygame
        pygame.init()
        w = 600
        h = 100 + 30 * n
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption('Serpent Multi - Results')
        font = pygame.font.Font(None, 32)
        clock = pygame.time.Clock()
        running = True

        # Determine best by score
        best_idx = max(results, key=lambda k: results[k][0])
        best_score = results[best_idx][0]
        start = time.time()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    running = False
            screen.fill(GRIS_FOND)
            title = font.render(f"Results (best: Instance {best_idx} score {best_score})", True, BLANC)
            screen.blit(title, (10, 10))
            y = 50
            for i in sorted(results):
                score, finish_ts, duration = results[i]
                if finish_ts:
                    ft = time.strftime('%H:%M:%S', time.localtime(finish_ts))
                    text = f"Instance {i}: score={score} finished={ft} dur={duration}s"
                else:
                    text = f"Instance {i}: score={score}"
                color = (180, 255, 180) if i == best_idx else BLANC
                surf = font.render(text, True, color)
                screen.blit(surf, (10, y))
                y += 28
            pygame.display.flip()
            clock.tick(30)
            # Auto-close scoreboard after 20s
            if time.time() - start > 20:
                running = False
        pygame.quit()
    except Exception as e:
        print('Could not open scoreboard window:', e)
        print('Results:')
        for i in sorted(results):
            print(f"Instance {i}: {results[i]}")

# --- ARGPARSE / MODE SWITCH ---
def parse_args(argv):
    mode = 'master'
    params = {'n': 10, 'timeout': 30, 'show_path': True}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == '--worker':
            mode = 'worker'
        elif a == '--id' and i + 1 < len(argv):
            params['id'] = int(argv[i+1]); i += 1
        elif a == '--timeout' and i + 1 < len(argv):
            params['timeout'] = int(argv[i+1]); i += 1
        elif a == '--show-path' and i + 1 < len(argv):
            params['show_path'] = bool(int(argv[i+1])); i += 1
        else:
            # master positional arg: n
            try:
                params['n'] = int(a)
            except Exception:
                pass
        i += 1
    return mode, params

if __name__ == '__main__':
    mode, params = parse_args(sys.argv)
    if mode == 'worker':
        instance_id = params.get('id', 0)
        timeout = params.get('timeout', 30)
        show_path = params.get('show_path', True)
        run_worker(instance_id, timeout, show_path)
    else:
        # Enforce 10 workers and 2 minutes timeout regardless of CLI args per request
        ENFORCED_WORKERS = 10
        ENFORCED_TIMEOUT = 120  # seconds (2 minutes)
        run_master(n=ENFORCED_WORKERS, timeout=ENFORCED_TIMEOUT, show_path=params.get('show_path', True))
