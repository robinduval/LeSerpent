import pygame
import random
import time
import heapq
import argparse

# --- CONSTANTES DE JEU (copiées de serpent.py) ---
GRID_SIZE = 30
CELL_SIZE = 30
GAME_SPEED = 5  # ticks per second
# Weight between Dijkstra (g) and Greedy Best-First (h).
# f = (1-weight)*g + weight*h
# Weight used in A* f = g + weight * h (heuristic multiplier)
HYBRID_WEIGHT = 1.0
# Show computed path visually
SHOW_PATH = True
# Minimum steps between recomputations (helps reduce repeated searches)
RECOMPUTE_COOLDOWN = 2
# Show computed path visually
SHOW_PATH = True

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


class Snake:
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos[:], [self.head_pos[0] - 1, self.head_pos[1]], [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        if (new_dir[0] * -1, new_dir[1] * -1) != tuple(self.direction):
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

    def check_wall_collision(self):
        x, y = self.head_pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

    def check_self_collision(self):
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        return self.check_wall_collision() or self.check_self_collision()

    def draw(self, surface):
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

    def draw(self, surface):
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            pygame.draw.circle(surface, BLANC, (rect.x + int(CELL_SIZE * 0.7), rect.y + int(CELL_SIZE * 0.3)), CELL_SIZE // 8)


def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def display_info(surface, font, snake, start_time):
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


def display_message(surface, font, message, color=BLANC, y_offset=0):
    text_surface = font.render(message, True, color)
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)


# --- DIJKSTRA IMPLEMENTATION ---
def neighbors(pos):
    x, y = pos
    for dx, dy in (UP, DOWN, LEFT, RIGHT):
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            yield (nx, ny)


def dijkstra(start, goal, blocked):
    # blocked is a set of (x,y) positions that cannot be traversed
    frontier = []
    heapq.heappush(frontier, (0, tuple(start)))
    came_from = {tuple(start): None}
    cost_so_far = {tuple(start): 0}

    while frontier:
        cost, current = heapq.heappop(frontier)
        if current == tuple(goal):
            break
        for nb in neighbors(current):
            if nb in blocked:
                continue
            new_cost = cost_so_far[current] + 1
            if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                cost_so_far[nb] = new_cost
                priority = new_cost
                heapq.heappush(frontier, (priority, nb))
                came_from[nb] = current

    # reconstruct path
    if tuple(goal) not in came_from:
        return None
    path = []
    cur = tuple(goal)
    while cur != tuple(start):
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


def manhattan_coords_axial(a_x, a_y, b_x, b_y):
    return abs(a_x - b_x) + abs(a_y - b_y)


# Helpers to convert between (x,y) and index
def pos_to_idx(pos):
    return pos[1] * GRID_SIZE + pos[0]


def idx_to_pos(idx):
    return (idx % GRID_SIZE, idx // GRID_SIZE)


def get_blocked_bool(snake):
    """Return a list of booleans length GRID_SIZE*GRID_SIZE where True=blocked.
    Blocks all body except tail (allow tail to be used).
    """
    n = GRID_SIZE * GRID_SIZE
    blocked = [False] * n
    for segment in snake.body[:-1]:
        blocked[pos_to_idx(segment)] = True
    return blocked


def a_star(start, goal, blocked_bool, weight=HYBRID_WEIGHT):
    """A* using preallocated lists for gscore and came_from to reduce dict overhead.
    Returns list of (x,y) tuples or None.
    """
    start_idx = pos_to_idx(start)
    goal_idx = pos_to_idx(goal)
    if blocked_bool[goal_idx]:
        return None

    n = GRID_SIZE * GRID_SIZE
    INF = 10**9
    gscore = [INF] * n
    came_from = [-1] * n
    closed = [False] * n

    start_x, start_y = start
    goal_x, goal_y = goal
    h0 = manhattan_coords_axial(start_x, start_y, goal_x, goal_y)

    frontier = []
    # heap entries: (f, g, idx)
    gscore[start_idx] = 0
    heapq.heappush(frontier, (gscore[start_idx] + weight * h0, 0, start_idx))

    found = False
    while frontier:
        f, g, current = heapq.heappop(frontier)
        if closed[current]:
            continue
        if current == goal_idx:
            found = True
            break
        closed[current] = True

        cx, cy = idx_to_pos(current)
        for dx, dy in (UP, DOWN, LEFT, RIGHT):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            ni = ny * GRID_SIZE + nx
            if blocked_bool[ni] or closed[ni]:
                continue
            ng = g + 1
            if ng < gscore[ni]:
                gscore[ni] = ng
                came_from[ni] = current
                h = manhattan_coords_axial(nx, ny, goal_x, goal_y)
                nf = ng + weight * h
                heapq.heappush(frontier, (nf, ng, ni))

    if not found:
        return None

    path = []
    cur = goal_idx
    while cur != start_idx and cur != -1:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return [idx_to_pos(i) for i in path]


def get_blocked_set(snake):
    # block all snake body except the tail (because tail will move)
    blocked = set()
    for segment in snake.body[:-1]:
        blocked.add(tuple(segment))
    return blocked


def choose_move_from_path(snake, path):
    if not path:
        return None
    next_cell = path[0]
    dx = next_cell[0] - snake.head_pos[0]
    dy = next_cell[1] - snake.head_pos[1]
    # normalize to (-1,0,1)
    if dx > 1: dx = 1
    if dx < -1: dx = -1
    if dy > 1: dy = 1
    if dy < -1: dy = -1
    return (dx, dy)


def safe_moves(snake):
    moves = []
    for d in (UP, DOWN, LEFT, RIGHT):
        nx = (snake.head_pos[0] + d[0])
        ny = (snake.head_pos[1] + d[1])
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if [nx, ny] not in snake.body[:-1]:
                moves.append(d)
    return moves


def main():
    parser = argparse.ArgumentParser(description='Snake IA (A*/hybrid)')
    parser.add_argument('--no-path', action='store_true', help='Disable drawing the computed path')
    args = parser.parse_args()

    # apply CLI overrides (only path drawing toggle). GAME_SPEED is read from the file variable.
    global SHOW_PATH
    if args.no_path:
        SHOW_PATH = False

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake - Dijkstra IA")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    snake = Snake()
    apple = Apple(snake.body)

    running = True
    game_over = False
    victory = False
    start_time = time.time()

    # cached path to avoid recomputing every tick
    current_path = None
    current_target = tuple(apple.position) if apple.position else None
    # allow immediate recompute
    steps_since_recompute = RECOMPUTE_COOLDOWN
    # time-based movement: moves per second = GAME_SPEED
    last_move_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    SHOW_PATH = not SHOW_PATH

        if not game_over and not victory:
            now = time.time()
            # move when enough time passed so GAME_SPEED = moves per second
            if now - last_move_time >= 1.0 / max(1, GAME_SPEED):
                last_move_time = now
                # compute or reuse cached path
                blocked = get_blocked_set(snake)

                need_recompute = False
                if not current_path:
                    need_recompute = True
                if apple.position and tuple(apple.position) != current_target:
                    # apple moved -> recompute
                    need_recompute = True
                # if next step in path is now blocked, recompute
                if current_path:
                    next_cell = current_path[0] if len(current_path) > 0 else None
                    if next_cell and next_cell in blocked:
                        need_recompute = True

                # only recompute if we need to and cooldown passed
                if need_recompute and steps_since_recompute >= RECOMPUTE_COOLDOWN:
                    blocked_bool = get_blocked_bool(snake)
                    current_path = a_star(snake.head_pos, apple.position, blocked_bool, weight=HYBRID_WEIGHT)
                    current_target = tuple(apple.position) if apple.position else None
                    steps_since_recompute = 0

                path = current_path

                chosen = None
                if path and len(path) > 0:
                    mv = choose_move_from_path(snake, path)
                    if mv:
                        chosen = mv
                        # pop consumed step
                        current_path = current_path[1:]
                else:
                    # no path or empty path: try any safe move
                    safes = safe_moves(snake)
                    if safes:
                        chosen = random.choice(safes)

                if chosen:
                    snake.set_direction(chosen)

                snake.move()
                steps_since_recompute += 1

                if snake.is_game_over():
                    game_over = True
                    continue

                if snake.head_pos == list(apple.position):
                    snake.grow()
                    if not apple.relocate(snake.body):
                        victory = True
                        game_over = True

        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        draw_grid(screen)
        apple.draw(screen)
        # draw planned path for debugging/visualisation
        if SHOW_PATH and 'path' in locals() and path:
            for cell in path:
                rect = pygame.Rect(cell[0] * CELL_SIZE, cell[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (120, 120, 255), rect, 2)
        snake.draw(screen)
        display_info(screen, font_main, snake, start_time)

        if game_over:
            if victory:
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
                display_message(screen, font_main, "ESPACE pour rejouer.", BLANC, y_offset=100)
            else:
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
                display_message(screen, font_main, "ESPACE pour rejouer.", BLANC, y_offset=100)

        pygame.display.flip()
        clock.tick(GAME_SPEED)

    pygame.quit()


if __name__ == '__main__':
    main()
