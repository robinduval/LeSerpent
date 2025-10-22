import pygame
import random
import time
import heapq
from collections import deque

# --- CONSTANTES DE JEU ---
GRID_SIZE = 15          # 15x15 cases
CELL_SIZE = 30          # pixels par case
GAME_SPEED = 12         # FPS

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)  # Tête
VERT = (0, 200, 0)      # Corps
ROUGE = (200, 0, 0)     # Pomme
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

DIRS = [UP, DOWN, LEFT, RIGHT]

# -----------------------------
# Utilitaires de grille (TORUS)
# -----------------------------

def wrap(x, y):
    """Retourne la position torique (wrap autour des bords)."""
    return x % GRID_SIZE, y % GRID_SIZE

def neighbors_torus(pos):
    """Voisins en 4-connecté avec wrap (aucune sortie de grille)."""
    x, y = pos
    for dx, dy in DIRS:
        nx, ny = wrap(x + dx, y + dy)
        yield (nx, ny)

def dijkstra(start, goal, blocked):
    """Dijkstra sur une grille torique."""
    if start == goal:
        return [start]

    pq = [(0, start)]
    dist = {start: 0}
    prev = {start: None}

    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        if d != dist[u]:
            continue

        for v in neighbors_torus(u):
            if v in blocked:
                continue
            nd = d + 1
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal not in prev:
        return []

    # reconstruction
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

def flood_reachable(start, blocked):
    """Nombre de cases atteignables (BFS) depuis start en évitant blocked, sur un tore."""
    if start in blocked:
        return 0
    q = deque([start])
    seen = {start}
    while q:
        u = q.popleft()
        for v in neighbors_torus(u):
            if v in seen or v in blocked:
                continue
            seen.add(v)
            q.append(v)
    return len(seen)

# -----------------------------
# Entités du jeu
# -----------------------------

class Snake:
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos[:],
                     [self.head_pos[0] - 1, self.head_pos[1]],
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        # Normalize body with wrap (au cas où positions négatives)
        self.body = [[p[0] % GRID_SIZE, p[1] % GRID_SIZE] for p in self.body]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        new_head_x = self.head_pos[0] + self.direction[0]
        new_head_y = self.head_pos[1] + self.direction[1]
        new_head_x, new_head_y = wrap(new_head_x, new_head_y)
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
        # Sur un tore, jamais de collision murale
        return False

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
        occupied = set(map(tuple, occupied_positions))
        available = [p for p in all_positions if p not in occupied]
        if not available:
            return None
        return random.choice(available)

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
            pygame.draw.circle(surface, BLANC, (int(rect.x + CELL_SIZE * 0.7), int(rect.y + CELL_SIZE * 0.3)), CELL_SIZE // 8)

# -----------------------------
# Affichage
# -----------------------------

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

# -----------------------------
# IA améliorée (survie/score) avec TORUS
# -----------------------------

def simulate_path_follow(body_list, path, will_grow_on_last=False):
    body = [p[:] for p in body_list]
    for i in range(1, len(path)):  # skip première case (tête)
        nxt = [path[i][0], path[i][1]]
        body.insert(0, nxt)
        if i == len(path) - 1 and will_grow_on_last:
            pass
        else:
            body.pop()
    return body

def choose_direction_with_strategy(snake: Snake, apple: Apple):
    head = tuple(snake.head_pos)
    goal = tuple(apple.position)

    blocked_now = set(tuple(p) for p in snake.body[:-1])  # permettre la queue
    path_to_apple = dijkstra(head, goal, blocked_now)

    def path_to_tail_exist(sim_body):
        new_head = tuple(sim_body[0])
        tail = tuple(sim_body[-1])
        blocked = set(tuple(p) for p in sim_body[:-1])
        p = dijkstra(new_head, tail, blocked)
        return p

    if len(path_to_apple) >= 2:
        sim_body_after_apple = simulate_path_follow(snake.body, path_to_apple, will_grow_on_last=True)
        if path_to_tail_exist(sim_body_after_apple):
            nx, ny = path_to_apple[1]
            x, y = head
            step = ( (nx - x + GRID_SIZE) % GRID_SIZE, (ny - y + GRID_SIZE) % GRID_SIZE )
            # Convertir les steps modulaires en DIRS équivalents
            if step == (1, 0) or step == (-(GRID_SIZE-1), 0):
                return RIGHT
            if step == (-1 % GRID_SIZE, 0) or step == (GRID_SIZE-1, 0):
                return LEFT
            if step == (0, 1) or step == (0, -(GRID_SIZE-1)):
                return DOWN
            if step == (0, -1 % GRID_SIZE) or step == (0, GRID_SIZE-1):
                return UP

    # Aller vers la queue sinon
    tail = tuple(snake.body[-1])
    path_to_tail = dijkstra(head, tail, blocked_now)
    if len(path_to_tail) >= 2:
        nx, ny = path_to_tail[1]
        x, y = head
        step = ( (nx - x + GRID_SIZE) % GRID_SIZE, (ny - y + GRID_SIZE) % GRID_SIZE )
        if step == (1, 0) or step == (-(GRID_SIZE-1), 0):
            return RIGHT
        if step == (GRID_SIZE-1, 0):
            return LEFT
        if step == (0, 1) or step == (0, -(GRID_SIZE-1)):
            return DOWN
        if step == (0, GRID_SIZE-1):
            return UP

    # Coup de secours: maximiser l'espace accessible après mouvement avec wrap
    best_dir = None
    best_space = -1
    for d in DIRS:
        nx, ny = wrap(snake.head_pos[0] + d[0], snake.head_pos[1] + d[1])
        if [nx, ny] in snake.body[:-1]:
            continue
        blocked_future = set(tuple(p) for p in snake.body[:-1])
        space = flood_reachable((nx, ny), blocked_future)
        if space > best_space:
            best_space = space
            best_dir = d

    if best_dir is not None:
        return best_dir

    return snake.direction

# -----------------------------
# Boucle principale
# -----------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake - Dijkstra (Tore)")
    clock = pygame.time.Clock()

    font_main = pygame.font.Font(None, 40)
    font_big = pygame.font.Font(None, 80)

    snake = Snake()
    apple = Apple(snake.body)

    running = True
    game_over = False
    victory = False
    start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if game_over and event.key == pygame.K_SPACE:
                    main()
                    return

        if not game_over and not victory:
            new_dir = choose_direction_with_strategy(snake, apple)
            snake.set_direction(new_dir)
            snake.move()

            if snake.is_game_over():
                game_over = True

            if not game_over and snake.head_pos == list(apple.position):
                snake.grow()
                if not apple.relocate(snake.body):
                    victory = True
                    game_over = True

        # Rendu
        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)

        draw_grid(screen)
        apple.draw(screen)
        snake.draw(screen)
        display_info(screen, font_main, snake, start_time)

        if game_over:
            if victory:
                display_message(screen, font_big, "VICTOIRE !", VERT)
                display_message(screen, font_main, "ESPACE pour rejouer.", BLANC, y_offset=100)
            else:
                display_message(screen, font_big, "GAME OVER", ROUGE)
                display_message(screen, font_main, "ESPACE pour rejouer.", BLANC, y_offset=100)

        pygame.display.flip()
        clock.tick(GAME_SPEED)

    pygame.quit()

if __name__ == "__main__":
    main()