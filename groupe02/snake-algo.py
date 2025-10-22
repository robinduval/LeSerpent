import pygame
import random
import time
import heapq

# --- CONSTANTES DE JEU ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 999
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
BLEU_CHEMIN = (100, 100, 255)
JAUNE = (255, 255, 0)
ROSE = (255, 100, 150)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


def heuristic(a, b):
    """Distance de Manhattan avec wrapping"""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    # Considérer le chemin le plus court en tenant compte du wrapping
    dx = min(dx, GRID_SIZE - dx)
    dy = min(dy, GRID_SIZE - dy)
    return dx + dy


def normalize_position(pos):
    """Normalise une position avec wrapping"""
    return (pos[0] % GRID_SIZE, pos[1] % GRID_SIZE)


def is_valid_position(pos):
    """Vérifie si une position est valide (elle l'est toujours avec wrapping)"""
    return True


def is_position_safe(pos, obstacles_set):
    """Vérifie rapidement si une position est sûre"""
    return normalize_position(pos) not in obstacles_set


def get_wrapped_distance(a, b):
    """Calcule la distance la plus courte avec wrapping"""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dx = min(dx, GRID_SIZE - dx)
    dy = min(dy, GRID_SIZE - dy)
    return dx + dy


def astar_pathfinding_optimized(start, goal, obstacles):
    """
    A* pathfinding OPTIMISÉ avec wrapping.
    - Utilise heapq pour la file de priorité (O(log n))
    - Utilise un set pour les obstacles (O(1) lookup)
    - Wrapping: la grille est un tore (cylindre)
    """
    start = normalize_position(start)
    goal = normalize_position(goal)

    if start == goal:
        return [start]

    obstacles_set = set(normalize_position(tuple(pos)) for pos in obstacles)

    counter = 0
    open_heap = [(heuristic(start, goal), counter, start)]
    counter += 1

    came_from = {}
    g_score = {start: 0}
    in_open_set = {start}
    closed_set = set()

    directions = [UP, DOWN, LEFT, RIGHT]

    while open_heap:
        current_f, _, current = heapq.heappop(open_heap)
        in_open_set.discard(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        closed_set.add(current)
        current_g = g_score[current]

        for direction in directions:
            neighbor = normalize_position((current[0] + direction[0], current[1] + direction[1]))

            if neighbor in closed_set:
                continue
            if not is_position_safe(neighbor, obstacles_set):
                continue

            tentative_g = current_g + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)

                if neighbor not in in_open_set:
                    heapq.heappush(open_heap, (f_score, counter, neighbor))
                    counter += 1
                    in_open_set.add(neighbor)

    return None


def bfs_find_longest_path(start, obstacles_set, max_depth=50):
    """
    BFS pour trouver le chemin le plus long (stratégie de survie) avec wrapping.
    """
    from collections import deque

    start = normalize_position(start)
    queue = deque([(start, 0)])
    visited = {start}
    farthest = start
    max_dist = 0

    directions = [UP, DOWN, LEFT, RIGHT]

    while queue and len(visited) < max_depth:
        current, dist = queue.popleft()

        if dist > max_dist:
            max_dist = dist
            farthest = current

        for direction in directions:
            neighbor = normalize_position((current[0] + direction[0], current[1] + direction[1]))

            if neighbor in visited:
                continue
            if neighbor in obstacles_set:
                continue

            visited.add(neighbor)
            queue.append((neighbor, dist + 1))

    return farthest


def is_apple_reachable_after_eating(apple_pos, snake_body, obstacles):
    """
    Vérifie si le serpent a au moins 2 chemins de sortie après avoir mangé la pomme.
    Fonctionne avec wrapping.
    """
    apple_pos = normalize_position(apple_pos)

    # Simulation: ajouter la pomme au corps du serpent
    new_body = [list(apple_pos)] + snake_body[:-1]
    new_obstacles = set(normalize_position(tuple(pos)) for pos in new_body[:-1])

    # Vérifier s'il y a au moins 2 directions de fuite
    head_pos = apple_pos
    directions = [UP, DOWN, LEFT, RIGHT]
    safe_directions = 0

    for direction in directions:
        next_pos = normalize_position((head_pos[0] + direction[0], head_pos[1] + direction[1]))
        if next_pos not in new_obstacles:
            safe_directions += 1

    return safe_directions >= 2


def get_direction_to_position(current_pos, target_pos):
    """Calcule la direction optimale avec wrapping"""
    current_pos = normalize_position(current_pos)
    target_pos = normalize_position(target_pos)

    # Calculer les distances avec wrapping
    dx_direct = target_pos[0] - current_pos[0]
    dy_direct = target_pos[1] - current_pos[1]

    # Prendre le chemin le plus court (peut être en passant par les bords)
    if abs(dx_direct) > GRID_SIZE / 2:
        dx_direct = dx_direct - GRID_SIZE if dx_direct > 0 else dx_direct + GRID_SIZE
    if abs(dy_direct) > GRID_SIZE / 2:
        dy_direct = dy_direct - GRID_SIZE if dy_direct > 0 else dy_direct + GRID_SIZE

    if abs(dx_direct) > abs(dy_direct):
        return (1 if dx_direct > 0 else -1, 0)
    else:
        return (0, 1 if dy_direct > 0 else -1)


class Snake:
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos[:],
                     [self.head_pos[0] - 1, self.head_pos[1]],
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.current_path = []
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0
        self.stuck_counter = 0
        self.safe_mode = False
        self.safe_mode_timer = 0

    def set_direction(self, new_dir):
        """Change la direction, empêchant le mouvement inverse immédiat"""
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def draw_path(self, surface):
        """Dessine le chemin bleu calculé par A*"""
        if self.current_path:
            for i, pos in enumerate(self.current_path):
                opacity = max(100, 255 - (i * 10))
                color = (100, 100, min(255, opacity))

                rect = pygame.Rect(pos[0] * CELL_SIZE + CELL_SIZE // 4,
                                   pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT + CELL_SIZE // 4,
                                   CELL_SIZE // 2, CELL_SIZE // 2)
                pygame.draw.rect(surface, color, rect, border_radius=3)

    def compute_next_move(self, apple_pos):
        """
        Calcule le prochain mouvement avec stratégie optimisée.
        Fonctionne avec wrapping.
        """
        start = normalize_position(self.head_pos)
        goal = normalize_position(apple_pos)
        obstacles = self.body[:-1]

        # Vérifier si manger la pomme serait dangereux
        apple_is_safe = is_apple_reachable_after_eating(goal, self.body, obstacles)

        if self.safe_mode:
            self.safe_mode_timer += 1
            if self.safe_mode_timer > 15:
                self.safe_mode = False
                self.safe_mode_timer = 0

        if apple_is_safe and not self.safe_mode:
            path = astar_pathfinding_optimized(start, goal, obstacles)

            if path and len(path) > 0:
                self.current_path = path
                self.stuck_counter = 0
                next_pos = path[0]
                new_direction = get_direction_to_position(self.head_pos, next_pos)
                self.set_direction(new_direction)
            else:
                self.current_path = []
                self._survival_mode(obstacles, start)
        else:
            self.safe_mode = True
            self.current_path = []
            self._survival_mode(obstacles, start)

    def _survival_mode(self, obstacles, start):
        """Mode survie: trouver l'espace le plus grand et s'y diriger (avec wrapping)"""
        obstacles_set = set(normalize_position(tuple(pos)) for pos in obstacles)
        farthest = bfs_find_longest_path(start, obstacles_set)

        if farthest != start:
            path = astar_pathfinding_optimized(start, farthest, obstacles)
            if path and len(path) > 0:
                next_pos = path[0]
                new_direction = get_direction_to_position(self.head_pos, next_pos)
                self.set_direction(new_direction)
                self.stuck_counter = 0
        else:
            self.stuck_counter += 1

    def move(self):
        """Déplace le serpent AVEC wrapping"""
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
        """Prépare le serpent à grandir au prochain mouvement"""
        self.grow_pending = True
        self.score += 1

    def check_wall_collision(self):
        """Pas de collision murale avec wrapping"""
        return False

    def check_self_collision(self):
        """Vérifie si la tête touche une partie du corps"""
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si le jeu est terminé"""
        return self.check_self_collision()

    def draw(self, surface):
        """Dessine le serpent sur la surface de jeu"""
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE,
                               segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)

        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE,
                                self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                                CELL_SIZE, CELL_SIZE)

        head_color = ROSE if self.safe_mode else ORANGE
        pygame.draw.rect(surface, head_color, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)


class Apple:
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aléatoire non occupée"""
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(pos) not in occupied_positions]
        if not available_positions:
            return None
        return random.choice(available_positions)

    def relocate(self, snake_body):
        """Déplace la pomme vers une nouvelle position"""
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False

    def draw(self, surface):
        """Dessine la pomme"""
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE,
                               self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            pygame.draw.circle(surface, BLANC,
                               (rect.x + int(CELL_SIZE * 0.7),
                                rect.y + int(CELL_SIZE * 0.3)),
                               CELL_SIZE // 8)


def draw_grid(surface):
    """Dessine la grille"""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def display_info(surface, font, snake, start_time):
    """Affiche le score, temps et mode"""
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

    mode_text = font.render("Mode: A* Safe" if snake.safe_mode else "Mode: A* Hunting", True,
                            JAUNE)
    surface.blit(mode_text, (10, 50))


def display_message(surface, font, message, color=BLANC, y_offset=0):
    """Affiche un message centré"""
    text_surface = font.render(message, True, color)
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))

    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)


def main():
    """Fonction principale - boucle de jeu"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake - A* Optimisé")
    clock = pygame.time.Clock()

    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    snake = Snake()
    apple = Apple(snake.body)

    running = True
    game_over = False
    victory = False
    start_time = time.time()
    move_counter = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if game_over and event.key == pygame.K_SPACE:
                    main()
                    return

        if not game_over and not victory:
            move_counter += 1
            if move_counter >= GAME_SPEED // 10:
                snake.compute_next_move(apple.position)
                snake.move()
                move_counter = 0

                if snake.is_game_over():
                    print(f"Game Over - Score final: {snake.score}")
                    game_over = True
                    continue

                if snake.head_pos == list(apple.position):
                    snake.grow()
                    if not apple.relocate(snake.body):
                        victory = True
                        game_over = True
                        print(f"VICTOIRE - Score final: {snake.score}")

        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)

        snake.draw_path(screen)
        draw_grid(screen)

        apple.draw(screen)
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
