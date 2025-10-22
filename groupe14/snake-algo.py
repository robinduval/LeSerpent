import pygame
import random
import time
from queue import PriorityQueue
from collections import deque

# --- CONSTANTES DE JEU ---
# Taille de la grille (20x20)
GRID_SIZE = 15
# Taille d'une cellule en pixels
CELL_SIZE = 30
# Vitesse de jeu (images par seconde)
GAME_SPEED = 100  # Vitesse très élevée pour maximiser le ratio temps/completion

# Dimensions de l'écran (avec espace pour le score/timer)
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (en RGB)
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0) # Tête du serpent
VERT = (0, 200, 0)    # Corps du serpent
ROUGE = (200, 0, 0)   # Pomme
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
ALL_DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

class Snake:
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos,
                    [self.head_pos[0] - 1, self.head_pos[1]],
                    [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0
        
    def set_direction(self, new_dir):
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
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

class SnakeAI:
    def __init__(self, snake, apple):
        self.snake = snake
        self.apple = apple
        self.path = []
        self.virtual_snake = None
        self.last_tail = None
        
    def manhattan_distance(self, pos1, pos2):
        """Calcule la distance de Manhattan entre deux points"""
        x1, y1 = pos1 if isinstance(pos1, tuple) else tuple(pos1)
        x2, y2 = pos2 if isinstance(pos2, tuple) else tuple(pos2)
        return abs(x1 - x2) + abs(y1 - y2)
    
    def is_position_safe(self, pos, virtual_body=None):
        """Vérifie si une position est sûre en tenant compte du corps virtuel du serpent"""
        body_to_check = virtual_body if virtual_body else self.snake.body[:-1]
        return list(pos) not in body_to_check
    
    def simulate_move(self, pos, virtual_body):
        """Simule un mouvement du serpent et retourne le nouveau corps"""
        new_body = [list(pos)] + virtual_body[:-1]
        return new_body
    
    def get_safe_neighbors(self, pos, virtual_body=None):
        """Retourne les voisins sûrs d'une position"""
        neighbors = []
        body_to_check = virtual_body if virtual_body else self.snake.body
        
        for dx, dy in ALL_DIRECTIONS:
            new_x = (pos[0] + dx) % GRID_SIZE
            new_y = (pos[1] + dy) % GRID_SIZE
            new_pos = (new_x, new_y)
            
            # Simule le mouvement pour vérifier s'il est sûr
            if self.is_position_safe(new_pos, body_to_check):
                # Vérifie si ce mouvement ne nous piège pas
                new_virtual_body = self.simulate_move(new_pos, body_to_check)
                if self.has_escape_path(new_pos, new_virtual_body):
                    neighbors.append(new_pos)
        
        return neighbors
        
    def has_escape_path(self, pos, virtual_body):
        """Vérifie rapidement si une position a une sortie de secours"""
        for dx, dy in ALL_DIRECTIONS:
            new_x = (pos[0] + dx) % GRID_SIZE
            new_y = (pos[1] + dy) % GRID_SIZE
            new_pos = (new_x, new_y)
            if self.is_position_safe(new_pos, virtual_body):
                return True
        return False
        
    def find_fastest_safe_direction(self):
        """Trouve la direction la plus rapide qui soit minimalement sûre"""
        apple_pos = self.apple.position
        if not apple_pos:
            return self.find_safest_direction()
            
        best_direction = None
        min_distance = float('inf')
        
        for direction in ALL_DIRECTIONS:
            new_x = (self.snake.head_pos[0] + direction[0]) % GRID_SIZE
            new_y = (self.snake.head_pos[1] + direction[1]) % GRID_SIZE
            new_pos = [new_x, new_y]
            
            if new_pos not in self.snake.body[:-1]:  # Évite le corps
                distance = self.manhattan_distance(new_pos, apple_pos)
                if distance < min_distance:
                    virtual_body = self.simulate_move(new_pos, self.snake.body)
                    if self.has_escape_path(new_pos, virtual_body):  # Vérification minimale
                        min_distance = distance
                        best_direction = direction
                        
        return best_direction if best_direction else self.find_safest_direction()

    def can_reach_tail(self, pos, virtual_body):
        """Vérifie si on peut atteindre la queue depuis une position donnée"""
        visited = set()
        queue = deque([(pos, virtual_body)])
        tail = tuple(virtual_body[-1])
        
        while queue:
            current, body = queue.popleft()
            if current == tail:
                return True
                
            if tuple(current) in visited:
                continue
                
            visited.add(tuple(current))
            
            for next_pos in self.get_safe_neighbors(current, body):
                new_body = self.simulate_move(next_pos, body)
                queue.append((next_pos, new_body))
                
        return False
        
    def find_path_to_target(self, target, virtual_body=None):
        """Trouve un chemin rapide vers une cible avec vérification minimale de sécurité"""
        start = tuple(self.snake.head_pos)
        goal = tuple(target)
        
        if not goal:  # Si pas de cible valide
            return []
            
        # Clone le corps du serpent pour la simulation si non fourni
        if virtual_body is None:
            virtual_body = [list(pos) for pos in self.snake.body]
        
        frontier = PriorityQueue()
        frontier.put((0, start, virtual_body))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while not frontier.empty():
            current_cost, current, current_body = frontier.get()
            
            if current == goal:
                # Vérification de sécurité minimale
                goal_body = self.simulate_move(goal, current_body)
                
                # Vérifie uniquement si on ne se coince pas
                if self.has_escape_path(goal, goal_body):
                    break
                continue  # Si le chemin n'est pas sûr, continue la recherche
                
            for next_pos in self.get_safe_neighbors(current, current_body):
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    # Simule le nouveau corps du serpent après ce mouvement
                    new_body = self.simulate_move(next_pos, current_body)
                    priority = new_cost + self.manhattan_distance(goal, next_pos)
                    frontier.put((priority, next_pos, new_body))
                    came_from[next_pos] = current
        
        # Reconstruit le chemin
        if goal in came_from:
            path = []
            current = goal
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        return []

    def find_path_to_tail(self):
        """Trouve un chemin sûr vers la queue du serpent."""
        return self.find_path_to_target(self.snake.body[-1])

    def count_free_spaces(self, pos, virtual_body):
        """Compte le nombre d'espaces libres accessibles depuis une position."""
        visited = set()
        queue = deque([tuple(pos)])
        count = 0
        
        while queue:
            current = queue.popleft()
            
            if current in visited:
                continue
                
            visited.add(current)
            count += 1
            
            for dx, dy in ALL_DIRECTIONS:
                new_x = (current[0] + dx) % GRID_SIZE
                new_y = (current[1] + dy) % GRID_SIZE
                new_pos = (new_x, new_y)
                
                if new_pos not in visited and list(new_pos) not in virtual_body:
                    queue.append(new_pos)
        
        return count

    def find_safest_direction(self):
        """Trouve la direction la plus sûre en évaluant chaque option."""
        best_direction = None
        max_free_spaces = -1
        
        for direction in ALL_DIRECTIONS:
            new_x = (self.snake.head_pos[0] + direction[0]) % GRID_SIZE
            new_y = (self.snake.head_pos[1] + direction[1]) % GRID_SIZE
            new_pos = [new_x, new_y]
            
            if new_pos not in self.snake.body[:-1]:
                # Simule le mouvement
                virtual_body = self.simulate_move(new_pos, self.snake.body)
                
                # Compte les espaces libres accessibles depuis cette position
                free_spaces = self.count_free_spaces(new_pos, virtual_body)
                
                if free_spaces > max_free_spaces:
                    max_free_spaces = free_spaces
                    best_direction = direction
        
        return best_direction if best_direction else self.follow_tail()
    
    def follow_tail(self):
        """Suit la queue du serpent de près avec une sécurité maximale."""
        tail = self.snake.body[-1]
        best_direction = None
        max_safety_score = -1

        for direction in ALL_DIRECTIONS:
            new_x = (self.snake.head_pos[0] + direction[0]) % GRID_SIZE
            new_y = (self.snake.head_pos[1] + direction[1]) % GRID_SIZE
            new_pos = [new_x, new_y]
            
            # Vérifie si la position est libre
            if new_pos not in self.snake.body[:-1]:
                virtual_body = self.simulate_move(new_pos, self.snake.body)
                
                # Calcule un score de sécurité basé sur plusieurs facteurs
                safety_score = 0
                
                # 1. Distance à la queue
                tail_distance = self.manhattan_distance(new_pos, tail)
                safety_score += (GRID_SIZE * 2 - tail_distance) * 2  # Plus proche = meilleur
                
                # 2. Nombre d'espaces libres
                free_spaces = self.count_free_spaces(new_pos, virtual_body)
                safety_score += free_spaces * 3
                
                # 3. Nombre de sorties disponibles
                escape_count = sum(1 for d in ALL_DIRECTIONS if self.is_position_safe(
                    ((new_x + d[0]) % GRID_SIZE, (new_y + d[1]) % GRID_SIZE),
                    virtual_body
                ))
                safety_score += escape_count * 10
                
                # 4. Bonus si on peut atteindre la queue
                if self.can_reach_tail(new_pos, virtual_body):
                    safety_score += 50
                
                if safety_score > max_safety_score:
                    max_safety_score = safety_score
                    best_direction = direction
        
        return best_direction if best_direction else self.snake.direction
        
        return self.snake.direction  # Dernier recours : continue tout droit
    
    def get_next_move(self):
        # Sauvegarde l'état actuel du serpent
        self.last_tail = tuple(self.snake.body[-1])
        
        # Met à jour le chemin si nécessaire
        if not self.path:
            # Vérifie rapidement si on est dans une situation critique
            current_space = self.count_free_spaces(self.snake.head_pos, self.snake.body)
            
            # Seuil de sécurité réduit pour plus de rapidité
            if current_space < len(self.snake.body) * 1.2:  # Seuil réduit de 1.5 à 1.2
                # Mode survie rapide : va vers l'espace le plus grand
                safe_dir = self.find_safest_direction()
                if safe_dir:
                    return safe_dir
                self.path = self.find_path_to_tail()
            else:
                # Mode agressif : va directement vers la pomme si possible
                self.path = self.find_path_to_target(self.apple.position)
            
            # Si pas de chemin trouvé
            if not self.path:
                # Vérification de sécurité minimale
                safe_dir = self.find_fastest_safe_direction()
                if safe_dir:
                    return safe_dir
                
                # Si aucune direction rapide n'est sûre, suit la queue
                return self.follow_tail()
        
        # Suit le chemin calculé
        if len(self.path) > 1:
            next_pos = self.path[1]
            self.path = self.path[1:]
            
            # Vérifie une dernière fois que le mouvement est sûr
            virtual_body = self.simulate_move(next_pos, self.snake.body)
            if not self.is_position_safe(next_pos, self.snake.body[:-1]) or \
               not self.has_escape_path(next_pos, virtual_body):
                self.path = []  # Reset le chemin si le mouvement n'est pas sûr
                return self.get_next_move()  # Recalcule un nouveau mouvement
            
            # Calcule la direction à prendre
            dx = (next_pos[0] - self.snake.head_pos[0]) % GRID_SIZE
            dy = (next_pos[1] - self.snake.head_pos[1]) % GRID_SIZE
            
            if dx == 1 or (dx == -(GRID_SIZE-1)):
                return RIGHT
            elif dx == -1 or (dx == (GRID_SIZE-1)):
                return LEFT
            elif dy == 1 or (dy == -(GRID_SIZE-1)):
                return DOWN
            elif dy == -1 or (dy == (GRID_SIZE-1)):
                return UP
        
        return self.find_safest_direction()

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
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)

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

def main():
    pygame.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake IA - Algorithme A*")
    clock = pygame.time.Clock()
    
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)
    
    snake = Snake()
    apple = Apple(snake.body)
    snake_ai = SnakeAI(snake, apple)
    
    running = True
    game_over = False
    victory = False
    
    start_time = time.time()
    move_counter = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN and game_over:
                if event.key == pygame.K_SPACE:
                    main()
                    return
        
        if not game_over and not victory:
            move_counter += 1
            if move_counter >= GAME_SPEED // 10:
                # Obtient la prochaine direction de l'IA
                next_direction = snake_ai.get_next_move()
                snake.set_direction(next_direction)
                snake.move()
                move_counter = 0

                if snake.is_game_over():
                    game_over = True
                    continue

                if snake.head_pos == list(apple.position):
                    snake.grow()
                    if not apple.relocate(snake.body):
                        victory = True
                        game_over = True
                    else:
                        snake_ai.path = []  # Reset le chemin pour recalculer vers la nouvelle pomme
        
        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        
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
