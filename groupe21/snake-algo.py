import pygame
import random
import time
from collections import deque
import heapq

# --- CONSTANTES DE JEU ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 20  # Doubled from 10

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
CYAN = (0, 255, 255)
JAUNE = (255, 255, 0)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# --- CLASSES DU JEU ---

class Snake:
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos,
                     [self.head_pos[0] - 1, self.head_pos[1]],
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.grow_pending = False
        self.score = 0

    def move_to(self, new_head):
        self.body.insert(0, list(new_head))
        self.head_pos = list(new_head)
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

    def draw(self, surface):
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE,
                               segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)
        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE,
                                self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                                CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)

class Apple:
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available = [p for p in all_positions if list(p) not in occupied_positions]
        return random.choice(available) if available else None

    def relocate(self, snake_body):
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False

    def draw(self, surface):
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE,
                               self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            pygame.draw.circle(surface, BLANC,
                               (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3),
                               CELL_SIZE // 8)

# --- HAMILTONIAN CYCLE & AI ---

def neighbors(pos):
    """Return valid neighboring positions in the grid."""
    x, y = pos
    possible = [
        ((x + 1) % GRID_SIZE, y),
        ((x - 1) % GRID_SIZE, y),
        (x, (y + 1) % GRID_SIZE),
        (x, (y - 1) % GRID_SIZE)
    ]
    return possible

def manhattan(a, b):
    """Calculate Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def generate_hamiltonian_cycle():
    """Edge-first serpentine Hamiltonian cycle."""
    cycle = {}
    distances = {}  # Store distances from start
    current_dist = 0
    
    # Generate cycle and distances
    for y in range(GRID_SIZE):
        if y % 2 == 0:
            for x in range(GRID_SIZE):
                pos = (x, y)
                distances[pos] = current_dist
                current_dist += 1
                if x < GRID_SIZE - 1:
                    cycle[pos] = (x + 1, y)
                else:
                    if y < GRID_SIZE - 1:
                        cycle[pos] = (x, y + 1)
                    else:
                        cycle[pos] = (0, 0)
        else:
            for x in reversed(range(GRID_SIZE)):
                pos = (x, y)
                distances[pos] = current_dist
                current_dist += 1
                if x > 0:
                    cycle[pos] = (x - 1, y)
                else:
                    if y < GRID_SIZE - 1:
                        cycle[pos] = (x, y + 1)
                    else:
                        cycle[pos] = (0, 0)
    return cycle, distances

HAM_CYCLE, HAM_DISTANCES = generate_hamiltonian_cycle()

def get_relative_distance(pos1, pos2):
    """Get relative distance between positions in Hamiltonian cycle"""
    d1 = HAM_DISTANCES[pos1]
    d2 = HAM_DISTANCES[pos2]
    total_cells = GRID_SIZE * GRID_SIZE
    forward_dist = (d2 - d1) % total_cells
    return forward_dist

def virtual_move_check(head, next_pos, snake_body, apple_pos):
    """Enhanced simulation of future moves with better safety"""
    virtual_body = [list(next_pos)] + snake_body[:-1]
    if list(next_pos) == list(apple_pos):  # If eating apple
        virtual_body.append(snake_body[-1])  # Keep tail
        
    # Check immediate collisions
    if list(next_pos) in snake_body[:-1]:
        return False
        
    # Check if we're moving into a tight space
    blocked_neighbors = sum(1 for n in neighbors(next_pos) 
                          if list(n) in virtual_body[:-1])
    if blocked_neighbors > 2:  # More strict neighbor check
        return False
        
    # Look ahead more moves when snake is longer
    look_ahead = 5 if len(snake_body) > GRID_SIZE * GRID_SIZE * 0.4 else 3
    
    # Simulate future moves
    virtual_head = next_pos
    seen_positions = set(tuple(p) for p in virtual_body)
    
    for _ in range(look_ahead):
        next_ham = HAM_CYCLE[virtual_head]
        if tuple(next_ham) in seen_positions:
            return False
        virtual_head = next_ham
        seen_positions.add(virtual_head)
    return True

def is_safe_shortcut(head, next_pos, snake_body, apple_pos):
    """Stricter safety check for shortcuts"""
    # More conservative shortcut threshold as snake grows
    max_length_for_shortcuts = GRID_SIZE * GRID_SIZE * 0.5  # Reduced from 0.6
    if len(snake_body) > max_length_for_shortcuts:
        return False
        
    # Basic collision checks
    if list(next_pos) in snake_body[:-1]:
        return False
    
    # Get distances in Hamiltonian cycle
    dist_to_apple = get_relative_distance(head, apple_pos)
    dist_shortcut_to_apple = get_relative_distance(next_pos, apple_pos)
    
    # Must get significantly closer to apple
    if dist_shortcut_to_apple >= dist_to_apple * 0.8:  # Must save at least 20% distance
        return False
    
    # Enhanced neighbor check
    blocked = set(tuple(p) for p in snake_body[:-1])
    neighbors_blocked = sum(1 for n in neighbors(next_pos) if n in blocked)
    if neighbors_blocked > 1:  # Stricter neighbor requirement
        return False
        
    # Virtual move simulation with enhanced checks
    if not virtual_move_check(head, next_pos, snake_body, apple_pos):
        return False
        
    # Verify next few moves in Hamiltonian cycle are safe
    current = next_pos
    seen = set(tuple(p) for p in snake_body[:-1])
    for _ in range(3):
        next_ham = HAM_CYCLE[current]
        if tuple(next_ham) in seen:
            return False
        current = next_ham
        seen.add(current)
        
    return True

def get_shortest_safe_move(head, apple_pos, snake_body):
    """Find shortest safe move towards apple"""
    possible_moves = list(neighbors(head))
    
    # Sort moves by Manhattan distance to apple
    possible_moves.sort(key=lambda pos: manhattan(pos, apple_pos))
    
    for next_pos in possible_moves:
        if is_safe_shortcut(head, next_pos, snake_body, apple_pos):
            return next_pos
            
    return None

def follow_hamiltonian(head, snake_body):
    """Enhanced Hamiltonian cycle following with better safety"""
    next_pos = HAM_CYCLE[head]
    blocked = set(tuple(p) for p in snake_body[:-1])
    
    # If next position is blocked, try to find safe escape
    if tuple(next_pos) in blocked:
        # Try each neighbor in order of Hamiltonian distance
        possible_moves = neighbors(head)
        possible_moves.sort(key=lambda pos: get_relative_distance(head, pos))
        
        for pos in possible_moves:
            if tuple(pos) not in blocked:
                # Verify we can get back to Hamiltonian cycle
                virtual_body = [list(pos)] + snake_body[:-1]
                next_ham = HAM_CYCLE[pos]
                if tuple(next_ham) not in set(tuple(p) for p in virtual_body):
                    return pos
                    
    return next_pos

def ai_next_move(snake, apple):
    head = tuple(snake.head_pos)
    apple_pos = tuple(apple.position)
    
    # Try to take a safe shortcut if snake isn't too long
    if len(snake.body) <= GRID_SIZE * GRID_SIZE * 0.5:  # More conservative threshold
        shortcut = get_shortest_safe_move(head, apple_pos, snake.body)
        if shortcut:
            # Triple-check safety with enhanced checks
            if (is_safe_shortcut(head, shortcut, snake.body, apple_pos) and 
                virtual_move_check(head, shortcut, snake.body, apple_pos)):
                ai_next_move.shortcut_path = [head, shortcut]
                return shortcut
    
    # Follow Hamiltonian cycle with enhanced safety
    ai_next_move.shortcut_path = None
    next_move = follow_hamiltonian(head, snake.body)
    
    # Final safety verification
    if list(next_move) in snake.body[:-1]:
        # Emergency: find ANY safe move
        for pos in neighbors(head):
            if list(pos) not in snake.body[:-1]:
                return pos
    
    return next_move

ai_next_move.shortcut_path = None

# --- AFFICHAGE ---

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
    fill_rate = (len(snake.body) / (GRID_SIZE * GRID_SIZE)) * 100
    fill_text = font.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
    surface.blit(fill_text, (SCREEN_WIDTH//2 - fill_text.get_width()//2, 20))

def display_message(surface, font, message, color=BLANC, y_offset=0):
    text_surface = font.render(message, True, color)
    center_y = (SCREEN_HEIGHT//2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, center_y))
    padding = 20
    bg_rect = rect.inflate(padding*2, padding*2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)

def draw_paths(surface, snake):
    # Draw Hamiltonian path (light green)
    for cell, next_cell in HAM_CYCLE.items():
        rect = pygame.Rect(cell[0]*CELL_SIZE, cell[1]*CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, CYAN, rect, 1)
    # Draw shortcut path (yellow)
    if ai_next_move.shortcut_path:
        for cell in ai_next_move.shortcut_path:
            rect = pygame.Rect(cell[0]*CELL_SIZE, cell[1]*CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, JAUNE, rect, 2)

# --- BOUCLE PRINCIPALE ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake AI - DHCR Hybrid")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    snake = Snake()
    apple = Apple(snake.body)
    running, game_over, victory = True, False, False
    start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and game_over:
                if event.key == pygame.K_SPACE:
                    main()
                    return

        if not game_over:
            next_cell = ai_next_move(snake, apple)
            snake.move_to(next_cell)

            if snake.head_pos == list(apple.position):
                snake.grow()
                if not apple.relocate(snake.body):
                    victory = True
                    game_over = True

            if snake.is_game_over():
                game_over = True

        # Dessin
        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        draw_grid(screen)
        draw_paths(screen, snake)
        apple.draw(screen)
        snake.draw(screen)
        display_info(screen, font_main, snake, start_time)

        if game_over:
            if victory:
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
            else:
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
            display_message(screen, font_main, "ESPACE pour rejouer", BLANC, y_offset=100)

        pygame.display.flip()
        clock.tick(40)  # Doubled from 20

    pygame.quit()

if __name__ == '__main__':
    main()
