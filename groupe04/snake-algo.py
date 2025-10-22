import pygame
import random
import time
import heapq
from collections import deque
from typing import List, Tuple, Optional, Set

# --- CONSTANTES DE JEU ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 10  # Plus rapide pour les algos

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)
BLEU = (0, 100, 255)  # Pour visualiser le chemin

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# --- CLASSES DU JEU ---

class Snake:
    """Représente le serpent, sa position, sa direction et son corps."""
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos, 
                     [self.head_pos[0] - 1, self.head_pos[1]], 
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        """Change la direction, empêchant le mouvement inverse immédiat."""
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        """Déplace le serpent d'une case dans la direction actuelle."""
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
        """Prépare le serpent à grandir au prochain mouvement."""
        self.grow_pending = True
        self.score += 1

    def check_self_collision(self):
        """Vérifie si la tête touche une partie du corps."""
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si le jeu est terminé."""
        return self.check_self_collision()

    def draw(self, surface):
        """Dessine le serpent sur la surface de jeu."""
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)

        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)


class Apple:
    """Représente la pomme (nourriture) et sa position."""
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aléatoire non occupée par le serpent."""
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(pos) not in occupied_positions]
        
        if not available_positions:
            return None
            
        return random.choice(available_positions)

    def relocate(self, snake_body):
        """Déplace la pomme vers une nouvelle position aléatoire."""
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False

    def draw(self, surface):
        """Dessine la pomme sur la surface de jeu."""
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)


# --- ALGORITHMES DE PATHFINDING ---

class PathfindingAlgorithm:
    """Classe de base pour les algorithmes de pathfinding."""
    
    @staticmethod
    def get_neighbors(pos: Tuple[int, int], snake_body: List[List[int]]) -> List[Tuple[int, int]]:
        """Retourne les voisins valides d'une position."""
        x, y = pos
        neighbors = []
        
        for dx, dy in DIRECTIONS:
            new_x = (x + dx) % GRID_SIZE
            new_y = (y + dy) % GRID_SIZE
            
            # Évite le corps du serpent (sauf la queue qui va bouger)
            if [new_x, new_y] not in snake_body[:-1]:
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    @staticmethod
    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calcule la distance de Manhattan entre deux positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    @staticmethod
    def reconstruct_path(came_from: dict, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruit le chemin depuis le dictionnaire came_from."""
        path = []
        current = goal
        
        while current != start:
            path.append(current)
            current = came_from.get(current)
            if current is None:
                return []
        
        path.reverse()
        return path


class AStar(PathfindingAlgorithm):
    """
    Algorithme A* pour trouver le chemin le plus court.
    Combine le coût réel (g) et l'heuristique (h) : f = g + h
    """
    
    @staticmethod
    def find_path(start: Tuple[int, int], goal: Tuple[int, int], snake_body: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Trouve le chemin optimal de start à goal en utilisant A*.
        """
        # File de priorité : (f_score, position)
        # f_score = g_score + heuristique
        h_start = PathfindingAlgorithm.manhattan_distance(start, goal)
        frontier = [(h_start, start)]
        came_from = {start: None}
        g_score = {start: 0}  # Coût réel depuis le départ
        
        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current == goal:
                return PathfindingAlgorithm.reconstruct_path(came_from, start, goal)
            
            for next_pos in PathfindingAlgorithm.get_neighbors(current, snake_body):
                # Coût réel pour atteindre next_pos
                tentative_g_score = g_score[current] + 1
                
                if next_pos not in g_score or tentative_g_score < g_score[next_pos]:
                    g_score[next_pos] = tentative_g_score
                    h_score = PathfindingAlgorithm.manhattan_distance(next_pos, goal)
                    f_score = tentative_g_score + h_score
                    heapq.heappush(frontier, (f_score, next_pos))
                    came_from[next_pos] = current
        
        return []  # Pas de chemin trouvé


class SnakeAI:
    """Intelligence artificielle pour contrôler le serpent avec A*."""
    
    def __init__(self):
        """
        Initialise l'IA avec l'algorithme A*.
        """
        self.current_path = []
        self.following_tail = False  # Indique si on suit la queue
        
    def get_direction_from_path(self, current_pos: List[int], next_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convertit une position cible en direction."""
        dx = (next_pos[0] - current_pos[0]) % GRID_SIZE
        dy = (next_pos[1] - current_pos[1]) % GRID_SIZE
        
        # Gère le wrap-around de la grille
        if dx > GRID_SIZE // 2:
            dx -= GRID_SIZE
        if dx < -GRID_SIZE // 2:
            dx += GRID_SIZE
        if dy > GRID_SIZE // 2:
            dy -= GRID_SIZE
        if dy < -GRID_SIZE // 2:
            dy += GRID_SIZE
        
        # Normalise la direction
        if abs(dx) > abs(dy):
            return (1 if dx > 0 else -1, 0)
        else:
            return (0, 1 if dy > 0 else -1)
    
    def get_next_move(self, snake: Snake, apple: Apple) -> Optional[Tuple[int, int]]:
        """
        Calcule le prochain mouvement pour le serpent.
        Stratégie :
        1. Cherche un chemin vers la pomme avec A*
        2. Si trouvé, vérifie qu'on peut encore atteindre la queue après
        3. Sinon, suit la queue pour rester en sécurité
        """
        start = tuple(snake.head_pos)
        goal = apple.position
        tail = tuple(snake.body[-1])
        
        # Recalcule le chemin si nécessaire
        if not self.current_path:
            # Essaie de trouver un chemin vers la pomme
            path_to_apple = AStar.find_path(start, goal, snake.body)
            
            if path_to_apple:
                # Simule la position après avoir mangé la pomme
                # Vérifie qu'on peut toujours atteindre la queue
                simulated_head = path_to_apple[-1] if path_to_apple else start
                simulated_body = [list(simulated_head)] + snake.body[:-1]
                
                # Cherche un chemin vers la queue depuis la position de la pomme
                path_to_tail = AStar.find_path(simulated_head, tail, simulated_body)
                
                if path_to_tail or len(snake.body) < 10:  # Si petit serpent, prend le risque
                    self.current_path = path_to_apple
                    self.following_tail = False
                else:
                    # Pas sûr, suit la queue à la place
                    self.current_path = AStar.find_path(start, tail, snake.body)
                    self.following_tail = True
            else:
                # Pas de chemin vers la pomme, suit la queue
                self.current_path = AStar.find_path(start, tail, snake.body)
                self.following_tail = True
        
        # Si un chemin existe, prend la première étape
        if self.current_path and len(self.current_path) > 0:
            next_pos = self.current_path[0]
            self.current_path.pop(0)
            return self.get_direction_from_path(snake.head_pos, next_pos)
        
        # Stratégie de secours : évite les collisions
        for direction in DIRECTIONS:
            new_x = (snake.head_pos[0] + direction[0]) % GRID_SIZE
            new_y = (snake.head_pos[1] + direction[1]) % GRID_SIZE
            if [new_x, new_y] not in snake.body[:-1]:
                return direction
        
        return snake.direction  # Continue dans la même direction


# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface):
    """Dessine la grille pour une meilleure visualisation."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def draw_path(surface, path: List[Tuple[int, int]]):
    """Dessine le chemin calculé par l'algorithme."""
    for pos in path:
        rect = pygame.Rect(pos[0] * CELL_SIZE + CELL_SIZE // 4, 
                          pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT + CELL_SIZE // 4, 
                          CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.rect(surface, BLEU, rect, border_radius=3)


def display_info(surface, font, snake, start_time, algorithm_name, ai):
    """Affiche le score, le temps, l'algorithme et le mode."""
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    # Score
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 10))

    # Temps
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_text = font.render(f"Temps: {minutes:02d}:{seconds:02d}", True, BLANC)
    surface.blit(time_text, (10, 35))
    
    # Mode actuel
    mode_color = ORANGE if ai.following_tail else VERT
    mode_text = "Suivi Queue" if ai.following_tail else "Vers Pomme"
    mode_display = font.render(f"Mode: {mode_text}", True, mode_color)
    surface.blit(mode_display, (10, 60))
    
    # Algorithme
    algo_text = font.render(f"Algo: {algorithm_name}", True, VERT)
    surface.blit(algo_text, (SCREEN_WIDTH - algo_text.get_width() - 10, 10))
    
    # Remplissage
    max_cells = GRID_SIZE * GRID_SIZE
    fill_rate = (len(snake.body) / max_cells) * 100
    fill_text = font.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
    surface.blit(fill_text, (SCREEN_WIDTH - fill_text.get_width() - 10, 35))
    
    # Longueur du serpent
    length_text = font.render(f"Longueur: {len(snake.body)}", True, BLANC)
    surface.blit(length_text, (SCREEN_WIDTH - length_text.get_width() - 10, 60))


def display_message(surface, font, message, color=BLANC, y_offset=0):
    """Affiche un message central sur l'écran."""
    text_surface = font.render(message, True, color)
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))
    
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)


# --- BOUCLE PRINCIPALE ---

def main():
    """Fonction principale pour exécuter le jeu Snake avec algorithmes."""
    pygame.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Algorithmique - A* avec suivi de queue")
    clock = pygame.time.Clock()
    
    font_main = pygame.font.Font(None, 30)
    font_game_over = pygame.font.Font(None, 80)
    
    # Initialisation
    snake = Snake()
    apple = Apple(snake.body)
    ai = SnakeAI()
    algorithm_name = "A*"
    
    print("="*60)
    print("🐍 Snake avec algorithme A* + Suivi de Queue")
    print("="*60)
    print("🎯 Stratégie intelligente :")
    print("  1. Cherche le chemin vers la pomme avec A*")
    print("  2. Vérifie qu'il peut encore atteindre sa queue")
    print("  3. Sinon, suit sa queue pour rester en sécurité")
    print("="*60)
    print("⏸️  ESPACE pour rejouer | Fermez la fenêtre pour quitter")
    print("="*60)
    
    running = True
    game_over = False
    victory = False
    start_time = time.time()
    
    # Boucle de jeu
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if game_over and event.key == pygame.K_SPACE:
                    main()
                    return
        
        # Logique du jeu
        if not game_over and not victory:
            # L'IA décide du mouvement
            next_direction = ai.get_next_move(snake, apple)
            if next_direction:
                snake.set_direction(next_direction)
            
            snake.move()
            
            # Vérification des collisions
            if snake.is_game_over():
                game_over = True
                continue
            
            # Vérification de la pomme
            if snake.head_pos == list(apple.position):
                snake.grow()
                if not apple.relocate(snake.body):
                    victory = True
                    game_over = True
        
        # Dessin
        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        
        draw_grid(screen)
        
        # Dessine le chemin prévu
        if ai.current_path:
            draw_path(screen, ai.current_path)
        
        apple.draw(screen)
        snake.draw(screen)
        
        display_info(screen, font_main, snake, start_time, algorithm_name, ai)
        
        # Messages de fin
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