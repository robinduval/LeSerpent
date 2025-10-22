"""
Version algorithmique du jeu Snake utilisant l'algorithme de Dijkstra
Groupe 21
"""

import pygame
import random
import time
from queue import PriorityQueue
import math

# --- CONSTANTES DE JEU ---
NOIR = (0, 0, 0)
BLANC = (255, 255, 255)
ROUGE = (255, 0, 0)
VERT = (0, 255, 0)
GRIS_FOND = (40, 40, 40)

# Taille de la grille (20x20)
GRID_SIZE = 15
# Taille d'une cellule en pixels
CELL_SIZE = 30
# Vitesse de jeu (images par seconde)
GAME_SPEED = 5

# Dimensions de l'écran (avec espace pour le score/timer)
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (en RGB)
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)  # Tête du serpent
VERT = (0, 200, 0)     # Corps du serpent
ROUGE = (200, 0, 0)    # Pomme
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions possibles
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        """Initialise le serpent avec une position de départ et une direction."""
        self.body = [[GRID_SIZE // 4, GRID_SIZE // 2]]
        self.direction = RIGHT
        self.head_pos = self.body[0]
        self.grow_pending = False
    
    def set_direction(self, new_direction):
        """Change la direction du serpent si ce n'est pas un demi-tour."""
        opposite = (-new_direction[0], -new_direction[1])
        if len(self.body) == 1 or new_direction != opposite:
            self.direction = new_direction
    
    def move(self):
        """Déplace le serpent d'une case dans sa direction actuelle."""
        # Calcule la nouvelle position de la tête
        new_head = [
            (self.head_pos[0] + self.direction[0]) % GRID_SIZE,
            (self.head_pos[1] + self.direction[1]) % GRID_SIZE
        ]
        
        # Ajoute la nouvelle tête au début du corps
        self.body.insert(0, new_head)
        self.head_pos = new_head
        
        # Si pas de croissance en attente, retire la queue
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False
    
    def grow(self):
        """Fait grandir le serpent à son prochain mouvement."""
        self.grow_pending = True
    
    def is_game_over(self):
        """Vérifie si le serpent est entré en collision avec lui-même."""
        return self.head_pos in self.body[1:]
    
    def draw(self, surface):
        """Dessine le serpent sur la surface de jeu."""
        # Dessine le corps
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)  # Bordure
        
        # Dessine la tête
        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 1)  # Bordure

class Apple:
    def __init__(self, snake_body):
        """Initialise la pomme avec une position aléatoire valide."""
        self.position = self._get_random_position(snake_body)
    
    def _get_random_position(self, snake_body):
        """Trouve une position aléatoire non occupée par le serpent."""
        all_positions = [
            (x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
            if [x, y] not in snake_body
        ]
        if all_positions:
            return random.choice(all_positions)
        return None
    
    def relocate(self, snake_body):
        """Déplace la pomme à une nouvelle position aléatoire."""
        new_position = self._get_random_position(snake_body)
        if new_position:
            self.position = new_position
            return True
        return False
    
    def draw(self, surface):
        """Dessine la pomme sur la surface donnée."""
        if self.position:
            rect = pygame.Rect(
                self.position[0] * CELL_SIZE,
                self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(surface, ROUGE, rect)

class PathFinder:
    """Utilise l'algorithme de Dijkstra pour trouver le chemin le plus court vers la pomme."""
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.directions = [UP, DOWN, LEFT, RIGHT]
        # Pour la visualisation
        self.explored_nodes = set()
        self.current_path = []
    
    def get_neighbors(self, pos, snake_body):
        """Retourne les voisins valides d'une position."""
        neighbors = []
        x, y = pos
        
        for dx, dy in self.directions:
            new_x = (x + dx) % self.grid_size
            new_y = (y + dy) % self.grid_size
            new_pos = [new_x, new_y]
            
            # Vérifie si la nouvelle position n'est pas dans le corps du serpent
            if new_pos not in snake_body[:-1]:  # Exclut la queue qui va bouger
                neighbors.append((new_pos, 1))  # Le coût est toujours 1 pour des mouvements simples
                
        return neighbors
    
    def find_path(self, start, goal, snake_body):
        """Trouve le chemin le plus court entre start et goal en évitant le corps du serpent."""
        start = tuple(start)
        goal = tuple(goal)
        
        # Réinitialise les données de visualisation
        self.explored_nodes = set()
        self.current_path = []
        
        # Initialisation des structures pour Dijkstra
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while not frontier.empty():
            current = frontier.get()[1]
            
            if current == goal:
                break
                
            for next_pos, step_cost in self.get_neighbors(current, snake_body):
                next_pos = tuple(next_pos)
                new_cost = cost_so_far[current] + step_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    frontier.put((priority, next_pos))
                    came_from[next_pos] = current
                    # Ajoute le nœud aux nœuds explorés pour la visualisation
                    self.explored_nodes.add(next_pos)
        
        # Reconstruction du chemin
        path = []
        current = goal
        if goal in came_from:  # Si un chemin a été trouvé
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            self.current_path = path
            return path
        self.current_path = []
        return None
    
    def heuristic(self, a, b):
        """Calcule une estimation de la distance entre deux points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_next_move(self, snake, apple):
        """Retourne la prochaine direction à prendre pour atteindre la pomme."""
        path = self.find_path(snake.head_pos, list(apple.position), snake.body)
        if not path or len(path) < 2:
            return None
        
        # Calcule la direction à partir des deux premiers points du chemin
        next_pos = path[1]
        dx = (next_pos[0] - snake.head_pos[0]) % self.grid_size
        dy = (next_pos[1] - snake.head_pos[1]) % self.grid_size
        
        # Ajuste les valeurs pour gérer le passage par les bords
        if dx > 1:
            dx = -1
        elif dx < -1:
            dx = 1
        if dy > 1:
            dy = -1
        elif dy < -1:
            dy = 1
            
        return (dx, dy)

# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface, pathfinder=None):
    """Dessine la grille de jeu et visualise l'algorithme de pathfinding."""
    LIGHT_BLUE = (135, 206, 250)  # Couleur pour les cellules visitées
    DEEP_BLUE = (0, 0, 139)      # Couleur pour le chemin prévu
    
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(
                x * CELL_SIZE,
                y * CELL_SIZE + SCORE_PANEL_HEIGHT,
                CELL_SIZE,
                CELL_SIZE
            )
            # Toujours dessiner un fond noir pour chaque cellule
            pygame.draw.rect(surface, NOIR, rect)
            
            if pathfinder:
                # Cellules visitées avec bordure bleu clair
                if (x, y) in pathfinder.explored_nodes:
                    pygame.draw.rect(surface, LIGHT_BLUE, rect, 1)
                
                # Chemin prévu avec bordure bleu foncé
                if (x, y) in [tuple(pos) for pos in pathfinder.current_path]:
                    pygame.draw.rect(surface, DEEP_BLUE, rect, 2)  # Bordure plus épaisse pour le chemin
            else:
                # Grille normale quand le pathfinder n'est pas actif
                pygame.draw.rect(surface, GRIS_GRILLE, rect, 1)

def display_info(surface, font, snake, start_time):
    """Affiche les informations de jeu (score, temps, etc.)."""
    # Afficher le score
    score_text = font.render(f"Score: {len(snake.body)}", True, BLANC)
    surface.blit(score_text, (10, 20))
    
    # Afficher le temps écoulé
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_text = font.render(f"Temps: {minutes:02d}:{seconds:02d}", True, BLANC)
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 20))
    
    # Afficher le taux de remplissage
    max_cells = GRID_SIZE * GRID_SIZE
    fill_rate = (len(snake.body) / max_cells) * 100
    fill_text = font.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
    surface.blit(fill_text, (SCREEN_WIDTH // 2 - fill_text.get_width() // 2, 20))

def display_message(surface, font, message, color=BLANC, y_offset=0):
    """
    Affiche un message central sur l'écran, avec un décalage vertical optionnel.
    y_offset permet de positionner plusieurs messages.
    """
    text_surface = font.render(message, True, color)
    # Applique le décalage vertical au centre
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))
    
    # Dessine un fond semi-transparent pour la lisibilité
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)

def main():
    """Fonction principale pour exécuter le jeu Snake avec l'algorithme de Dijkstra."""
    pygame.init()
    
    # Configuration de l'écran
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake - Dijkstra - Groupe 21")
    clock = pygame.time.Clock()
    
    # Configuration des polices
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)
    
    # Initialisation des objets du jeu
    snake = Snake()
    apple = Apple(snake.body)
    pathfinder = PathFinder(GRID_SIZE)
    
    # Variables de jeu
    running = True
    game_over = False
    victory = False
    ai_active = True  # L'IA est active par défaut
    
    # Démarrage du chronomètre
    start_time = time.time()
    
    # Variable pour la gestion de la vitesse (pour ne bouger qu'une fois par tic)
    move_counter = 0

    # --- Boucle de jeu ---
    while running:
        # 1. Gestion des Événements (Contrôles Clavier)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if game_over:
                    if event.key == pygame.K_SPACE:
                        main()  # Redémarre le jeu
                        return
                else:
                    # Active/désactive l'IA avec la touche 'A'
                    if event.key == pygame.K_a:
                        ai_active = not ai_active
        
        # 2. Logique de Mise à Jour du Jeu
        if not game_over and not victory:
            move_counter += 1
            if move_counter >= GAME_SPEED // 10:  # Déplace le serpent à un rythme constant
                if ai_active:
                    # Utilise Dijkstra pour trouver le prochain mouvement
                    next_direction = pathfinder.get_next_move(snake, apple)
                    if next_direction:
                        snake.set_direction(next_direction)
                
                snake.move()
                move_counter = 0

                # Vérification des collisions (murs et corps)
                if snake.is_game_over():
                    game_over = True
                    continue

                # Vérification de la pomme mangée
                if snake.head_pos == list(apple.position):
                    snake.grow()
                    
                    # Tente de replacer la pomme, vérifie la Victoire si échec
                    if not apple.relocate(snake.body):
                        victory = True
                        game_over = True
        
        # 3. Dessin
        screen.fill(GRIS_FOND)
        
        # Zone de jeu (décalée par la hauteur du panneau de score)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        
        # Dessine la grille avec la visualisation du pathfinding
        draw_grid(screen, pathfinder if ai_active else None)
        
        # Dessine la pomme et le serpent
        apple.draw(screen)
        snake.draw(screen)
        
        # Affiche le score et le temps
        display_info(screen, font_main, snake, start_time)
        
        # Affichage des messages de fin de jeu
        if game_over:
            if victory:
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
                message_details = "ESPACE pour rejouer."
                display_message(screen, font_main, message_details, BLANC, y_offset=100)
            else:
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
                message_details = "ESPACE pour rejouer."
                display_message(screen, font_main, message_details, BLANC, y_offset=100)
        

        # Mise à jour de l'affichage
        pygame.display.flip()
        
        # Contrôle la vitesse du jeu
        clock.tick(GAME_SPEED)

    pygame.quit()

if __name__ == "__main__":
    main()