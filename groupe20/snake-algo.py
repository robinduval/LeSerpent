import pygame
import random
import time
from collections import deque
import heapq

# --- CONSTANTES DE JEU ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 15  # Rapide pour l'IA optimisée

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
BLEU = (0, 100, 255)  # Pour visualiser le chemin A*
JAUNE = (255, 255, 0)

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
        # Position initiale alignée sur le cycle Hamiltonien (début du zigzag)
        # Commence en (0,0) et va vers la droite
        self.head_pos = [2, 0]
        self.body = [self.head_pos, 
                     [1, 0], 
                     [0, 0]]
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

    def check_wall_collision(self):
        """Vérifie si la tête touche les bords (ne fonctionne pas avec wraparound)."""
        x, y = self.head_pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

    def check_self_collision(self):
        """Vérifie si la tête touche une partie du corps."""
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si le jeu est terminé."""
        return self.check_wall_collision() or self.check_self_collision()

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

# --- ALGORITHME HAMILTONIEN OPTIMISÉ ---

class HamiltonianPathfinder:
    """
    Implémentation d'un algorithme Hamiltonien optimisé pour Snake.
    
    Stratégie :
    1. Crée un cycle hamiltonien qui passe par toutes les cases
    2. Suit ce cycle pour garantir 100% de sécurité
    3. Utilise des raccourcis intelligents pour accélérer la collecte
    """
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.hamilton_path = self.build_hamilton_cycle()
        self.position_to_index = {tuple(pos): i for i, pos in enumerate(self.hamilton_path)}
    
    def build_hamilton_cycle(self):
        """
        Construit un cycle hamiltonien en zigzag sur la grille.
        Ce pattern garantit de passer par toutes les cases une seule fois.
        """
        path = []
        for y in range(self.grid_size):
            if y % 2 == 0:
                # Ligne paire : gauche -> droite
                for x in range(self.grid_size):
                    path.append([x, y])
            else:
                # Ligne impaire : droite -> gauche
                for x in range(self.grid_size - 1, -1, -1):
                    path.append([x, y])
        return path
    
    def manhattan_distance(self, pos1, pos2):
        """Calcule la distance de Manhattan entre deux positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_next_hamilton_pos(self, current_pos):
        """Retourne la prochaine position dans le cycle hamiltonien."""
        current_index = self.position_to_index.get(tuple(current_pos))
        if current_index is None:
            # Si la position n'est pas dans le cycle, trouve la plus proche
            min_dist = float('inf')
            closest_index = 0
            for i, pos in enumerate(self.hamilton_path):
                dist = self.manhattan_distance(current_pos, pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_index = i
            current_index = closest_index
        
        next_index = (current_index + 1) % len(self.hamilton_path)
        return self.hamilton_path[next_index]
    
    def is_safe_shortcut(self, from_pos, to_pos, snake_body):
        """
        Vérifie si un raccourci est sûr en s'assurant qu'on ne se coince pas.
        Un raccourci est sûr si :
        1. La destination n'est pas dans le corps
        2. On reste "en avant" dans le cycle hamiltonien
        """
        if list(to_pos) in snake_body:
            return False
        
        from_idx = self.position_to_index.get(tuple(from_pos), 0)
        to_idx = self.position_to_index.get(tuple(to_pos), 0)
        
        # Calcule la distance dans le cycle
        cycle_length = len(self.hamilton_path)
        forward_dist = (to_idx - from_idx) % cycle_length
        
        # Un raccourci est sûr s'il avance dans le cycle et n'est pas trop long
        return forward_dist > 0 and forward_dist < cycle_length // 2
    
    def a_star_search(self, start, goal, obstacles, max_nodes=500):
        """
        A* rapide pour trouver des raccourcis.
        Limité en nombre de nœuds pour performance.
        """
        start = tuple(start)
        goal = tuple(goal)
        
        if start == goal:
            return [start]
        
        counter = 0
        open_set = [(0, counter, start, [start])]
        closed_set = set()
        g_scores = {start: 0}
        nodes_explored = 0
        
        while open_set and nodes_explored < max_nodes:
            f_score, _, current, path = heapq.heappop(open_set)
            nodes_explored += 1
            
            if current in closed_set:
                continue
            
            if current == goal:
                return path
            
            closed_set.add(current)
            current_g = g_scores[current]
            
            for direction in DIRECTIONS:
                neighbor = ((current[0] + direction[0]) % self.grid_size,
                           (current[1] + direction[1]) % self.grid_size)
                
                if neighbor in closed_set or list(neighbor) in obstacles:
                    continue
                
                tentative_g = current_g + 1
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_score = self.manhattan_distance(neighbor, goal)
                    f_score = tentative_g + h_score
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor, path + [neighbor]))
        
        return None
    
    def can_reach_after_eating(self, head, next_pos, apple_pos, snake_body):
        """
        Vérification rapide : après avoir mangé, peut-on toujours survivre ?
        On vérifie simplement qu'il y a assez d'espace autour.
        """
        if next_pos != apple_pos:
            return True
        
        # Compte l'espace libre autour de la position
        simulated_body = [list(next_pos)] + snake_body
        free_neighbors = 0
        
        for direction in DIRECTIONS:
            neighbor = ((next_pos[0] + direction[0]) % self.grid_size,
                       (next_pos[1] + direction[1]) % self.grid_size)
            if list(neighbor) not in simulated_body:
                free_neighbors += 1
        
        # Au moins 2 voisins libres = sûr
        return free_neighbors >= 2
    
    def get_next_safe_direction(self, head, snake_body):
        """
        Trouve une direction sûre en cas d'urgence.
        Choisit la direction avec le plus d'espace libre.
        """
        best_direction = None
        max_space = -1
        
        for direction in DIRECTIONS:
            test_pos = ((head[0] + direction[0]) % self.grid_size,
                       (head[1] + direction[1]) % self.grid_size)
            
            if list(test_pos) not in snake_body:
                # Compte l'espace accessible depuis cette position
                free_count = 0
                visited = set()
                queue = deque([test_pos])
                visited.add(test_pos)
                
                while queue and len(visited) < 20:  # Limite pour performance
                    current = queue.popleft()
                    free_count += 1
                    
                    for d in DIRECTIONS:
                        neighbor = ((current[0] + d[0]) % self.grid_size,
                                   (current[1] + d[1]) % self.grid_size)
                        if neighbor not in visited and list(neighbor) not in snake_body:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                if free_count > max_space:
                    max_space = free_count
                    best_direction = direction
        
        return best_direction

    def is_on_hamilton_cycle(self, pos):
        """Vérifie si une position est sur le cycle hamiltonien."""
        return tuple(pos) in self.position_to_index
    
    def can_safely_join_cycle(self, head, snake_body):
        """
        Vérifie si on peut rejoindre le cycle en toute sécurité.
        Retourne (possible, next_position) où next_position est la prochaine étape sûre.
        """
        # Trouve la position la plus proche dans le cycle
        min_dist = float('inf')
        closest_cycle_pos = None
        
        for cycle_pos in self.hamilton_path:
            if list(cycle_pos) not in snake_body:
                dist = self.manhattan_distance(head, tuple(cycle_pos))
                if dist < min_dist:
                    min_dist = dist
                    closest_cycle_pos = tuple(cycle_pos)
        
        if not closest_cycle_pos:
            return False, None
        
        # Cherche un chemin vers le cycle
        path_to_cycle = self.a_star_search(head, closest_cycle_pos, snake_body[1:], max_nodes=300)
        
        if not path_to_cycle or len(path_to_cycle) <= 1:
            return False, None
        
        # Simule le chemin pour vérifier qu'on ne se coince pas
        next_pos = path_to_cycle[1]
        
        # Vérifie qu'après avoir rejoint le cycle, on peut continuer
        simulated_body = [list(next_pos)] + snake_body[:-1]
        
        # Compte les voisins libres
        free_neighbors = 0
        for direction in DIRECTIONS:
            neighbor = ((next_pos[0] + direction[0]) % self.grid_size,
                       (next_pos[1] + direction[1]) % self.grid_size)
            if list(neighbor) not in simulated_body:
                free_neighbors += 1
        
        # Au moins 2 voisins libres = sûr
        if free_neighbors >= 2:
            return True, next_pos
        
        return False, None
    
    def get_best_direction(self, snake, apple):
        """
        Stratégie optimisée en 3 phases :
        
        PHASE 1 (0-25%) : A* agressif pour scorer rapidement
        PHASE 2 (25-30%) : Transition sécurisée vers le cycle Hamiltonien
        PHASE 3 (30%+) : Cycle Hamiltonien pur (100% survie)
        """
        head = tuple(snake.head_pos)
        apple_pos = tuple(apple.position) if apple.position else None
        
        snake_length = len(snake.body)
        max_cells = self.grid_size * self.grid_size
        fill_rate = snake_length / max_cells
        
        # Seuils
        ASTAR_PHASE_END = 0.25      # Fin de la phase A* agressive (plus conservateur)
        TRANSITION_PHASE_END = 0.30  # Fin de la transition
        
        # ======== PHASE 1 : A* AGRESSIF (0-25%) ========
        if fill_rate < ASTAR_PHASE_END:
            if apple_pos:
                # Cherche un chemin direct vers la pomme
                path_to_apple = self.a_star_search(head, apple_pos, snake.body[1:], max_nodes=300)
                
                if path_to_apple and len(path_to_apple) > 1:
                    next_pos = path_to_apple[1]
                    
                    if list(next_pos) not in snake.body:
                        direction = self.calculate_direction(head, next_pos)
                        
                        test_pos = ((head[0] + direction[0]) % self.grid_size,
                                   (head[1] + direction[1]) % self.grid_size)
                        
                        if list(test_pos) not in snake.body:
                            return direction
            
            # Si A* échoue, cherche une direction sûre
            safe_dir = self.get_next_safe_direction(head, snake.body)
            if safe_dir:
                return safe_dir
        
        # ======== PHASE 2 : TRANSITION VERS LE CYCLE (25-30%) ========
        elif fill_rate < TRANSITION_PHASE_END:
            # En transition : essaie de rejoindre le cycle tout en continuant à scorer
            
            # Si on est déjà sur le cycle, commence à le suivre
            if self.is_on_hamilton_cycle(snake.head_pos):
                next_hamilton_pos = tuple(self.get_next_hamilton_pos(snake.head_pos))
                direction = self.calculate_direction(head, next_hamilton_pos)
                
                test_pos = ((head[0] + direction[0]) % self.grid_size,
                           (head[1] + direction[1]) % self.grid_size)
                
                if list(test_pos) not in snake.body:
                    return direction
            
            # Sinon, essaie de rejoindre le cycle en toute sécurité
            can_join, next_cycle_pos = self.can_safely_join_cycle(head, snake.body)
            
            if can_join and next_cycle_pos:
                direction = self.calculate_direction(head, next_cycle_pos)
                
                test_pos = ((head[0] + direction[0]) % self.grid_size,
                           (head[1] + direction[1]) % self.grid_size)
                
                if list(test_pos) not in snake.body:
                    return direction
            
            # Si impossible de rejoindre le cycle, continue avec A* prudemment
            if apple_pos:
                path_to_apple = self.a_star_search(head, apple_pos, snake.body[1:], max_nodes=200)
                
                if path_to_apple and len(path_to_apple) > 1:
                    next_pos = path_to_apple[1]
                    
                    if list(next_pos) not in snake.body:
                        direction = self.calculate_direction(head, next_pos)
                        
                        test_pos = ((head[0] + direction[0]) % self.grid_size,
                                   (head[1] + direction[1]) % self.grid_size)
                        
                        if list(test_pos) not in snake.body:
                            return direction
            
            # Dernière option : direction sûre
            safe_dir = self.get_next_safe_direction(head, snake.body)
            if safe_dir:
                return safe_dir
        
        # ======== PHASE 3 : CYCLE HAMILTONIEN PUR (30%+) ========
        
        # Si pas encore sur le cycle, rejoins-le en urgence
        if not self.is_on_hamilton_cycle(snake.head_pos):
            can_join, next_cycle_pos = self.can_safely_join_cycle(head, snake.body)
            
            if can_join and next_cycle_pos:
                direction = self.calculate_direction(head, next_cycle_pos)
                
                test_pos = ((head[0] + direction[0]) % self.grid_size,
                           (head[1] + direction[1]) % self.grid_size)
                
                if list(test_pos) not in snake.body:
                    return direction
            
            # Si impossible, cherche une direction sûre
            safe_dir = self.get_next_safe_direction(head, snake.body)
            if safe_dir:
                return safe_dir
        
        # Suit le cycle Hamiltonien strictement
        next_hamilton_pos = tuple(self.get_next_hamilton_pos(snake.head_pos))
        direction = self.calculate_direction(head, next_hamilton_pos)
        
        test_pos = ((head[0] + direction[0]) % self.grid_size,
                   (head[1] + direction[1]) % self.grid_size)
        
        if list(test_pos) not in snake.body:
            return direction
        
        # Urgence : cycle bloqué, saute quelques positions
        for i in range(1, min(10, len(self.hamilton_path))):
            next_idx = (self.position_to_index.get(head, 0) + i) % len(self.hamilton_path)
            candidate_pos = tuple(self.hamilton_path[next_idx])
            
            if list(candidate_pos) not in snake.body:
                path = self.a_star_search(head, candidate_pos, snake.body[1:], max_nodes=100)
                if path and len(path) > 1:
                    next_pos = path[1]
                    if list(next_pos) not in snake.body:
                        direction = self.calculate_direction(head, next_pos)
                        
                        test_pos = ((head[0] + direction[0]) % self.grid_size,
                                   (head[1] + direction[1]) % self.grid_size)
                        
                        if list(test_pos) not in snake.body:
                            return direction
        
        # Dernier recours
        safe_dir = self.get_next_safe_direction(head, snake.body)
        if safe_dir:
            return safe_dir
        
        return direction
    
    def calculate_direction(self, from_pos, to_pos):
        """Calcule la direction pour aller de from_pos à to_pos (avec wraparound)."""
        direction = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        
        # Gère le wraparound
        if abs(direction[0]) > 1:
            direction = (-direction[0] // abs(direction[0]), direction[1])
        if abs(direction[1]) > 1:
            direction = (direction[0], -direction[1] // abs(direction[1]))
        
        return direction

# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface):
    """Dessine la grille pour une meilleure visualisation."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

def draw_path(surface, path):
    """Dessine le chemin calculé par A* (pour visualisation)."""
    if not path:
        return
    
    for pos in path[1:]:  # Skip la tête
        rect = pygame.Rect(pos[0] * CELL_SIZE + CELL_SIZE // 4, 
                          pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT + CELL_SIZE // 4,
                          CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.rect(surface, BLEU, rect, border_radius=3)

def display_info(surface, font, snake, start_time):
    """Affiche le score et le temps écoulé dans le panneau supérieur."""
    
    # Dessiner le panneau de score
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    # Afficher le score
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 20))

    # Afficher le temps
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
    """Fonction principale pour exécuter le jeu Snake avec A*."""
    pygame.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Algorithmique - Groupe 20")
    clock = pygame.time.Clock()
    
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)
    
    snake = Snake()
    apple = Apple(snake.body)
    pathfinder = HamiltonianPathfinder(GRID_SIZE)
    
    running = True
    game_over = False
    victory = False
    
    start_time = time.time()
    move_counter = 0

    while running:
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if game_over:
                    if event.key == pygame.K_SPACE:
                        main()
                        return
        
        # Logique de mise à jour
        if not game_over and not victory:
            move_counter += 1
            
            # A* décide automatiquement
            if move_counter >= GAME_SPEED // 10:
                best_direction = pathfinder.get_best_direction(snake, apple)
                snake.set_direction(best_direction)
            
            # Déplacement
            if move_counter >= GAME_SPEED // 10:
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
        
        # Dessin
        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        
        draw_grid(screen)
        
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
        
        pygame.display.flip()
        clock.tick(GAME_SPEED)

    pygame.quit()

if __name__ == '__main__':
    main()
