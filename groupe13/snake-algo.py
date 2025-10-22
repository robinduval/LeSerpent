import heapq
import random
import time
from collections import deque

import pygame

# --- CONSTANTES DE JEU ---
# Taille de la grille (20x20)
GRID_SIZE = 15
# Taille d'une cellule en pixels
CELL_SIZE = 30
# Vitesse de jeu (images par seconde)
GAME_SPEED = 120

# Dimensions de l'écran (avec espace pour le score/timer)
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (en RGB)
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)  # Tête du serpent
VERT = (0, 200, 0)    # Corps du serpent
ROUGE = (200, 0, 0)   # Pomme
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# --- CLASSES DU JEU ---


class Snake:
    """Représente le serpent, sa position, sa direction et son corps."""

    def __init__(self):
        # Position initiale au centre
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        # Le corps est une liste de positions (x, y), incluant la tête
        self.body = [self.head_pos,
                     [self.head_pos[0] - 1, self.head_pos[1]],
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        """Change la direction, empêchant le mouvement inverse immédiat."""
        # Vérifie que la nouvelle direction n'est pas l'inverse de l'actuelle
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        """Déplace le serpent d'une case dans la direction actuelle."""
        # Calcul de la nouvelle position de la tête
        new_head_x = (self.head_pos[0] + self.direction[0]) % GRID_SIZE
        new_head_y = (self.head_pos[1] + self.direction[1]) % GRID_SIZE

        # Mettre à jour la tête (la nouvelle position devient la nouvelle tête)
        new_head_pos = [new_head_x, new_head_y]
        self.body.insert(0, new_head_pos)
        self.head_pos = new_head_pos

        # Si le serpent ne doit pas grandir, supprime la queue (mouvement normal)
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False  # Réinitialise le drapeau

    def grow(self):
        """Prépare le serpent à grandir au prochain mouvement."""
        self.grow_pending = True
        self.score += 1

    def check_wall_collision(self):
        """Vérifie si la tête touche les bords (Game Over si hors grille)."""
        # Note: Le jeu utilise le wrap-around (% GRID_SIZE dans move()),
        # donc cette fonction ne détecte jamais de collision de mur.
        # C'est un choix de design pour ce jeu.
        x, y = self.head_pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

    def check_self_collision(self):
        """Vérifie si la tête touche une partie du corps (Game Over si auto-morsure)."""
        # On vérifie si la position de la tête est dans le reste du corps (body[1:])
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si le jeu est terminé (mur ou morsure)."""
        return self.check_wall_collision() or self.check_self_collision()

    def draw(self, surface):
        """Dessine le serpent sur la surface de jeu."""
        # Dessine le corps
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1]
                               * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)  # Bordure

        # Dessine la tête (couleur différente)
        head_rect = pygame.Rect(
            self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)  # Bordure plus épaisse


class Apple:
    """Représente la pomme (nourriture) et sa position."""

    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aléatoire non occupée par le serpent."""
        all_positions = [(x, y) for x in range(GRID_SIZE)
                         for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(
            pos) not in occupied_positions]

        if not available_positions:
            # Toutes les cases sont pleines (condition de Victoire)
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
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1]
                               * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            # Ajout d'un petit reflet pour un aspect "pomme"
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE *
                               0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)

# --- ALGORITHME DE PATHFINDING ---


class SnakeAI:
    """Algo A*"""

    def __init__(self):
        self.path = []
        self.BLUE = (100, 150, 255)  # Couleur pour visualiser le chemin

    def heuristic(self, pos1, pos2):
        """Distance de Manhattan entre deux positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, pos):
        """Voisins où le serpent peut se déplacer à partir d'une position"""
        x, y = pos
        neighbors = []
        for direction in [UP, DOWN, LEFT, RIGHT]:
            new_x = (x + direction[0]) % GRID_SIZE
            new_y = (y + direction[1]) % GRID_SIZE
            neighbors.append((new_x, new_y))
        return neighbors

    def is_safe_position(self, pos, snake_body, allow_tail=True):
        """Vérifie si la position n'est pas occupée par le corps du serpent."""
        pos_list = list(pos)
        # On peut occuper la position de la queue car elle va bouger
        if allow_tail and pos_list == snake_body[-1]:
            return True
        # Vérifie que la position n'est pas dans le corps
        return pos_list not in snake_body

    def a_star(self, start, goal, snake_body):
        """
        Algorithme A* pour trouver le chemin optimal de start à goal.
        Retourne une liste de positions formant le chemin.
        """
        start_tuple = tuple(start)
        goal_tuple = tuple(goal)

        # File de priorité : (f_score, position)
        open_set = [(0, start_tuple)]

        # Dictionnaires pour reconstruire le chemin
        came_from = {}
        g_score = {start_tuple: 0}
        f_score = {start_tuple: self.heuristic(start_tuple, goal_tuple)}

        open_set_hash = {start_tuple}

        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.discard(current)

            # Objectif atteint
            if current == goal_tuple:
                path = []
                while current in came_from:
                    path.append(list(current))
                    current = came_from[current]
                path.reverse()
                return path

            # Explore les voisins
            for neighbor in self.get_neighbors(current):
                # Vérifie si la position est sûre
                if not self.is_safe_position(neighbor, snake_body):
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + \
                        self.heuristic(neighbor, goal_tuple)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        return []  # Aucun chemin trouvé

    def find_safe_move(self, snake):
        """
        Trouve un mouvement sûr si aucun chemin vers la pomme n'existe.
        Privilégie les mouvements qui maximisent l'espace libre.
        """
        head_pos = tuple(snake.head_pos)
        max_space = -1
        best_direction = None

        for direction in [UP, DOWN, LEFT, RIGHT]:
            # Évite les demi-tours
            if (direction[0] * -1, direction[1] * -1) == snake.direction:
                continue

            # Calcule la nouvelle position
            new_x = (head_pos[0] + direction[0]) % GRID_SIZE
            new_y = (head_pos[1] + direction[1]) % GRID_SIZE
            new_pos = (new_x, new_y)

            # Vérifie si c'est sûr
            if not self.is_safe_position(new_pos, snake.body):
                continue

            # Compte l'espace accessible depuis cette position (BFS simple)
            accessible_space = self.count_accessible_space(new_pos, snake.body)

            if accessible_space > max_space:
                max_space = accessible_space
                best_direction = direction

        return best_direction

    def count_accessible_space(self, start, snake_body):
        """Compte le nombre de cases accessibles depuis une position (BFS)."""
        visited = {tuple(start)}
        queue = deque([start])
        count = 0

        while queue and count < 50:  # Limite pour éviter trop de calculs
            current = queue.popleft()
            count += 1

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and self.is_safe_position(neighbor, snake_body):
                    visited.add(neighbor)
                    queue.append(neighbor)

        return count

    def get_next_direction(self, snake, apple):
        """
        Calcule la prochaine direction que le serpent doit prendre.
        Retourne une direction (UP, DOWN, LEFT, RIGHT).
        """
        # Quand le serpent est long (> largeur de la grille), recalcule à chaque mouvement
        # pour prendre en compte les changements de position de la queue
        force_recalculate = len(snake.body) > GRID_SIZE
        
        # Si on a déjà un chemin et qu'il est toujours valide, on le suit
        # SAUF si le serpent est très long (auquel cas on force le recalcul)
        if not force_recalculate and self.path and len(self.path) > 0:
            next_pos = self.path[0]
            # Vérifie que le prochain mouvement est toujours sûr
            if self.is_safe_position(next_pos, snake.body):
                self.path.pop(0)
                dx = next_pos[0] - snake.head_pos[0]
                dy = next_pos[1] - snake.head_pos[1]

                # Gestion du wrap-around
                if abs(dx) > 1:
                    dx = -1 if dx > 0 else 1
                if abs(dy) > 1:
                    dy = -1 if dy > 0 else 1

                return (dx, dy)

        # Cherche un nouveau chemin vers la pomme
        # (systématiquement si le serpent est long, ou si le chemin est invalide)
        self.path = self.a_star(
            snake.head_pos, list(apple.position), snake.body)

        # Si un chemin existe, on le suit
        if self.path and len(self.path) > 0:
            next_pos = self.path.pop(0)
            dx = next_pos[0] - snake.head_pos[0]
            dy = next_pos[1] - snake.head_pos[1]

            # Gestion du wrap-around
            if abs(dx) > 1:
                dx = -1 if dx > 0 else 1
            if abs(dy) > 1:
                dy = -1 if dy > 0 else 1

            return (dx, dy)

        # Pas de chemin vers la pomme, trouve un mouvement sûr
        safe_direction = self.find_safe_move(snake)
        if safe_direction:
            return safe_direction

        # Dernière option : continue dans la direction actuelle
        return snake.direction

    def draw_path(self, surface, path):
        """Dessine le chemin planifié sur l'écran (pour le debug)."""
        for i, pos in enumerate(path):
            # Diminue l'opacité pour les positions lointaines
            alpha = 255 - (i * 10)
            if alpha < 50:
                alpha = 50

            rect = pygame.Rect(pos[0] * CELL_SIZE + CELL_SIZE // 4,
                               pos[1] * CELL_SIZE +
                               SCORE_PANEL_HEIGHT + CELL_SIZE // 4,
                               CELL_SIZE // 2,
                               CELL_SIZE // 2)
            pygame.draw.rect(surface, self.BLUE, rect, border_radius=3)

# --- FONCTIONS D'AFFICHAGE ---


def draw_grid(surface):
    """Dessine la grille pour une meilleure visualisation."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE,
                         (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def display_info(surface, font, snake, start_time, auto_mode=False):
    """Affiche le score et le temps écoulé dans le panneau supérieur."""

    # Dessiner le panneau de score
    pygame.draw.rect(surface, GRIS_FOND,
                     (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2),
                     (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

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
    surface.blit(fill_text, (SCREEN_WIDTH // 2 -
                 fill_text.get_width() // 2, 20))

    # Afficher le mode (AUTO ou MANUEL)
    mode_text = "MODE: AUTO (A*)" if auto_mode else "MODE: MANUEL"
    mode_color = (100, 150, 255) if auto_mode else BLANC
    mode_surface = font.render(mode_text, True, mode_color)
    surface.blit(mode_surface, (10, 50))


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

# --- BOUCLE PRINCIPALE DU JEU ---


def main():
    """Fonction principale pour exécuter le jeu Snake avec algorithme A*."""
    pygame.init()

    # Configuration de l'écran
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake avec A* - Groupe 13")
    clock = pygame.time.Clock()

    # Configuration des polices
    font_main = pygame.font.Font(None, 25)
    font_game_over = pygame.font.Font(None, 60)

    # Initialisation des objets du jeu
    snake = Snake()
    apple = Apple(snake.body)
    ai = SnakeAI()

    # Variables de jeu
    running = True
    game_over = False
    victory = False
    auto_mode = True  # Mode automatique activé par défaut
    show_path = True  # Afficher le chemin planifié

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
                    # Logique de redémarrage : seulement si le jeu est terminé
                    if event.key == pygame.K_SPACE:
                        main()  # Redémarre le jeu en appelant main()
                        return
                else:
                    # Touche 'A' pour basculer entre mode auto et manuel
                    if event.key == pygame.K_a:
                        auto_mode = not auto_mode
                        if not auto_mode:
                            ai.path = []  # Réinitialise le chemin en mode manuel

                    # Touche 'P' pour afficher/masquer le chemin
                    elif event.key == pygame.K_p:
                        show_path = not show_path

                    # Logique de déplacement manuel : seulement si mode manuel
                    elif not auto_mode:
                        if event.key == pygame.K_UP:
                            snake.set_direction(UP)
                        elif event.key == pygame.K_DOWN:
                            snake.set_direction(DOWN)
                        elif event.key == pygame.K_LEFT:
                            snake.set_direction(LEFT)
                        elif event.key == pygame.K_RIGHT:
                            snake.set_direction(RIGHT)

        # 2. Logique de Mise à Jour du Jeu
        if not game_over and not victory:
            # Le serpent se déplace à la vitesse définie
            move_counter += 1
            # Calcul du seuil de mouvement (minimum 1 pour éviter division par zéro)
            move_threshold = max(1, GAME_SPEED // 10)
            
            if move_counter >= move_threshold:  # Déplace le serpent à un rythme constant
                # En mode automatique, l'IA calcule la direction JUSTE AVANT le mouvement
                if auto_mode:
                    next_direction = ai.get_next_direction(snake, apple)
                    if next_direction:
                        snake.set_direction(next_direction)
                
                snake.move()
                move_counter = 0

                # Vérification des collisions (murs et corps)
                if snake.is_game_over():
                    game_over = True
                    continue  # Passe à l'affichage de Game Over

                # Vérification de la pomme mangée
                if snake.head_pos == list(apple.position):
                    snake.grow()

                    # Tente de replacer la pomme, vérifie la Victoire si échec
                    if not apple.relocate(snake.body):
                        victory = True  # Plus d'espace pour la pomme
                        game_over = True  # Met fin au jeu

        # 3. Dessin
        screen.fill(GRIS_FOND)  # Fond gris pour la zone de score

        # Zone de jeu (décalée par la hauteur du panneau de score)
        game_area_rect = pygame.Rect(
            0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)

        draw_grid(screen)

        # Dessine le chemin planifié (si activé et en mode auto)
        if show_path and auto_mode and ai.path:
            ai.draw_path(screen, ai.path)

        # Dessine la pomme et le serpent
        apple.draw(screen)
        snake.draw(screen)

        # Affiche le score et le temps
        display_info(screen, font_main, snake, start_time, auto_mode)

        # Affichage des messages de fin de jeu et instructions
        if game_over:
            if victory:
                # Le premier message est centré (y_offset=0 par défaut)
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
                message_details = f"Score final: {snake.score} - ESPACE pour rejouer"
                # Le deuxième message est décalé vers le bas
                display_message(screen, font_main,
                                message_details, BLANC, y_offset=100)
            else:
                # Le premier message est centré (y_offset=0 par défaut)
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
                message_details = f"Score: {snake.score} - ESPACE pour rejouer"
                # Le deuxième message est décalé vers le bas
                display_message(screen, font_main,
                                message_details, BLANC, y_offset=100)
        else:
            # Instructions en bas de l'écran pendant le jeu
            font_small = pygame.font.Font(None, 20)
            instructions = "A: Mode Auto/Manuel | P: Afficher chemin | Fleches: Controle manuel"
            instr_surface = font_small.render(instructions, True, GRIS_GRILLE)
            screen.blit(instr_surface, (10, SCREEN_HEIGHT - 20))

        # Mise à jour de l'affichage
        pygame.display.flip()

        # Contrôle la vitesse du jeu
        clock.tick(GAME_SPEED)

    pygame.quit()


if __name__ == '__main__':
    main()
