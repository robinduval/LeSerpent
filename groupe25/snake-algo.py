import pygame
import random
import time
from collections import deque
import heapq

# --- CONSTANTES DE JEU ---
# Taille de la grille (20x20)
GRID_SIZE = 15
# Taille d'une cellule en pixels
CELL_SIZE = 30
# Vitesse de jeu (images par seconde)
GAME_SPEED = 15  # Augmenté pour l'algorithme automatique

# Dimensions de l'écran (avec espace pour le score/timer)
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (en RGB)
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)  # Tête du serpent
VERT = (0, 200, 0)  # Corps du serpent
ROUGE = (200, 0, 0)  # Pomme
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
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE,
                               CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)  # Bordure

        # Dessine la tête (couleur différente)
        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                                CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)  # Bordure plus épaisse


class Apple:
    """Représente la pomme (nourriture) et sa position."""

    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aléatoire non occupée par le serpent."""
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(pos) not in occupied_positions]

        if not available_positions:
            return None  # Toutes les cases sont pleines (condition de Victoire)

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
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            # Ajout d'un petit reflet pour un aspect "pomme"
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)


# --- ALGORITHME DE RÉSOLUTION AUTOMATIQUE ---

class SnakeAI:
    """
    Algorithme de résolution automatique du Snake utilisant A* et stratégie de suivre sa queue.
    Pas d'IA/ML, uniquement de la logique algorithmique pure.
    """

    def __init__(self):
        self.path = []

    def get_neighbors(self, pos):
        """Retourne les 4 voisins d'une position."""
        x, y = pos
        neighbors = []
        for dx, dy in [UP, DOWN, LEFT, RIGHT]:
            new_x = (x + dx) % GRID_SIZE
            new_y = (y + dy) % GRID_SIZE
            neighbors.append((new_x, new_y))
        return neighbors

    def is_safe(self, pos, snake_body, future_length=None):
        """Vérifie si une position est sûre (pas dans le corps du serpent)."""
        # Si on spécifie future_length, on simule le corps futur
        if future_length is not None:
            body_to_check = snake_body[:future_length]
        else:
            body_to_check = snake_body

        return [pos[0], pos[1]] not in body_to_check

    def manhattan_distance(self, pos1, pos2):
        """Distance de Manhattan entre deux positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def a_star(self, start, goal, snake_body):
        """
        Algorithme A* pour trouver le chemin le plus court.
        Retourne le chemin complet (liste de positions) ou None si impossible.
        """
        start = tuple(start)
        goal = tuple(goal)

        # File de priorité : (f_score, compteur, position, chemin)
        counter = 0
        heap = [(0, counter, start, [start])]
        visited = set()

        while heap:
            f_score, _, current, path = heapq.heappop(heap)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue

                # Vérifie si la case est sûre
                # On ignore les dernières cases de la queue car elles vont bouger
                future_body_length = len(snake_body) - len(path)
                if not self.is_safe(neighbor, snake_body, max(1, future_body_length)):
                    continue

                g_score = len(path)
                h_score = self.manhattan_distance(neighbor, goal)
                f = g_score + h_score

                counter += 1
                heapq.heappush(heap, (f, counter, neighbor, path + [neighbor]))

        return None

    def can_reach_tail_after(self, start, snake_body):
        """
        Vérifie si on peut atteindre la queue après avoir atteint 'start'.
        Cela garantit qu'on ne se piège pas.
        """
        # Simule le corps du serpent après avoir atteint start
        simulated_body = [list(start)] + snake_body[:-1]
        tail = tuple(simulated_body[-1])

        # Essaie de trouver un chemin vers la queue
        path = self.a_star(start, tail, simulated_body)
        return path is not None

    def get_longest_path_direction(self, head, snake_body):
        """
        En dernier recours : choisit la direction qui offre le plus d'espace libre.
        """
        best_direction = None
        max_space = -1

        for direction in [UP, DOWN, LEFT, RIGHT]:
            new_x = (head[0] + direction[0]) % GRID_SIZE
            new_y = (head[1] + direction[1]) % GRID_SIZE
            new_pos = (new_x, new_y)

            if not self.is_safe(new_pos, snake_body[:-1]):
                continue

            # Compte l'espace accessible depuis cette position
            space = self.count_accessible_space(new_pos, snake_body)

            if space > max_space:
                max_space = space
                best_direction = direction

        return best_direction if best_direction else RIGHT

    def count_accessible_space(self, start, snake_body):
        """Compte le nombre de cases accessibles depuis une position (BFS)."""
        visited = set()
        queue = deque([tuple(start)])
        visited.add(tuple(start))

        while queue:
            current = queue.popleft()

            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                if not self.is_safe(neighbor, snake_body[:-1]):
                    continue

                visited.add(neighbor)
                queue.append(neighbor)

        return len(visited)

    def get_next_move(self, snake, apple):
        """
        Décide du prochain mouvement basé sur l'algorithme A* et stratégie de sécurité.

        Stratégie :
        1. Chercher un chemin sûr vers la pomme avec A*
        2. Vérifier qu'après avoir mangé, on peut atteindre la queue (sécurité)
        3. Sinon, suivre la queue
        4. En dernier recours, aller vers l'espace le plus ouvert
        """
        head = tuple(snake.head_pos)
        apple_pos = tuple(apple.position) if apple.position else None
        body = snake.body

        # 1. Essayer de trouver un chemin vers la pomme
        if apple_pos:
            path_to_apple = self.a_star(head, apple_pos, body)

            if path_to_apple and len(path_to_apple) > 1:
                # Vérifie si c'est sûr (peut-on atteindre la queue après ?)
                if self.can_reach_tail_after(apple_pos, body):
                    # Chemin sûr trouvé, prend la première étape
                    next_pos = path_to_apple[1]
                    direction = (next_pos[0] - head[0], next_pos[1] - head[1])
                    # Normalise en cas de wrap-around
                    if abs(direction[0]) > 1:
                        direction = (-direction[0] // abs(direction[0]), direction[1])
                    if abs(direction[1]) > 1:
                        direction = (direction[0], -direction[1] // abs(direction[1]))
                    return direction

        # 2. Pas de chemin sûr vers la pomme, suivre la queue
        tail = tuple(body[-1])
        path_to_tail = self.a_star(head, tail, body)

        if path_to_tail and len(path_to_tail) > 1:
            next_pos = path_to_tail[1]
            direction = (next_pos[0] - head[0], next_pos[1] - head[1])
            # Normalise en cas de wrap-around
            if abs(direction[0]) > 1:
                direction = (-direction[0] // abs(direction[0]), direction[1])
            if abs(direction[1]) > 1:
                direction = (direction[0], -direction[1] // abs(direction[1]))
            return direction

        # 3. Dernier recours : aller vers l'espace le plus ouvert
        return self.get_longest_path_direction(head, body)


# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface):
    """Dessine la grille pour une meilleure visualisation."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def display_info(surface, font, snake, start_time):
    """Affiche le score et le temps écoulé dans le panneau supérieur."""

    # Dessiner le panneau de score
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    # Police plus petite pour le panneau
    small_font = pygame.font.Font(None, 28)

    # Afficher le score
    score_text = small_font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 20))

    # Afficher le temps
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_text = small_font.render(f"Temps: {minutes:02d}:{seconds:02d}", True, BLANC)
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 20))

    # Afficher le taux de remplissage
    max_cells = GRID_SIZE * GRID_SIZE
    fill_rate = (len(snake.body) / max_cells) * 100
    fill_text = small_font.render(f"Taux: {fill_rate:.1f}%", True, BLANC)
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


# --- BOUCLE PRINCIPALE DU JEU ---

def main():
    """Fonction principale pour exécuter le jeu Snake avec algorithme automatique."""
    pygame.init()

    # Configuration de l'écran
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Auto")
    clock = pygame.time.Clock()

    # Configuration des polices
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    # Initialisation des objets du jeu
    snake = Snake()
    apple = Apple(snake.body)
    ai = SnakeAI()  # Algorithme de résolution automatique

    # Variables de jeu
    running = True
    game_over = False
    victory = False
    auto_mode = True  # Mode automatique activé par défaut

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
                    # Toggle mode automatique avec la touche 'A'
                    if event.key == pygame.K_a:
                        auto_mode = not auto_mode

                    # Logique de déplacement manuel : seulement si mode auto désactivé
                    if not auto_mode:
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
            if move_counter >= GAME_SPEED // 10:  # Déplace le serpent à un rythme constant
                # Mode automatique : l'algorithme décide
                if auto_mode:
                    next_direction = ai.get_next_move(snake, apple)
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
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)

        draw_grid(screen)

        # Dessine la pomme et le serpent
        apple.draw(screen)
        snake.draw(screen)

        # Affiche le score et le temps
        display_info(screen, font_main, snake, start_time)

        # Affiche le mode (Auto/Manuel)
        small_font = pygame.font.Font(None, 28)
        mode_text = small_font.render(f"Mode: {'AUTO' if auto_mode else 'MANUEL'}", True,
                                      ORANGE if auto_mode else BLANC)
        screen.blit(mode_text, (10, 50))

        # Affichage des messages de fin de jeu
        if game_over:
            if victory:
                # Le premier message est centré (y_offset=0 par défaut)
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
                message_details = "ESPACE pour rejouer."
                # Le deuxième message est décalé vers le bas
                display_message(screen, font_main, message_details, BLANC, y_offset=100)
            else:
                # Le premier message est centré (y_offset=0 par défaut)
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
                message_details = "ESPACE pour rejouer."
                # Le deuxième message est décalé vers le bas
                display_message(screen, font_main, message_details, BLANC, y_offset=100)

        # Mise à jour de l'affichage
        pygame.display.flip()

        # Contrôle la vitesse du jeu
        clock.tick(GAME_SPEED)

    pygame.quit()


if __name__ == '__main__':
    main()