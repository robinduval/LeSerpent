import pygame
import random
import time

# --- CONSTANTES DE JEU ---
# Taille de la grille (15x15 par défaut)
GRID_SIZE = 15
# Taille d'une cellule en pixels
CELL_SIZE = 30
# Framerate d'affichage (images par seconde)
GAME_FPS = 60
# Vitesse de déplacement du serpent (mouvements par seconde)
MOVE_RATE = 60  # GBFS ultra-rapide

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
BLEU = (0, 150, 255)  # Chemin prévu

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
        # Calcul de la nouvelle position de la tête (wrap-around)
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
        """(Non utilisé ici car on fait wrap-around) Vérifie si la tête touche les bords."""
        x, y = self.head_pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

    def check_self_collision(self):
        """Vérifie si la tête touche une partie du corps (Game Over si auto-morsure)."""
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si le jeu est terminé (mur ou morsure)."""
        return self.check_wall_collision() or self.check_self_collision()

    def draw(self, surface):
        """Dessine le serpent sur la surface de jeu."""
        # Dessine le corps
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)  # Bordure

        # Dessine la tête (couleur différente)
        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)  # Bordure plus épaisse


class Apple:
    """Représente la pomme (nourriture) et sa position."""
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aléatoire non occupée par le serpent."""
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if [pos[0], pos[1]] not in occupied_positions]

        if not available_positions:
            return None  # Toutes les cases sont pleines (condition de Victoire)

        px, py = random.choice(available_positions)
        return [px, py]

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
            # Ajout d'un petit reflet pour un aspect "pomme"
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)


# --- FONCTIONS D'AFFICHAGE ---


def draw_grid(surface):
    """Dessine la grille pour une meilleure visualisation."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def draw_planned_path(surface, path):
    """Dessine le chemin prévu en bleu pour voir l'anticipation du serpent."""
    if not path:
        return
    for pos in path:
        # pos est déjà un tuple (x, y)
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x, y = pos[0], pos[1]
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            # Dessine un cercle bleu semi-transparent
            pygame.draw.circle(surface, BLEU, rect.center, CELL_SIZE // 4)


###############################
# Contrôleur automatique (non-IA)
###############################
# Si True, le serpent sera contrôlé automatiquement par un algorithme déterministe
# qui suit un parcours Hamiltonien (parcours en zigzag qui couvre toutes les cases).
# Ce n'est ni Dijkstra ni GBFS — juste un parcours couvrant constamment toute la grille.
AUTO_PLAY = True


def make_hamiltonian_path(grid_size):
    """Construit un parcours Hamiltonien (zigzag ligne par ligne) qui visite
    chaque case exactement une fois. Le parcours est une liste de (x,y).

    Pour toute grille rectangulaire, le zigzag (aller-retour par lignes)
    est un Hamiltonian path (pas forcément un cycle si les deux dimensions
    sont impaires). Ce parcours garantit que, si on le suit strictement,
    le serpent n'effectuera jamais une auto-collision et finira par visiter
    toutes les cases (victoire lorsque plus de place pour la pomme).
    """
    path = []
    for y in range(grid_size):
        # lignes paires: gauche->droite, lignes impaires: droite->gauche
        if y % 2 == 0:
            for x in range(grid_size):
                path.append((x, y))
        else:
            for x in range(grid_size - 1, -1, -1):
                path.append((x, y))
    return path


def build_pos_index_map(path):
    """Retourne un dictionnaire position -> index dans le parcours."""
    return {pos: idx for idx, pos in enumerate(path)}


def bfs_find_path(start, goal, snake_body, grid_size):
    """BFS pour trouver le chemin le plus court de start à goal.
    Retourne le chemin (liste de positions) ou None si impossible.
    Évite les cases occupées par le corps du serpent.
    """
    from collections import deque
    
    if start == goal:
        return [start]
    
    snake_set = set(tuple(seg) for seg in snake_body)
    visited = {tuple(start)}
    queue = deque([(tuple(start), [tuple(start)])])
    
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    while queue:
        pos, path = queue.popleft()
        
        for dx, dy in directions:
            nx = (pos[0] + dx) % grid_size
            ny = (pos[1] + dy) % grid_size
            next_pos = (nx, ny)
            
            if next_pos == tuple(goal):
                return path + [next_pos]
            
            if next_pos not in visited and next_pos not in snake_set:
                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))
    
    return None


def is_safe_move(head, next_pos, apple_pos, snake_body, tail_pos, grid_size):
    """Vérifie si un mouvement est sûr en s'assurant qu'après avoir mangé la pomme,
    on peut toujours atteindre la queue (garantit qu'on ne se piège pas).
    """
    # Simule le corps après le mouvement
    future_body = [list(next_pos)] + [list(seg) for seg in snake_body[:-1]]
    
    # Si on mange la pomme, le serpent grandit
    if list(next_pos) == apple_pos:
        future_body = [list(next_pos)] + [list(seg) for seg in snake_body]
        future_tail = future_body[-1]
    else:
        future_tail = future_body[-1]
    
    # Vérifie qu'on peut atteindre la queue depuis la nouvelle position
    path_to_tail = bfs_find_path(list(next_pos), future_tail, future_body[1:], grid_size)
    return path_to_tail is not None and len(path_to_tail) > 1


def flood_fill_count(start_pos, snake_body, grid_size):
    """Compte le nombre de cases accessibles depuis start_pos (flood fill).
    Utilisé pour détecter les poches (règle E).
    """
    from collections import deque
    
    snake_set = set(tuple(seg) for seg in snake_body)
    visited = {tuple(start_pos)}
    queue = deque([tuple(start_pos)])
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    while queue:
        pos = queue.popleft()
        
        for dx, dy in directions:
            nx = (pos[0] + dx) % grid_size
            ny = (pos[1] + dy) % grid_size
            next_pos = (nx, ny)
            
            if next_pos not in visited and next_pos not in snake_set:
                visited.add(next_pos)
                queue.append(next_pos)
    
    return len(visited)


def can_reach_tail_after_eating(head, next_pos, apple_pos, snake_body, grid_size):
    """RÈGLE A(2) : Vérifie si on peut atteindre la queue après avoir mangé la pomme.
    
    Simule le corps après avoir mangé, en traitant la queue actuelle comme libre
    à l'étape suivante (elle aura bougé).
    """
    # Simule le futur corps après avoir mangé
    if list(next_pos) == list(apple_pos):
        # On mange : le corps grandit
        future_body = [list(next_pos)] + [list(seg) for seg in snake_body]
    else:
        # Mouvement normal
        future_body = [list(next_pos)] + [list(seg) for seg in snake_body[:-1]]
    
    if len(future_body) < 2:
        return True
    
    future_tail = future_body[-1]
    
    # BFS vers la queue, en excluant le corps sauf la queue (qui sera libre)
    path_to_tail = bfs_find_path(list(next_pos), future_tail, future_body[1:-1], grid_size)
    return path_to_tail is not None


def is_move_consistent_with_cycle(current_idx, next_pos, pos_to_index, tail_pos):
    """RÈGLE B : Vérifie que le mouvement respecte l'ordre du cycle hamiltonien.
    
    Autorise seulement les mouvements vers des indices croissants (modulo)
    ou vers la queue (exception tolérée).
    """
    if next_pos not in pos_to_index:
        return False
    
    next_idx = pos_to_index[next_pos]
    
    # Exception : peut toujours rejoindre la queue
    if next_pos == tail_pos:
        return True
    
    # Vérifie que l'index progresse (modulo len(cycle))
    # Autorise une progression normale : idx+1, idx+2, etc.
    # Refuse les retours en arrière dans le cycle
    total_positions = len(pos_to_index)
    forward_dist = (next_idx - current_idx) % total_positions
    
    # Autorise si on avance de 1 à 10% du cycle max
    return 1 <= forward_dist <= max(1, total_positions // 10)


def check_parity_compatible(head_pos, apple_pos, snake_length, grid_size):
    """RÈGLE D : Vérifie la compatibilité de parité (damier).
    
    Sur une grille en damier, la distance de Manhattan entre deux cases
    doit avoir la même parité que la différence de longueur nécessaire.
    """
    # Distance de Manhattan (sans wrap pour simplifier)
    dx = abs(head_pos[0] - apple_pos[0])
    dy = abs(head_pos[1] - apple_pos[1])
    
    # Ajuste pour le wrap-around
    dx = min(dx, grid_size - dx)
    dy = min(dy, grid_size - dy)
    
    manhattan_dist = dx + dy
    
    # La parité doit correspondre
    # Si la distance est paire, on peut atteindre en nombre pair de mouvements
    return manhattan_dist % 2 == 0 or snake_length < grid_size * grid_size * 0.5


def avoid_pocket(next_pos, snake_body, snake_length, grid_size):
    """RÈGLE E : Évite de créer une poche trop petite.
    
    Utilise flood fill pour vérifier que la zone accessible après le mouvement
    est suffisamment grande pour contenir le serpent.
    """
    # Simule le corps après le mouvement
    future_body = [list(next_pos)] + [list(seg) for seg in snake_body[:-1]]
    
    # Compte les cases accessibles depuis next_pos
    accessible_count = flood_fill_count(next_pos, future_body[1:], grid_size)
    
    # Exige que la zone accessible soit au moins égale à la longueur du serpent + marge
    required_space = snake_length + 2
    
    return accessible_count >= required_space


def get_available_neighbors(pos, snake_body, grid_size):
    """Retourne les voisins accessibles (pas dans le corps du serpent)."""
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    neighbors = []
    snake_set = set(tuple(seg) for seg in snake_body)
    
    for dx, dy in directions:
        nx = (pos[0] + dx) % grid_size
        ny = (pos[1] + dy) % grid_size
        if (nx, ny) not in snake_set:
            neighbors.append((nx, ny))
    
    return neighbors


# Variable globale pour le cooldown des raccourcis (RÈGLE F)
last_shortcut_turn = -999


def manhattan_distance(pos1, pos2):
    """Calcule la distance de Manhattan entre deux positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def gbfs_find_path(start, goal, snake_body, grid_size):
    """GBFS (Greedy Best-First Search) pour trouver un chemin vers goal.
    
    Utilise la distance de Manhattan comme heuristique.
    Plus agressif que BFS : explore d'abord les cases les plus proches du but.
    
    Retourne le chemin ou None si impossible.
    """
    from collections import deque
    import heapq
    
    if start == goal:
        return [start]
    
    snake_set = set(tuple(seg) for seg in snake_body)
    visited = {tuple(start)}
    
    # Priority queue : (heuristique, position, chemin)
    pq = [(manhattan_distance(start, goal), tuple(start), [tuple(start)])]
    
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    while pq:
        _, pos, path = heapq.heappop(pq)
        
        for dx, dy in directions:
            nx = (pos[0] + dx) % grid_size
            ny = (pos[1] + dy) % grid_size
            next_pos = (nx, ny)
            
            if next_pos == tuple(goal):
                return path + [next_pos]
            
            if next_pos not in visited and next_pos not in snake_set:
                visited.add(next_pos)
                h = manhattan_distance(next_pos, goal)
                heapq.heappush(pq, (h, next_pos, path + [next_pos]))
    
    return None


def is_trapped(head_pos, snake_body, grid_size):
    """Détecte si le serpent s'est enfermé dans une zone sans issue.
    
    Retourne True si aucune direction ne mène à un espace suffisant.
    """
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    snake_set = set(tuple(seg) for seg in snake_body[1:])
    
    # Vérifie chaque direction possible
    for dx, dy in directions:
        nx = (head_pos[0] + dx) % grid_size
        ny = (head_pos[1] + dy) % grid_size
        next_pos = (nx, ny)
        
        if next_pos not in snake_set:
            # Vérifie l'espace disponible après ce mouvement
            accessible = flood_fill_count(next_pos, snake_body, grid_size)
            
            # Si au moins une direction donne assez d'espace, pas piégé
            if accessible >= len(snake_body) + 5:
                return False
    
    # Toutes les directions mènent à des espaces trop petits
    return True


def next_direction_along_path(snake, apple, path, pos_to_index, grid_size):
    """Algorithme GBFS (Greedy Best-First Search) avec Hamiltonian fallback.
    
    Stratégie ultra-agressive :
    1. Utilise GBFS (heuristique distance de Manhattan) vers la pomme
       - Plus rapide que BFS
       - Explore d'abord les directions qui se rapprochent du but
    2. Seule vérification : case libre (pas de check tail!)
    3. Si bloqué (détection par flood fill), bascule sur cycle Hamiltonien
       jusqu'à se débloquer
    
    Retourne : (direction, chemin_prevu) pour visualisation
    """
    head = tuple(snake.head_pos)
    apple_pos = tuple(apple.position) if apple.position else None
    tail_pos = tuple(snake.body[-1])
    snake_body_set = set(tuple(seg) for seg in snake.body[1:])
    
    # Détecte si on est piégé dans une poche
    trapped = is_trapped(head, snake.body, grid_size)
    
    # Si piégé, mode survie immédiat
    if trapped:
        if head not in pos_to_index:
            # Dernière chance : cherche n'importe quelle direction libre
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx = (head[0] + dx) % grid_size
                ny = (head[1] + dy) % grid_size
                if (nx, ny) not in snake_body_set:
                    return ((dx, dy), [])
            return (snake.direction, [])
        
        # Suit le cycle Hamiltonien pour se débloquer
        idx = pos_to_index[head]
        next_idx = (idx + 1) % len(path)
        
        nx, ny = path[next_idx]
        dx = nx - head[0]
        dy = ny - head[1]
        
        ham_path = []
        for i in range(min(15, len(path))):
            ham_path.append(path[(idx + i) % len(path)])
        
        return ((dx, dy), ham_path)
    
    # MODE AGRESSIF : GBFS vers la pomme
    if apple_pos:
        path_to_apple = gbfs_find_path(list(head), list(apple_pos), snake.body[1:], grid_size)
        
        if path_to_apple and len(path_to_apple) > 1:
            next_pos = tuple(path_to_apple[1])
            
            # Vérification minimale : juste que la case est libre
            if next_pos not in snake_body_set:
                dx = next_pos[0] - head[0]
                dy = next_pos[1] - head[1]
                
                # Gère le wrap-around
                if abs(dx) > 1:
                    dx = -1 if dx > 0 else 1
                if abs(dy) > 1:
                    dy = -1 if dy > 0 else 1
                
                return ((dx, dy), path_to_apple)
    
    # PLAN B : Suit la queue si pas de chemin vers pomme
    if tail_pos != head:
        path_to_tail = gbfs_find_path(list(head), list(tail_pos), snake.body[1:-1], grid_size)
        
        if path_to_tail and len(path_to_tail) > 1:
            next_pos = tuple(path_to_tail[1])
            
            if next_pos not in snake_body_set:
                dx = next_pos[0] - head[0]
                dy = next_pos[1] - head[1]
                
                if abs(dx) > 1:
                    dx = -1 if dx > 0 else 1
                if abs(dy) > 1:
                    dy = -1 if dy > 0 else 1
                
                return ((dx, dy), path_to_tail)
    
    # FALLBACK : Cycle Hamiltonien (sécurité absolue)
    if head not in pos_to_index:
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx = (head[0] + dx) % grid_size
            ny = (head[1] + dy) % grid_size
            if (nx, ny) not in snake_body_set:
                return ((dx, dy), [])
        return (snake.direction, [])
    
    idx = pos_to_index[head]
    next_idx = (idx + 1) % len(path)
    
    nx, ny = path[next_idx]
    dx = nx - head[0]
    dy = ny - head[1]
    
    ham_path = []
    for i in range(min(15, len(path))):
        ham_path.append(path[(idx + i) % len(path)])
    
    return ((dx, dy), ham_path)


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
    """Fonction principale pour exécuter le jeu Snake Classique."""
    global AUTO_PLAY
    pygame.init()

    # Configuration de l'écran
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Classique - Socle de Base")
    clock = pygame.time.Clock()

    # Configuration des polices
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    # --- Prépare le contrôleur automatique ---
    hamiltonian_path = make_hamiltonian_path(GRID_SIZE)
    pos_to_index = build_pos_index_map(hamiltonian_path)

    # Initialisation des objets du jeu
    snake = Snake()
    # Place le serpent initialement sur trois cases consécutives du parcours
    # pour assurer que l'auto-contrôleur peut immédiatement prendre la main.
    # Choisis un index de départ (ici 2) pour avoir des indices valides i-1, i-2.
    start_i = 2 if len(hamiltonian_path) > 3 else 0
    snake.body = [[hamiltonian_path[start_i][0], hamiltonian_path[start_i][1]],
                  [hamiltonian_path[start_i - 1][0], hamiltonian_path[start_i - 1][1]],
                  [hamiltonian_path[start_i - 2][0], hamiltonian_path[start_i - 2][1]]]
    snake.head_pos = snake.body[0]
    # Définir la direction initiale vers la prochaine case du chemin
    if start_i + 1 < len(hamiltonian_path):
        nx, ny = hamiltonian_path[start_i + 1]
        dx = nx - snake.head_pos[0]
        dy = ny - snake.head_pos[1]
        snake.direction = (dx, dy)

    apple = Apple(snake.body)

    # Variables de jeu
    running = True
    game_over = False
    victory = False

    # Démarrage du chronomètre
    start_time = time.time()

    # Accumulateur pour la gestion temporelle des mouvements
    time_since_move = 0.0
    
    # Chemin prévu pour visualisation
    planned_path = []

    # --- Boucle de jeu ---
    while running:
        # Limiter la fréquence et récupérer dt
        dt = clock.tick(GAME_FPS) / 1000.0

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
                # Basculer auto-play avec la touche A
                if event.key == pygame.K_a:
                    AUTO_PLAY = not AUTO_PLAY
                    print(f"AUTO_PLAY = {AUTO_PLAY}")
                else:
                    # Logique de déplacement : seulement si le jeu est en cours
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
            time_since_move += dt
            move_interval = 1.0 / MOVE_RATE
            if time_since_move >= move_interval:
                # Si le contrôleur automatique est activé, met à jour la direction
                if AUTO_PLAY:
                    dir_to_set, planned_path = next_direction_along_path(snake, apple, hamiltonian_path, pos_to_index, GRID_SIZE)
                    snake.set_direction(dir_to_set)

                snake.move()
                time_since_move -= move_interval

                # Vérification des collisions (murs et corps)
                if snake.is_game_over():
                    game_over = True
                    print(f"\n❌ GAME OVER au score {snake.score}")
                    print(f"   Position tête: {snake.head_pos}")
                    print(f"   Direction: {snake.direction}")
                    print(f"   Corps longueur: {len(snake.body)} segments")
                    print(f"   Index dans cycle: {pos_to_index.get(tuple(snake.head_pos), 'PAS DANS LE CYCLE!')}")
                    
                    # Vérifie si c'est une collision avec le corps
                    if snake.head_pos in snake.body[1:]:
                        collision_idx = snake.body.index(snake.head_pos, 1)
                        print(f"   💥 Collision avec le corps à l'index {collision_idx}")
                        print(f"   Segment touché: {snake.body[collision_idx]}")
                    
                    # Affiche les 5 derniers mouvements
                    print(f"   Chemin prévu avait {len(planned_path)} cases")
                    if len(planned_path) > 0:
                        print(f"   Prochaines cases prévues: {planned_path[:5]}")
                    
                    continue  # Passe à l'affichage de Game Over

                # Vérification de la pomme mangée
                if snake.head_pos == apple.position:
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
        
        # Dessine le chemin prévu (en bleu)
        draw_planned_path(screen, planned_path)

        # Dessine la pomme et le serpent
        apple.draw(screen)
        snake.draw(screen)

        # Affiche le score et le temps
        display_info(screen, font_main, snake, start_time)

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

    pygame.quit()


if __name__ == '__main__':
    main()
