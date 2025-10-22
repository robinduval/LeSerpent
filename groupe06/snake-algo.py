"""
snake-algo.py - Agents IA basés sur le pathfinding algorithmique
Algorithmes: Dijkstra et Greedy Best-First Search (GBFS)
S'intègre avec le jeu Snake de serpent.py
"""

import heapq
import random
from serpent import Snake, Apple, UP, DOWN, LEFT, RIGHT, GRID_SIZE


class GameWrapper:
    """Wrapper pour adapter l'interface du jeu aux algorithmes de pathfinding."""
    
    def __init__(self, snake_obj, apple_obj):
        """
        Args:
            snake_obj: Objet Snake
            apple_obj: Objet Apple
        """
        self.snake = snake_obj
        self.apple = apple_obj
    
    def get_snake_head(self):
        """Retourne la position de la tête du serpent."""
        return tuple(self.snake.head_pos)
    
    def get_apple_position(self):
        """Retourne la position de la pomme."""
        return tuple(self.apple.position) if self.apple.position else None
    
    def is_walkable(self, x, y):
        """Vérifie si une position est accessible."""
        # Hors limites?
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return False
        # Corps du serpent? (exclut la tête)
        if [x, y] in self.snake.body[1:]:
            return False
        return True
    
    def get_neighbors(self, pos):
        """Retourne les 4 voisins orthogonaux avec leurs directions."""
        x, y = pos
        neighbors = []
        directions = [
            ((0, -1), UP),      # HAUT
            ((0, 1), DOWN),     # BAS
            ((-1, 0), LEFT),    # GAUCHE
            ((1, 0), RIGHT),    # DROITE
        ]
        for (dx, dy), direction in directions:
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                neighbors.append(((nx, ny), direction))
        return neighbors


def manhattan_distance(pos1, pos2):
    """Calcule la distance de Manhattan entre deux positions."""
    if pos2 is None:
        return float('inf')
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def dijkstra(wrapper, start, goal):
    """
    Algorithme de Dijkstra - Trouve le chemin le plus court garanti.
    
    Args:
        wrapper: GameWrapper contenant l'état du jeu
        start: Position de départ (tuple)
        goal: Position d'arrivée (tuple)
        
    Returns:
        list: Liste de positions formant le chemin optimal
    """
    if goal is None or start == goal:
        return []
    
    # Priorité: (coût, position, chemin)
    open_set = [(0, start, [start])]
    visited = set()
    
    while open_set:
        cost, current, path = heapq.heappop(open_set)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Objectif atteint?
        if current == goal:
            return path
        
        # Explorer les voisins
        for neighbor_pos, _ in wrapper.get_neighbors(current):
            if neighbor_pos not in visited:
                new_cost = cost + 1
                new_path = path + [neighbor_pos]
                heapq.heappush(open_set, (new_cost, neighbor_pos, new_path))
    
    return []  # Pas de chemin trouvé


def greedy_best_first_search(wrapper, start, goal):
    """
    Algorithme GBFS - Recherche glouonne avec heuristique Manhattan.
    Plus rapide que Dijkstra mais ne garantit pas l'optimalité.
    
    Args:
        wrapper: GameWrapper contenant l'état du jeu
        start: Position de départ (tuple)
        goal: Position d'arrivée (tuple)
        
    Returns:
        list: Liste de positions formant un bon chemin
    """
    if goal is None or start == goal:
        return []
    
    # Priorité: (heuristique, position, chemin)
    open_set = [(manhattan_distance(start, goal), start, [start])]
    visited = set()
    
    while open_set:
        _, current, path = heapq.heappop(open_set)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Objectif atteint?
        if current == goal:
            return path
        
        # Explorer les voisins
        for neighbor_pos, _ in wrapper.get_neighbors(current):
            if neighbor_pos not in visited:
                heuristic = manhattan_distance(neighbor_pos, goal)
                new_path = path + [neighbor_pos]
                heapq.heappush(open_set, (heuristic, neighbor_pos, new_path))
    
    return []  # Pas de chemin trouvé


def a_star(wrapper, start, goal):
    """
    Algorithme A* - Combine Dijkstra (coût réel g(n)) et GBFS (heuristique h(n)).
    Garantit l'optimalité tout en étant plus rapide que Dijkstra.
    
    f(n) = g(n) + h(n)
    - g(n): coût réel depuis le départ
    - h(n): heuristique (distance Manhattan)
    
    Args:
        wrapper: GameWrapper contenant l'état du jeu
        start: Position de départ (tuple)
        goal: Position d'arrivée (tuple)
        
    Returns:
        list: Liste de positions formant le chemin optimal
    """
    if goal is None or start == goal:
        return []
    
    # Priorité: (f(n), g(n), position, chemin)
    # g(n) = coût réel, h(n) = heuristique, f(n) = g(n) + h(n)
    h_start = manhattan_distance(start, goal)
    open_set = [(h_start, 0, start, [start])]
    visited = set()
    g_scores = {start: 0}  # Coût réel pour atteindre chaque nœud
    
    while open_set:
        f_score, g_current, current, path = heapq.heappop(open_set)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Objectif atteint?
        if current == goal:
            return path
        
        # Explorer les voisins
        for neighbor_pos, _ in wrapper.get_neighbors(current):
            if neighbor_pos not in visited:
                g_new = g_current + 1  # Coût réel
                
                # Si on a trouvé un meilleur chemin vers ce voisin
                if neighbor_pos not in g_scores or g_new < g_scores[neighbor_pos]:
                    g_scores[neighbor_pos] = g_new
                    h_new = manhattan_distance(neighbor_pos, goal)  # Heuristique
                    f_new = g_new + h_new  # f(n) = g(n) + h(n)
                    
                    new_path = path + [neighbor_pos]
                    heapq.heappush(open_set, (f_new, g_new, neighbor_pos, new_path))
    
    return []  # Pas de chemin trouvé


def get_algorithmic_move(snake, apple, algorithm_name='dijkstra'):
    """
    Calcule le prochain mouvement pour suivre le chemin optimal.
    
    Args:
        snake: Objet Snake
        apple: Objet Apple
        algorithm_name: 'dijkstra', 'gbfs', ou 'astar'
        
    Returns:
        tuple: Direction du prochain mouvement (dx, dy)
    """
    wrapper = GameWrapper(snake, apple)
    start = wrapper.get_snake_head()
    goal = wrapper.get_apple_position()
    
    if goal is None:
        return random.choice([UP, DOWN, LEFT, RIGHT])
    
    # Sélectionner l'algorithme
    if algorithm_name.lower() == 'dijkstra':
        path = dijkstra(wrapper, start, goal)
    elif algorithm_name.lower() == 'gbfs':
        path = greedy_best_first_search(wrapper, start, goal)
    elif algorithm_name.lower() == 'astar':
        path = a_star(wrapper, start, goal)
    else:
        raise ValueError(f"Algorithme inconnu: {algorithm_name}")
    
    # Si pas de chemin, retourner une direction aléatoire valide
    if not path or len(path) < 2:
        directions = [UP, DOWN, LEFT, RIGHT]
        return random.choice(directions)
    
    # Retourner la direction du premier pas vers la pomme
    next_pos = path[1]
    current_pos = path[0]
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    
    return (dx, dy)


if __name__ == '__main__':
    """
    Script de test pour les algorithmes de pathfinding.
    
    Usage:
        python snake-algo.py dijkstra
        python snake-algo.py gbfs
        python snake-algo.py astar
    """
    import sys
    import pygame
    import time
    from serpent import (
        SCREEN_WIDTH, SCREEN_HEIGHT, SCORE_PANEL_HEIGHT, GRIS_FOND, NOIR,
        draw_grid, display_info, display_message
    )
    
    # Initialiser pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake - Pathfinding Algorithmique")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)
    
    # Sélectionner l'algorithme
    algorithm = 'astar'  # Par défaut: A*
    if len(sys.argv) > 1:
        algorithm = sys.argv[1].lower()
    
    if algorithm not in ['dijkstra', 'gbfs', 'astar']:
        print(f"Algorithme invalide: {algorithm}")
        print("Utilisez: dijkstra, gbfs, ou astar")
        sys.exit(1)
    
    # Initialiser le jeu
    snake = Snake()
    apple = Apple(snake.body)
    
    print(f"Algorithme: {algorithm.upper()}")
    print("Appuyez sur ESPACE pour recommencer ou ÉCHAP pour quitter")
    
    running = True
    game_over = False
    total_score = 0
    total_games = 0
    move_counter = 0
    MOVE_DELAY = 5  # Vitesse du jeu
    
    start_time = time.time()
    
    while running:
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE and game_over:
                    snake = Snake()
                    apple = Apple(snake.body)
                    game_over = False
                    start_time = time.time()
        
        # Logique de jeu
        if not game_over:
            move_counter += 1
            if move_counter >= MOVE_DELAY // 10:
                # Obtenir le prochain mouvement
                move = get_algorithmic_move(snake, apple, algorithm)
                snake.set_direction(move)
                snake.move()
                move_counter = 0
                
                # Vérifier les collisions
                if snake.is_game_over():
                    game_over = True
                    total_games += 1
                    total_score += snake.score
                    continue
                
                # Vérifier si pomme mangée
                if snake.head_pos == list(apple.position):
                    snake.grow()
                    if not apple.relocate(snake.body):
                        # Victoire!
                        game_over = True
                        total_games += 1
                        total_score += snake.score
        
        # Dessin
        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        
        draw_grid(screen)
        apple.draw(screen)
        snake.draw(screen)
        display_info(screen, font_main, snake, start_time)
        
        # Afficher message de fin
        if game_over:
            display_message(screen, font_game_over, f"GAME OVER", (200, 0, 0))
            if total_games > 0:
                avg_score = total_score / total_games
                msg = f"Score: {snake.score} | Moyenne: {avg_score:.1f} | Parties: {total_games}"
                display_message(screen, font_main, msg, (255, 255, 255), y_offset=100)
            else:
                display_message(screen, font_main, "ESPACE pour rejouer", (255, 255, 255), y_offset=100)
        
        pygame.display.flip()
        clock.tick(20)
    
    pygame.quit()
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"\nRésultats finaux ({algorithm.upper()}):")
    print(f"  Parties jouées: {total_games}")
    if total_games > 0:
        print(f"  Score total: {total_score}")
        print(f"  Score moyen: {total_score / total_games:.1f}")
    print(f"  Temps écoulé: {minutes}m {seconds:.1f}s")
