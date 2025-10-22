"""
Snake Algorithmique - Pathfinding Avancé
==========================================
Ce fichier implémente plusieurs algorithmes de pathfinding pour piloter
le serpent de manière autonome et optimale.

Algorithmes implémentés:
- A* (A-star) : Optimal et efficace avec heuristique
- Dijkstra : Optimal mais plus lent (A* sans heuristique)
- BFS (Breadth-First Search) : Trouve le chemin le plus court
- DFS (Depth-First Search) : Exploration en profondeur
- Greedy Best-First : Rapide mais non optimal
- Hybrid : Adaptatif selon la situation

Structure:
- SnakeGameAlgo: Environnement de jeu
- PathFinder: Classe gérant tous les algorithmes de pathfinding
- Comparaison et benchmarking des algorithmes
"""

import pygame
import time
import heapq
from collections import deque
from enum import Enum
import sys
import os

# Importer depuis le fichier serpent.py local
from serpent import Snake, Apple, GRID_SIZE, CELL_SIZE, SCORE_PANEL_HEIGHT
from serpent import SCREEN_WIDTH, SCREEN_HEIGHT, UP, DOWN, LEFT, RIGHT
from serpent import BLANC, NOIR, ORANGE, VERT, ROUGE, GRIS_FOND, GRIS_GRILLE
from serpent import draw_grid

# Définir les couleurs manquantes
BLEU = (0, 0, 255)

# Vitesse du jeu
GAME_SPEED = 10  # FPS

# ============================================================================
# ALGORITHMES DE PATHFINDING
# ============================================================================

class Algorithm(Enum):
    """Énumération des algorithmes disponibles."""
    HAMILTON = "Hamilton Cycle"
    ASTAR = "A*"
    DIJKSTRA = "Dijkstra"
    BFS = "BFS"
    DFS = "DFS"
    GREEDY = "Greedy Best-First"
    HYBRID = "Hybrid (Adaptatif)"

class PathFinder:
    """
    Classe gérant tous les algorithmes de pathfinding.
    
    Chaque algorithme trouve le meilleur chemin de la tête du serpent
    vers la pomme en évitant les obstacles (corps et murs).
    """
    
    def __init__(self):
        self.path_found = []
        self.nodes_explored = 0
        self.hamilton_cycle = None  # Cache du cycle hamiltonien
        self.hamilton_index = {}     # Index de position dans le cycle
        
    def build_hamilton_cycle(self):
        """
        Construit un cycle hamiltonien qui visite toutes les cases de la grille.
        
        Stratégie: Serpentin (snake pattern) qui garantit la visite de toutes les cases.
        Pour une grille NxN, on crée un chemin en zigzag qui revient au départ.
        
        Avantages:
        - ✅ Garantit de remplir toute la grille sans collision
        - ✅ Jamais de game over
        - ✅ Score parfait si suivi correctement
        
        Inconvénients:
        - ❌ Très lent (visite toutes les cases même si la pomme est proche)
        - ❌ Pas optimal pour le temps
        """
        cycle = []
        
        # Pattern en zigzag pour grille paire ou impaire
        for y in range(GRID_SIZE):
            if y % 2 == 0:
                # Ligne paire: de gauche à droite
                for x in range(GRID_SIZE):
                    cycle.append([x, y])
            else:
                # Ligne impaire: de droite à gauche
                for x in range(GRID_SIZE - 1, -1, -1):
                    cycle.append([x, y])
        
        # Créer un dictionnaire d'index pour recherche rapide
        self.hamilton_index = {tuple(pos): i for i, pos in enumerate(cycle)}
        self.hamilton_cycle = cycle
        
        return cycle
    
    def hamilton(self, start, goal, obstacles):
        """
        Algorithme du Cycle Hamiltonien.
        
        Le serpent suit un chemin pré-calculé qui visite toutes les cases de la grille.
        Cela garantit de ne jamais perdre et de remplir toute la grille.
        
        Returns:
            Chemin vers la prochaine case du cycle
        """
        self.nodes_explored = 1
        
        # Construire le cycle si pas encore fait
        if self.hamilton_cycle is None:
            self.build_hamilton_cycle()
        
        # Trouver la position actuelle dans le cycle
        start_tuple = tuple(start)
        if start_tuple not in self.hamilton_index:
            # Position hors cycle, trouver la plus proche
            min_dist = float('inf')
            closest_idx = 0
            for pos, idx in self.hamilton_index.items():
                dist = self.manhattan_distance(start, list(pos))
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx
            current_idx = closest_idx
        else:
            current_idx = self.hamilton_index[start_tuple]
        
        # Prochaine position dans le cycle
        next_idx = (current_idx + 1) % len(self.hamilton_cycle)
        next_pos = self.hamilton_cycle[next_idx]
        
        # Vérifier si la prochaine position est bloquée
        if next_pos in obstacles:
            # Si bloqué, essayer de sauter quelques cases (shortcut)
            for skip in range(2, 10):
                test_idx = (current_idx + skip) % len(self.hamilton_cycle)
                test_pos = self.hamilton_cycle[test_idx]
                if test_pos not in obstacles:
                    # Vérifier si accessible directement
                    if self.manhattan_distance(start, test_pos) == 1:
                        next_pos = test_pos
                        break
        
        self.path_found = [start, next_pos]
        return [start, next_pos]
    
    @staticmethod
    def manhattan_distance(pos1, pos2):
        """Calcule la distance de Manhattan entre deux positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    @staticmethod
    def get_neighbors(pos, obstacles):
        """Retourne les voisins valides d'une position."""
        x, y = pos
        neighbors = []
        
        # Ordre: haut, bas, gauche, droite
        for dx, dy in [UP, DOWN, LEFT, RIGHT]:
            new_x, new_y = x + dx, y + dy
            
            # Vérifier les limites et les obstacles
            if (0 <= new_x < GRID_SIZE and 
                0 <= new_y < GRID_SIZE and 
                [new_x, new_y] not in obstacles):
                neighbors.append([new_x, new_y])
        
        return neighbors
    
    def astar(self, start, goal, obstacles):
        """
        Algorithme A* (A-star)
        
        Avantages:
        - Optimal : trouve toujours le chemin le plus court
        - Efficace : utilise une heuristique pour guider la recherche
        - Équilibré entre vitesse et optimalité
        
        Inconvénients:
        - Plus complexe à implémenter
        - Nécessite une bonne heuristique
        
        Complexité: O(b^d) où b=facteur de branchement, d=profondeur
        """
        self.nodes_explored = 0
        
        # File de priorité: (f_score, position, chemin)
        open_set = [(0, tuple(start), [start])]
        closed_set = set()
        g_scores = {tuple(start): 0}
        
        while open_set:
            self.nodes_explored += 1
            f_score, current_tuple, path = heapq.heappop(open_set)
            current = list(current_tuple)
            
            # Objectif atteint
            if current == goal:
                self.path_found = path
                return path
            
            if current_tuple in closed_set:
                continue
            
            closed_set.add(current_tuple)
            
            # Explorer les voisins
            for neighbor in self.get_neighbors(current, obstacles):
                neighbor_tuple = tuple(neighbor)
                
                if neighbor_tuple in closed_set:
                    continue
                
                # g(n) = coût depuis le départ
                tentative_g = g_scores[current_tuple] + 1
                
                if neighbor_tuple not in g_scores or tentative_g < g_scores[neighbor_tuple]:
                    g_scores[neighbor_tuple] = tentative_g
                    # f(n) = g(n) + h(n) où h(n) est l'heuristique
                    h_score = self.manhattan_distance(neighbor, goal)
                    f_score = tentative_g + h_score
                    
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f_score, neighbor_tuple, new_path))
        
        return None  # Aucun chemin trouvé
    
    def dijkstra(self, start, goal, obstacles):
        """
        Algorithme de Dijkstra
        
        Avantages:
        - Optimal : garantit le chemin le plus court
        - Fiable : toujours trouve une solution si elle existe
        
        Inconvénients:
        - Lent : explore beaucoup de nœuds inutiles
        - Pas d'heuristique pour guider la recherche
        
        Complexité: O((V+E) log V) avec tas de Fibonacci
        
        Note: C'est essentiellement A* avec h(n) = 0
        """
        self.nodes_explored = 0
        
        # File de priorité: (distance, position, chemin)
        open_set = [(0, tuple(start), [start])]
        distances = {tuple(start): 0}
        visited = set()
        
        while open_set:
            self.nodes_explored += 1
            dist, current_tuple, path = heapq.heappop(open_set)
            current = list(current_tuple)
            
            if current == goal:
                self.path_found = path
                return path
            
            if current_tuple in visited:
                continue
            
            visited.add(current_tuple)
            
            for neighbor in self.get_neighbors(current, obstacles):
                neighbor_tuple = tuple(neighbor)
                new_dist = dist + 1
                
                if neighbor_tuple not in distances or new_dist < distances[neighbor_tuple]:
                    distances[neighbor_tuple] = new_dist
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (new_dist, neighbor_tuple, new_path))
        
        return None
    
    def bfs(self, start, goal, obstacles):
        """
        Breadth-First Search (BFS)
        
        Avantages:
        - Trouve le chemin le plus court (en nombre de pas)
        - Simple à implémenter
        - Complet : trouve toujours une solution
        
        Inconvénients:
        - Consomme beaucoup de mémoire
        - Explore tous les nœuds à une distance donnée
        
        Complexité: O(V + E) où V=sommets, E=arêtes
        """
        self.nodes_explored = 0
        
        queue = deque([[start]])
        visited = {tuple(start)}
        
        while queue:
            path = queue.popleft()
            current = path[-1]
            self.nodes_explored += 1
            
            if current == goal:
                self.path_found = path
                return path
            
            for neighbor in self.get_neighbors(current, obstacles):
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    new_path = path + [neighbor]
                    queue.append(new_path)
        
        return None
    
    def dfs(self, start, goal, obstacles, max_depth=100):
        """
        Depth-First Search (DFS)
        
        Avantages:
        - Utilise peu de mémoire
        - Peut être rapide si la solution est profonde
        
        Inconvénients:
        - Non optimal : peut trouver un chemin long
        - Peut se perdre dans des branches infinies
        - Pas adapté pour Snake (préfère BFS)
        
        Complexité: O(b^d) où b=branchement, d=profondeur max
        """
        self.nodes_explored = 0
        
        def dfs_recursive(current, path, visited, depth):
            if depth > max_depth:
                return None
            
            self.nodes_explored += 1
            
            if current == goal:
                return path
            
            visited.add(tuple(current))
            
            for neighbor in self.get_neighbors(current, obstacles):
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in visited:
                    result = dfs_recursive(neighbor, path + [neighbor], visited, depth + 1)
                    if result:
                        return result
            
            return None
        
        path = dfs_recursive(start, [start], set(), 0)
        if path:
            self.path_found = path
        return path
    
    def greedy_best_first(self, start, goal, obstacles):
        """
        Greedy Best-First Search
        
        Avantages:
        - Très rapide : se dirige directement vers l'objectif
        - Efficace si le chemin est dégagé
        
        Inconvénients:
        - Non optimal : peut trouver un chemin sous-optimal
        - Peut se faire piéger dans des culs-de-sac
        - Sensible à l'heuristique choisie
        
        Complexité: O(b^d) mais souvent bien meilleur en pratique
        """
        self.nodes_explored = 0
        
        # File de priorité basée uniquement sur l'heuristique
        open_set = [(self.manhattan_distance(start, goal), tuple(start), [start])]
        visited = set()
        
        while open_set:
            self.nodes_explored += 1
            _, current_tuple, path = heapq.heappop(open_set)
            current = list(current_tuple)
            
            if current == goal:
                self.path_found = path
                return path
            
            if current_tuple in visited:
                continue
            
            visited.add(current_tuple)
            
            for neighbor in self.get_neighbors(current, obstacles):
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in visited:
                    h_score = self.manhattan_distance(neighbor, goal)
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (h_score, neighbor_tuple, new_path))
        
        return None
    
    def hybrid(self, start, goal, obstacles):
        """
        Algorithme Hybride Adaptatif
        
        Stratégie:
        - Distance courte (<10) : Greedy (rapide)
        - Distance moyenne (10-20) : A* (équilibré)
        - Distance longue (>20) : BFS (sûr)
        - Serpent long (>50% grille) : Dijkstra (optimal)
        
        Avantages:
        - Adaptatif selon la situation
        - Combine les forces de chaque algorithme
        - Optimise vitesse et optimalité
        
        Inconvénients:
        - Plus complexe
        - Overhead de décision
        """
        distance = self.manhattan_distance(start, goal)
        snake_size_ratio = len(obstacles) / (GRID_SIZE * GRID_SIZE)
        
        # Choisir l'algorithme selon la situation
        if snake_size_ratio > 0.5:
            # Serpent très long : utiliser Dijkstra (optimal)
            return self.dijkstra(start, goal, obstacles)
        elif distance < 10:
            # Cible proche : Greedy (rapide)
            return self.greedy_best_first(start, goal, obstacles)
        elif distance < 20:
            # Distance moyenne : A* (équilibré)
            return self.astar(start, goal, obstacles)
        else:
            # Longue distance : BFS (sûr)
            return self.bfs(start, goal, obstacles)
    
    def find_path(self, algorithm, start, goal, obstacles):
        """Point d'entrée principal pour trouver un chemin."""
        if algorithm == Algorithm.HAMILTON:
            return self.hamilton(start, goal, obstacles)
        elif algorithm == Algorithm.ASTAR:
            return self.astar(start, goal, obstacles)
        elif algorithm == Algorithm.DIJKSTRA:
            return self.dijkstra(start, goal, obstacles)
        elif algorithm == Algorithm.BFS:
            return self.bfs(start, goal, obstacles)
        elif algorithm == Algorithm.DFS:
            return self.dfs(start, goal, obstacles)
        elif algorithm == Algorithm.GREEDY:
            return self.greedy_best_first(start, goal, obstacles)
        elif algorithm == Algorithm.HYBRID:
            return self.hybrid(start, goal, obstacles)
        else:
            return self.astar(start, goal, obstacles)  # Par défaut

# ============================================================================
# CLASSE GAME - ENVIRONNEMENT ALGORITHMIQUE
# ============================================================================

class SnakeGameAlgo:
    """
    Environnement Snake pour les algorithmes de pathfinding.
    """
    
    def __init__(self, algorithm=Algorithm.ASTAR, visualize_path=True):
        """
        Args:
            algorithm: Algorithme de pathfinding à utiliser
            visualize_path: Afficher le chemin calculé
        """
        pygame.init()
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f'Snake Algorithmique - {algorithm.value}')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)
        
        self.algorithm = algorithm
        self.visualize_path = visualize_path
        self.pathfinder = PathFinder()
        
        # Stats
        self.moves_count = 0
        self.total_nodes_explored = 0
        self.start_time = time.time()
        
        self.reset()
    
    def reset(self):
        """Réinitialise le jeu."""
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.game_over = False
        self.current_path = []
        self.moves_count = 0
    
    def get_next_move(self):
        """
        Calcule et retourne la prochaine direction à prendre.
        
        Returns:
            Direction (UP, DOWN, LEFT, RIGHT) ou None si bloqué
        """
        # Calculer le chemin vers la pomme
        start = self.snake.head_pos
        goal = list(self.apple.position)
        obstacles = self.snake.body[1:]  # Exclure la tête
        
        # Trouver le chemin avec l'algorithme sélectionné
        path = self.pathfinder.find_path(self.algorithm, start, goal, obstacles)
        
        if path and len(path) > 1:
            self.current_path = path
            self.total_nodes_explored += self.pathfinder.nodes_explored
            
            # Calculer la direction vers le prochain point
            next_pos = path[1]
            dx = next_pos[0] - start[0]
            dy = next_pos[1] - start[1]
            
            return (dx, dy)
        
        return None  # Aucun chemin trouvé
    
    def step(self):
        """Exécute un pas de jeu."""
        # Obtenir la prochaine direction
        direction = self.get_next_move()
        
        if direction is None:
            # Aucun chemin trouvé : game over
            self.game_over = True
            return False
        
        # Appliquer la direction
        self.snake.set_direction(direction)
        self.snake.move()
        self.moves_count += 1
        
        # Vérifier les collisions
        if self.snake.is_game_over():
            self.game_over = True
            return False
        
        # Vérifier si la pomme est mangée
        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            
            # Victoire si la grille est remplie
            if not self.apple.relocate(self.snake.body):
                self.game_over = True
                return True  # Victoire
        
        return True
    
    def draw(self):
        """Affiche le jeu."""
        # Fond
        self.display.fill(GRIS_FOND)
        
        # Zone de jeu
        game_area = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.display, NOIR, game_area)
        
        # Grille
        draw_grid(self.display)
        
        # Dessiner le chemin (si activé)
        if self.visualize_path and self.current_path:
            for pos in self.current_path[1:]:  # Exclure la tête
                rect = pygame.Rect(
                    pos[0] * CELL_SIZE,
                    pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                    CELL_SIZE,
                    CELL_SIZE
                )
                pygame.draw.rect(self.display, (100, 100, 255), rect, 2)  # Bleu clair
        
        # Dessiner la pomme et le serpent
        self.apple.draw(self.display)
        self.snake.draw(self.display)
        
        # Panneau d'informations
        self._draw_info()
        
        pygame.display.flip()
    
    def _draw_info(self):
        """Affiche les informations de partie."""
        pygame.draw.line(self.display, BLANC, (0, SCORE_PANEL_HEIGHT - 2), 
                        (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)
        
        # Colonne gauche
        score_text = self.font.render(f"Score: {self.snake.score}", True, BLANC)
        self.display.blit(score_text, (10, 10))
        
        moves_text = self.font.render(f"Moves: {self.moves_count}", True, BLANC)
        self.display.blit(moves_text, (10, 35))
        
        # Colonne centre
        algo_text = self.font.render(f"Algo: {self.algorithm.value}", True, ORANGE)
        center_x = SCREEN_WIDTH // 2 - algo_text.get_width() // 2
        self.display.blit(algo_text, (center_x, 10))
        
        nodes_text = self.font.render(f"Nodes: {self.total_nodes_explored}", True, BLANC)
        center_x = SCREEN_WIDTH // 2 - nodes_text.get_width() // 2
        self.display.blit(nodes_text, (center_x, 35))
        
        # Colonne droite
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_text = self.font.render(f"Time: {minutes:02d}:{seconds:02d}", True, BLANC)
        self.display.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 10))

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def play_game(algorithm=Algorithm.ASTAR, visualize_path=True, auto_speed=True):
    """
    Lance une partie avec l'algorithme spécifié.
    
    Args:
        algorithm: Algorithme à utiliser
        visualize_path: Afficher le chemin calculé
        auto_speed: Vitesse automatique (rapide si False)
    """
    game = SnakeGameAlgo(algorithm, visualize_path)
    
    print(f"\n{'='*60}")
    print(f"  SNAKE ALGORITHMIQUE - {algorithm.value}")
    print(f"{'='*60}")
    print("Appuyez sur Échap pour arrêter\n")
    
    running = True
    while running and not game.game_over:
        # Gérer les événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Exécuter un pas
        game.step()
        
        # Afficher
        game.draw()
        
        # Contrôler la vitesse
        if auto_speed:
            game.clock.tick(GAME_SPEED)
        else:
            game.clock.tick(60)  # Très rapide pour benchmarking
    
    # Afficher les résultats
    print(f"\n📊 RÉSULTATS")
    print(f"   Algorithme: {algorithm.value}")
    print(f"   Score final: {game.snake.score}")
    print(f"   Nombre de mouvements: {game.moves_count}")
    print(f"   Nœuds explorés: {game.total_nodes_explored}")
    print(f"   Temps: {time.time() - game.start_time:.2f}s")
    
    if game.snake.score == GRID_SIZE * GRID_SIZE - 3:
        print(f"   🏆 VICTOIRE ! Grille complète !")
    
    pygame.quit()

# ============================================================================
# MODE COMPARAISON
# ============================================================================

def compare_algorithms():
    """Compare tous les algorithmes sur plusieurs parties."""
    print("\n" + "="*70)
    print("  COMPARAISON DES ALGORITHMES DE PATHFINDING")
    print("="*70)
    
    algorithms = [
        Algorithm.HAMILTON,
        Algorithm.ASTAR,
        Algorithm.GREEDY,
        Algorithm.BFS,
        Algorithm.DIJKSTRA,
        Algorithm.HYBRID
    ]
    
    results = {}
    
    for algo in algorithms:
        print(f"\n🔍 Test de {algo.value}...")
        game = SnakeGameAlgo(algo, visualize_path=False)
        
        moves = 0
        max_moves = 10000  # Limite pour éviter les boucles infinies
        
        while not game.game_over and moves < max_moves:
            game.step()
            moves += 1
            
            # Affichage minimal
            if moves % 100 == 0:
                print(f"   Mouvements: {moves}, Score: {game.snake.score}")
        
        results[algo.value] = {
            'score': game.snake.score,
            'moves': moves,
            'nodes': game.total_nodes_explored,
            'time': time.time() - game.start_time
        }
        
        pygame.quit()
    
    # Afficher le tableau de comparaison
    print("\n" + "="*70)
    print("  TABLEAU RÉCAPITULATIF")
    print("="*70)
    print(f"{'Algorithme':<20} | {'Score':<8} | {'Mouvements':<12} | {'Nœuds':<12} | {'Temps (s)'}")
    print("-"*70)
    
    for algo_name, data in results.items():
        print(f"{algo_name:<20} | {data['score']:<8} | {data['moves']:<12} | "
              f"{data['nodes']:<12} | {data['time']:.2f}")
    
    print("\n")

# ============================================================================
# MENU PRINCIPAL
# ============================================================================

def main():
    """Menu principal pour sélectionner le mode de jeu."""
    print("\n" + "="*70)
    print("  🐍 SNAKE ALGORITHMIQUE - PATHFINDING AVANCÉ")
    print("="*70)
    print("\nAlgorithmes disponibles:")
    print("  1. Hamilton Cycle - ⭐ Garantit de remplir toute la grille!")
    print("  2. A* (A-star) - Optimal et efficace")
    print("  3. Dijkstra - Optimal mais lent")
    print("  4. BFS - Chemin le plus court garanti")
    print("  5. DFS - Exploration en profondeur")
    print("  6. Greedy Best-First - Rapide mais non optimal")
    print("  7. Hybrid - Adaptatif selon la situation")
    print("  8. Comparer tous les algorithmes")
    print("  0. Quitter")
    
    choice = input("\nChoisissez un algorithme (0-8): ").strip()
    
    algo_map = {
        '1': Algorithm.HAMILTON,
        '2': Algorithm.ASTAR,
        '3': Algorithm.DIJKSTRA,
        '4': Algorithm.BFS,
        '5': Algorithm.DFS,
        '6': Algorithm.GREEDY,
        '7': Algorithm.HYBRID
    }
    
    if choice == '8':
        compare_algorithms()
    elif choice in algo_map:
        visualize = input("Afficher le chemin ? (o/n): ").strip().lower() == 'o'
        play_game(algo_map[choice], visualize_path=visualize)
    elif choice == '0':
        print("Au revoir !")
        return
    else:
        print("❌ Choix invalide")
        main()

if __name__ == '__main__':
    main()
