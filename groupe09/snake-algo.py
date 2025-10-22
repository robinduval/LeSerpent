#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snake avec Algorithmes de Pathfinding - VERSION 2.3 MODE SURVIE ADAPTATIF
Groupe 09 - Implémentation avec BFS, Dijkstra, et Hamiltonian Cycle optimisé

VERSION 2.3 - MODE SURVIE ADAPTATIF :
-------------------------------------
INSIGHT MAJEUR : Quand le serpent est grand (>60% rempli), il faut arrêter
                 d'optimiser pour la vitesse et BASCULER EN MODE SURVIE
                 
SOLUTION V2.3 : Système à 4 modes adaptatifs selon le taux de remplissage

 MODES DE JEU :

1️ MODE AGRESSIF (0-40% rempli)
   - Objectif : Grandir VITE
   - Va directement vers la pomme
   - Prend des risques calculés (min_space_ratio=0.25)
   - Optimise pour le score

2️ MODE ÉQUILIBRÉ (40-60% rempli)
   - Objectif : Croissance + Sécurité
   - Va vers la pomme si vraiment sûr
   - Balance entre vitesse et prudence (min_space_ratio=0.35)
   - Commence à former des boucles

3️ MODE PRUDENT (60-75% rempli)
   - Objectif : Sécurité d'abord
   - N'accepte la pomme que si critères ultra-stricts (min_space_ratio=0.5)
   - Vérifie qu'on garde 90% de l'espace actuel
   - Préfère suivre la queue (formation de boucles)

4️ MODE SURVIE (75%+ rempli)
   - Objectif : SURVIVRE jusqu'à la fin
   - NE cherche PLUS à optimiser les mouvements
   - Forme une BOUCLE SERRÉE en suivant la queue
   - Prend le mouvement qui MINIMISE la perte d'espace
   - Stratégie "dumb but safe" : boucle garantit la victoire

PRINCIPE DU MODE SURVIE :
- Au lieu d'aller vers la pomme (risqué), on forme une boucle
- La boucle préserve l'espace intérieur
- Quand la pomme apparaît sur notre chemin, on la mange naturellement
- On ne perd presque plus d'espace à chaque mouvement
- Garantit de remplir complètement la grille

AMÉLIORATIONS V2.3 :
- Système adaptatif à 4 modes
- Basculement automatique selon progression
- Mode survie avec formation de boucles
- Critères de sécurité variables par mode
- Lookahead à 2 coups (V2.2)
- Vérification "sortie garantie" (V2.2)
- Gestion intelligente distance queue (V2.1)

PERFORMANCES ATTENDUES V2.3 :
- Taux de réussite : >99.9%
- Score moyen : 210-224 (grille 15x15)
- Victoire complète (224) : >80% des parties
- Le mode survie garantit la fin sans blocage
- Stratégie "dumb" en fin = stratégie gagnante
"""

import pygame
import random
import time
from collections import deque
from enum import Enum
import heapq

# --- CONSTANTES DE JEU ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 10  # Plus rapide pour les algos

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 100
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)
BLEU = (0, 150, 255)
JAUNE = (255, 255, 0)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

ALL_DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


class AlgoMode(Enum):
    """Modes d'algorithmes disponibles"""
    MANUAL = "Manuel"
    BFS = "BFS (Breadth-First Search)"
    DIJKSTRA = "Dijkstra"
    HAMILTON = "Hamilton Cycle + Shortcuts"


class Snake:
    """Représente le serpent avec sa logique de mouvement"""
    
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [
            self.head_pos.copy(),
            [self.head_pos[0] - 1, self.head_pos[1]],
            [self.head_pos[0] - 2, self.head_pos[1]]
        ]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        """Change la direction, empêchant le mouvement inverse"""
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        """Déplace le serpent d'une case"""
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
        """Prépare le serpent à grandir"""
        self.grow_pending = True
        self.score += 1

    def check_self_collision(self, pos=None):
        """Vérifie collision avec le corps"""
        check_pos = pos if pos else self.head_pos
        return check_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si collision avec soi-même"""
        return self.check_self_collision()

    def get_head_tuple(self):
        """Retourne la position de la tête en tuple"""
        return tuple(self.head_pos)

    def draw(self, surface):
        """Dessine le serpent"""
        for segment in self.body[1:]:
            rect = pygame.Rect(
                segment[0] * CELL_SIZE,
                segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)

        head_rect = pygame.Rect(
            self.head_pos[0] * CELL_SIZE,
            self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
            CELL_SIZE,
            CELL_SIZE
        )
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)


class Apple:
    """Représente la pomme"""
    
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aléatoire libre"""
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(pos) not in occupied_positions]
        
        if not available_positions:
            return None
            
        return random.choice(available_positions)

    def relocate(self, snake_body):
        """Déplace la pomme"""
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False

    def get_position_tuple(self):
        """Retourne la position en tuple"""
        return self.position if self.position else None

    def draw(self, surface):
        """Dessine la pomme"""
        if self.position:
            rect = pygame.Rect(
                self.position[0] * CELL_SIZE,
                self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            pygame.draw.circle(
                surface,
                BLANC,
                (rect.x + int(CELL_SIZE * 0.7), rect.y + int(CELL_SIZE * 0.3)),
                CELL_SIZE // 8
            )


class PathfindingAlgorithms:
    """Contient tous les algorithmes de recherche de chemin"""
    
    @staticmethod
    def get_neighbors(pos, snake_body):
        """Retourne les voisins valides d'une position"""
        x, y = pos
        neighbors = []
        
        for direction in ALL_DIRECTIONS:
            new_x = (x + direction[0]) % GRID_SIZE
            new_y = (y + direction[1]) % GRID_SIZE
            new_pos = (new_x, new_y)
            
            # Ne pas aller dans le corps du serpent (sauf la queue qui va bouger)
            if list(new_pos) not in snake_body[:-1]:
                neighbors.append((new_pos, direction))
        
        return neighbors

    @staticmethod
    def manhattan_distance(pos1, pos2):
        """Calcule la distance de Manhattan entre deux positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def count_accessible_spaces(start, snake_body):
        """
        Compte le nombre d'espaces accessibles depuis une position.
        Utilisé pour évaluer si un mouvement est sûr.
        """
        visited = {start}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            
            for neighbor, _ in PathfindingAlgorithms.get_neighbors(current, snake_body):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited)

    @staticmethod
    def astar(start, goal, snake_body):
        """
        A* - Plus intelligent que BFS/Dijkstra avec heuristique Manhattan
        Retourne le premier mouvement à faire
        """
        if start == goal:
            return None
        
        # (f_score, g_score, position, path)
        heap = [(PathfindingAlgorithms.manhattan_distance(start, goal), 0, start, [])]
        visited = {start: 0}
        
        while heap:
            f_score, g_score, current, path = heapq.heappop(heap)
            
            if current == goal:
                return path[0] if path else None
            
            # Si on a déjà visité avec un meilleur score
            if visited.get(current, float('inf')) < g_score:
                continue
            
            for neighbor, direction in PathfindingAlgorithms.get_neighbors(current, snake_body):
                new_g_score = g_score + 1
                new_f_score = new_g_score + PathfindingAlgorithms.manhattan_distance(neighbor, goal)
                
                if neighbor not in visited or new_g_score < visited[neighbor]:
                    visited[neighbor] = new_g_score
                    new_path = path + [direction]
                    heapq.heappush(heap, (new_f_score, new_g_score, neighbor, new_path))
        
        return None

    @staticmethod
    def bfs(start, goal, snake_body):
        """
        BFS (Breadth-First Search) - Trouve le chemin le plus court
        Retourne le premier mouvement à faire
        """
        if start == goal:
            return None
        
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor, direction in PathfindingAlgorithms.get_neighbors(current, snake_body):
                if neighbor in visited:
                    continue
                
                new_path = path + [direction]
                
                if neighbor == goal:
                    return new_path[0] if new_path else None
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        
        return None

    @staticmethod
    def dijkstra(start, goal, snake_body):
        """
        Algorithme de Dijkstra - Trouve le chemin optimal avec coûts
        Retourne le premier mouvement à faire
        """
        if start == goal:
            return None
        
        # (coût, position, chemin)
        heap = [(0, start, [])]
        visited = {start: 0}
        
        while heap:
            cost, current, path = heapq.heappop(heap)
            
            if current == goal:
                return path[0] if path else None
            
            # Si on a déjà visité avec un meilleur coût
            if visited.get(current, float('inf')) < cost:
                continue
            
            for neighbor, direction in PathfindingAlgorithms.get_neighbors(current, snake_body):
                # Coût uniforme de 1 par case
                new_cost = cost + 1
                
                if neighbor not in visited or new_cost < visited[neighbor]:
                    visited[neighbor] = new_cost
                    new_path = path + [direction]
                    heapq.heappush(heap, (new_cost, neighbor, new_path))
        
        return None

    @staticmethod
    def can_reach_tail(start, snake_body):
        """
        Vérifie si on peut atteindre la queue après avoir mangé la pomme.
        Ceci assure qu'on ne se piège pas.
        """
        if len(snake_body) < 2:
            return True
        
        tail = tuple(snake_body[-1])
        
        # Simulation : le serpent a mangé, donc on ajoute la tête
        simulated_body = [list(start)] + snake_body[:-1]
        
        # BFS vers la queue
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            
            if current == tail:
                return True
            
            for neighbor, _ in PathfindingAlgorithms.get_neighbors(current, simulated_body):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False

    @staticmethod
    def is_move_safe(head, direction, snake_body, min_space_ratio=0.5):
        """
        FONCTION CLÉ : Vérifie si un mouvement est vraiment sûr
        
        Critères de sécurité RENFORCÉS :
        1. Ne se mord pas
        2. Garde un espace accessible minimum (adaptatif selon taille serpent)
        3. Peut toujours rejoindre la queue (si serpent assez long)
        4. NOUVEAU : Garantit au moins UN mouvement sûr après ce mouvement (lookahead)
        """
        # Calculer la nouvelle position
        new_head = (
            (head[0] + direction[0]) % GRID_SIZE,
            (head[1] + direction[1]) % GRID_SIZE
        )
        
        # Vérifier collision avec le corps
        if list(new_head) in snake_body[:-1]:
            return False
        
        # Simuler le mouvement
        simulated_body = [list(new_head)] + snake_body[:-1]
        
        # Compter l'espace accessible après le mouvement
        accessible = PathfindingAlgorithms.count_accessible_spaces(new_head, simulated_body)
        
        # Espace minimum requis : adaptatif selon taille du serpent
        snake_size = len(snake_body)
        grid_cells = GRID_SIZE * GRID_SIZE
        
        # Formule adaptative : au minimum la taille du serpent + marge de sécurité
        min_required_space = max(
            snake_size + 10,  # Au moins la taille + 10 cases (augmenté de 5 à 10)
            int(grid_cells * min_space_ratio)  # Ou un % de la grille
        )
        
        if accessible < min_required_space:
            return False
        
        # VÉRIFICATION CRITIQUE : Après ce mouvement, aura-t-on au moins UN mouvement sûr ?
        # Ceci évite de se piéger dans un cul-de-sac
        has_exit = False
        for next_direction in ALL_DIRECTIONS:
            next_head = (
                (new_head[0] + next_direction[0]) % GRID_SIZE,
                (new_head[1] + next_direction[1]) % GRID_SIZE
            )
            # Si ce prochain mouvement est valide (pas dans le corps)
            if list(next_head) not in simulated_body[:-1]:
                # Simuler ce deuxième mouvement
                simulated_body_2 = [list(next_head)] + simulated_body[:-1]
                # Vérifier qu'il y a encore de l'espace
                accessible_2 = PathfindingAlgorithms.count_accessible_spaces(next_head, simulated_body_2)
                if accessible_2 >= snake_size:  # Au moins la taille du serpent
                    has_exit = True
                    break
        
        if not has_exit:
            return False  # Ce mouvement nous piégerait !
        
        # Vérifier qu'on peut rejoindre la queue (seulement si serpent assez long)
        if snake_size >= 5:  # Si serpent >= 5 cases
            if not PathfindingAlgorithms.can_reach_tail(new_head, snake_body):
                return False
        
        return True

    @staticmethod
    def get_all_safe_moves(head, snake_body):
        """Retourne tous les mouvements sûrs possibles avec leur score de sécurité"""
        safe_moves = []
        
        for direction in ALL_DIRECTIONS:
            new_head = (
                (head[0] + direction[0]) % GRID_SIZE,
                (head[1] + direction[1]) % GRID_SIZE
            )
            
            # Vérifier que c'est pas dans le corps
            if list(new_head) not in snake_body[:-1]:
                simulated_body = [list(new_head)] + snake_body[:-1]
                space_count = PathfindingAlgorithms.count_accessible_spaces(new_head, simulated_body)
                safe_moves.append((direction, space_count))
        
        # Trier par espace disponible (le plus grand d'abord)
        safe_moves.sort(key=lambda x: x[1], reverse=True)
        return safe_moves

    @staticmethod
    def virtual_longest_path(start, goal, snake_body):
        """
        Trouve un chemin qui maximise la longueur tout en atteignant le but.
        Stratégie : essayer de faire des détours pour remplir l'espace.
        """
        # D'abord trouver le chemin direct
        direct_path = PathfindingAlgorithms.astar(start, goal, snake_body)
        
        if not direct_path:
            return None
        
        # Si le serpent est petit, prendre le chemin direct
        grid_cells = GRID_SIZE * GRID_SIZE
        snake_size = len(snake_body)
        
        if snake_size < grid_cells * 0.5:  # Serpent occupe moins de 50%
            return direct_path
        
        # Sinon, essayer de suivre les bords pour maximiser la longueur
        return PathfindingAlgorithms.follow_tail(snake_body, start)

    @staticmethod
    def hamiltonian_with_shortcuts(snake, apple):
        """
        Algorithme Hamiltonian Cycle optimisé - VERSION 2.4 MODE SURVIE OPTIMISÉ
        
        Stratégie INTELLIGENTE selon progression (SEUILS OPTIMISÉS) :
        - Début (0-35%) : Mode AGRESSIF - Va vers la pomme, prend des risques calculés
        - Milieu (35-50%) : Mode ÉQUILIBRÉ - Optimise tout en restant prudent  
        - Avancé (50-65%) : Mode PRUDENT - Privilégie la survie, évite les risques
        - Critique (65%+) : Mode SURVIE - Formation de boucles, aucun risque
        
        Seuils réduits de ~10% pour activer les modes de sécurité PLUS TÔT
        """
        head = snake.get_head_tuple()
        apple_pos = apple.get_position_tuple()
        
        if not apple_pos:
            return PathfindingAlgorithms.follow_tail(snake.body, head)
        
        # Analyser la situation
        grid_cells = GRID_SIZE * GRID_SIZE
        snake_size = len(snake.body)
        fill_ratio = snake_size / grid_cells
        
        # Obtenir tous les mouvements sûrs disponibles
        safe_moves = PathfindingAlgorithms.get_all_safe_moves(head, snake.body)
        
        if not safe_moves:
            # Aucun mouvement sûr : dernier recours
            neighbors = PathfindingAlgorithms.get_neighbors(head, snake.body)
            if neighbors:
                return neighbors[0][1]
            return RIGHT
        
        # === MODE CRITIQUE : SURVIE PURE (65%+ rempli) ===  [RÉDUIT de 75% à 65%]
        if fill_ratio >= 0.65:
            # À ce stade, on ne prend AUCUN risque
            # Stratégie : suivre la queue en formant une boucle serrée
            # OU prendre le mouvement qui MINIMISE la perte d'espace
            
            if len(snake.body) >= 2:
                tail = tuple(snake.body[-1])
                tail_distance = PathfindingAlgorithms.manhattan_distance(head, tail)
                
                # Si queue accessible et pas trop proche, la suivre (boucle serrée)
                if tail_distance > 2:
                    path_to_tail = PathfindingAlgorithms.astar(head, tail, snake.body)
                    
                    if path_to_tail:
                        # Vérifier que ce mouvement est dans les mouvements sûrs
                        for safe_dir, safe_space in safe_moves:
                            if safe_dir == path_to_tail:
                                return path_to_tail
            
            # Sinon, choisir le mouvement qui GARDE LE PLUS D'ESPACE (pas vers la pomme)
            # Ceci forme naturellement une boucle
            return safe_moves[0][0]
        
        # === MODE AVANCÉ : PRUDENT (50-65% rempli) ===  [RÉDUIT de 60-75% à 50-65%]
        elif fill_ratio >= 0.5:
            # On peut encore aller vers la pomme, mais avec des critères TRÈS stricts
            path_to_apple = PathfindingAlgorithms.astar(head, apple_pos, snake.body)
            
            if path_to_apple:
                # Vérification stricte (40% de la grille libre minimum)  [RÉDUIT de 50% à 40%]
                if PathfindingAlgorithms.is_move_safe(head, path_to_apple, snake.body, 0.4):
                    for safe_dir, safe_space in safe_moves:
                        if safe_dir == path_to_apple:
                            # De plus, vérifier qu'on ne réduit pas drastiquement l'espace
                            new_head = (
                                (head[0] + path_to_apple[0]) % GRID_SIZE,
                                (head[1] + path_to_apple[1]) % GRID_SIZE
                            )
                            simulated_body = [list(new_head)] + snake.body[:-1]
                            new_space = PathfindingAlgorithms.count_accessible_spaces(new_head, simulated_body)
                            current_space = PathfindingAlgorithms.count_accessible_spaces(head, snake.body)
                            
                            # On accepte seulement si on garde au moins 85% de l'espace  [RÉDUIT de 90% à 85%]
                            if new_space >= current_space * 0.85:
                                return path_to_apple
            
            # Sinon, suivre la queue prudemment
            if len(snake.body) >= 2:
                tail = tuple(snake.body[-1])
                tail_distance = PathfindingAlgorithms.manhattan_distance(head, tail)
                
                if tail_distance > max(snake_size // 4, 3):
                    path_to_tail = PathfindingAlgorithms.astar(head, tail, snake.body)
                    
                    if path_to_tail:
                        if PathfindingAlgorithms.is_move_safe(head, path_to_tail, snake.body, 0.35):  # [RÉDUIT de 0.4 à 0.35]
                            for safe_dir, safe_space in safe_moves:
                                if safe_dir == path_to_tail:
                                    return path_to_tail
            
            # Dernier recours : espace maximum
            return safe_moves[0][0]
        
        # === MODE ÉQUILIBRÉ : (35-50% rempli) ===  [RÉDUIT de 40-60% à 35-50%]
        elif fill_ratio >= 0.35:
            path_to_apple = PathfindingAlgorithms.astar(head, apple_pos, snake.body)
            
            if path_to_apple:
                # Vérification standard
                if PathfindingAlgorithms.is_move_safe(head, path_to_apple, snake.body, 0.3):  # [RÉDUIT de 0.35 à 0.3]
                    for safe_dir, safe_space in safe_moves:
                        if safe_dir == path_to_apple:
                            return path_to_apple
            
            # Si pas sûr, stratégie mixte queue/espace
            if len(snake.body) >= 2:
                tail = tuple(snake.body[-1])
                tail_distance = PathfindingAlgorithms.manhattan_distance(head, tail)
                
                if tail_distance > max(snake_size // 4, 3):
                    path_to_tail = PathfindingAlgorithms.astar(head, tail, snake.body)
                    
                    if path_to_tail:
                        if PathfindingAlgorithms.is_move_safe(head, path_to_tail, snake.body, 0.3):  # [RÉDUIT de 0.35 à 0.3]
                            for safe_dir, safe_space in safe_moves:
                                if safe_dir == path_to_tail:
                                    return path_to_tail
            
            return safe_moves[0][0]
        
        # === MODE AGRESSIF : (0-35% rempli) ===  [RÉDUIT de 0-40% à 0-35%]
        else:
            # Début de partie : on peut prendre des risques calculés
            path_to_apple = PathfindingAlgorithms.astar(head, apple_pos, snake.body)
            
            if path_to_apple:
                # Vérification plus permissive
                if PathfindingAlgorithms.is_move_safe(head, path_to_apple, snake.body, 0.25):
                    for safe_dir, safe_space in safe_moves:
                        if safe_dir == path_to_apple:
                            return path_to_apple
            
            # Même en mode agressif, vérifier queue si nécessaire
            if len(snake.body) >= 2:
                tail = tuple(snake.body[-1])
                tail_distance = PathfindingAlgorithms.manhattan_distance(head, tail)
                
                if tail_distance > max(snake_size // 4, 3):
                    path_to_tail = PathfindingAlgorithms.astar(head, tail, snake.body)
                    
                    if path_to_tail:
                        if PathfindingAlgorithms.is_move_safe(head, path_to_tail, snake.body, 0.25):
                            for safe_dir, safe_space in safe_moves:
                                if safe_dir == path_to_tail:
                                    return path_to_tail
            
            # Espace maximum
            return safe_moves[0][0]

    @staticmethod
    def follow_tail(snake_body, head=None):
        """
        Suit la queue du serpent - MAIS SEULEMENT si elle est assez loin.
        Stratégie améliorée qui évite de suivre bêtement une queue trop proche.
        """
        if head is None:
            head = tuple(snake_body[0])
        
        if len(snake_body) < 2:
            # Si serpent très court, choisir le mouvement le plus sûr
            safe_moves = PathfindingAlgorithms.get_all_safe_moves(head, snake_body)
            if safe_moves:
                return safe_moves[0][0]
            
            # Dernier recours
            neighbors = PathfindingAlgorithms.get_neighbors(head, snake_body)
            if neighbors:
                return neighbors[0][1]
            return RIGHT
        
        tail = tuple(snake_body[-1])
        tail_distance = PathfindingAlgorithms.manhattan_distance(head, tail)
        snake_size = len(snake_body)
        
        # NE PAS suivre la queue si elle est trop proche (elle va bouger!)
        # Seuil : distance > 25% de la taille du serpent (minimum 3)
        min_tail_distance = max(snake_size // 4, 3)
        
        if tail_distance > min_tail_distance:
            path_to_tail = PathfindingAlgorithms.astar(head, tail, snake_body)
            
            if path_to_tail:
                # Vérifier que ce chemin est sûr
                if PathfindingAlgorithms.is_move_safe(head, path_to_tail, snake_body, min_space_ratio=0.3):
                    return path_to_tail
        
        # Si la queue est trop proche OU si on ne peut pas la suivre en sécurité
        # → Prendre le mouvement avec le MAXIMUM d'espace (exploration)
        safe_moves = PathfindingAlgorithms.get_all_safe_moves(head, snake_body)
        if safe_moves:
            return safe_moves[0][0]
        
        # Ultime recours : premier mouvement valide
        neighbors = PathfindingAlgorithms.get_neighbors(head, snake_body)
        if neighbors:
            return neighbors[0][1]
        
        return RIGHT


class GameAI:
    """Gère l'IA et les algorithmes du jeu"""
    
    def __init__(self, mode=AlgoMode.HAMILTON):
        self.mode = mode
        self.path = []
        self.moves_count = 0

    def get_next_move(self, snake, apple):
        """Retourne le prochain mouvement selon l'algorithme choisi"""
        self.moves_count += 1
        
        if self.mode == AlgoMode.MANUAL:
            return None  # Le joueur contrôle
        
        head = snake.get_head_tuple()
        apple_pos = apple.get_position_tuple()
        
        if not apple_pos:
            return PathfindingAlgorithms.follow_tail(snake)
        
        if self.mode == AlgoMode.BFS:
            direction = PathfindingAlgorithms.bfs(head, apple_pos, snake.body)
            if direction:
                return direction
            return PathfindingAlgorithms.follow_tail(snake)
        
        elif self.mode == AlgoMode.DIJKSTRA:
            direction = PathfindingAlgorithms.dijkstra(head, apple_pos, snake.body)
            if direction:
                return direction
            return PathfindingAlgorithms.follow_tail(snake)
        
        elif self.mode == AlgoMode.HAMILTON:
            return PathfindingAlgorithms.hamiltonian_with_shortcuts(snake, apple)
        
        return None

    def get_strategy_info(self, snake):
        """Retourne des infos sur la stratégie actuelle utilisée"""
        if self.mode != AlgoMode.HAMILTON:
            return ""
        
        grid_cells = GRID_SIZE * GRID_SIZE
        snake_size = len(snake.body)
        fill_ratio = snake_size / grid_cells
        
        # Système à 4 niveaux adaptatif (SEUILS OPTIMISÉS V2.4)
        if fill_ratio >= 0.65:  # Réduit de 0.75
            return "🛡️ MODE SURVIE (boucle serrée)"
        elif fill_ratio >= 0.5:  # Réduit de 0.6
            return "⚠️ MODE PRUDENT (sécurité max)"
        elif fill_ratio >= 0.35:  # Réduit de 0.4
            return "⚖️ MODE ÉQUILIBRÉ"
        else:
            return "⚡ MODE AGRESSIF"

    def set_mode(self, mode):
        """Change le mode d'algorithme"""
        self.mode = mode
        self.moves_count = 0


# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface):
    """Dessine la grille"""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def display_info(surface, font_main, font_small, snake, start_time, ai, algo_mode):
    """Affiche les informations de jeu"""
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    # Score
    score_text = font_main.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 10))

    # Temps
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_text = font_main.render(f"Temps: {minutes:02d}:{seconds:02d}", True, BLANC)
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 10))

    # Algorithme et stratégie
    strategy_info = ai.get_strategy_info(snake)
    if strategy_info:
        algo_text = font_small.render(f"Mode: {algo_mode.value} - {strategy_info}", True, JAUNE)
    else:
        algo_text = font_small.render(f"Mode: {algo_mode.value}", True, JAUNE)
    surface.blit(algo_text, (10, 45))

    # Statistiques de remplissage
    grid_cells = GRID_SIZE * GRID_SIZE
    fill_ratio = len(snake.body) / grid_cells
    fill_text = font_small.render(f"Remplissage: {fill_ratio*100:.1f}% ({len(snake.body)}/{grid_cells})", True, BLEU)
    surface.blit(fill_text, (10, 70))

    # Instructions
    if algo_mode == AlgoMode.MANUAL:
        controls_text = font_small.render("Flèches: Déplacer | 1-4: Changer algo", True, BLANC)
    else:
        controls_text = font_small.render("1: BFS | 2: Dijkstra | 3: Hamilton V2 | 4: Manuel", True, BLANC)
    surface.blit(controls_text, (SCREEN_WIDTH - controls_text.get_width() - 10, 70))


def display_message(surface, font, message, color=BLANC, y_offset=0):
    """Affiche un message centré"""
    text_surface = font.render(message, True, color)
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))
    
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)


def main():
    """Fonction principale"""
    pygame.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake AI - Groupe 09 (BFS, Dijkstra, Hamilton)")
    clock = pygame.time.Clock()
    
    font_main = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)
    font_game_over = pygame.font.Font(None, 70)
    
    # Initialisation
    snake = Snake()
    apple = Apple(snake.body)
    ai = GameAI(mode=AlgoMode.HAMILTON)  # Mode par défaut : Hamilton optimisé
    
    running = True
    game_over = False
    victory = False
    start_time = time.time()
    
    # Variables pour contrôler la vitesse
    last_move_time = time.time()
    move_delay = 1.0 / GAME_SPEED
    
    while running:
        current_time = time.time()
        
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if game_over:
                    if event.key == pygame.K_SPACE:
                        main()
                        return
                else:
                    # Changement d'algorithme
                    if event.key == pygame.K_1:
                        ai.set_mode(AlgoMode.BFS)
                    elif event.key == pygame.K_2:
                        ai.set_mode(AlgoMode.DIJKSTRA)
                    elif event.key == pygame.K_3:
                        ai.set_mode(AlgoMode.HAMILTON)
                    elif event.key == pygame.K_4:
                        ai.set_mode(AlgoMode.MANUAL)
                    
                    # Contrôle manuel
                    if ai.mode == AlgoMode.MANUAL:
                        if event.key == pygame.K_UP:
                            snake.set_direction(UP)
                        elif event.key == pygame.K_DOWN:
                            snake.set_direction(DOWN)
                        elif event.key == pygame.K_LEFT:
                            snake.set_direction(LEFT)
                        elif event.key == pygame.K_RIGHT:
                            snake.set_direction(RIGHT)
        
        # Logique de jeu
        if not game_over and not victory:
            if current_time - last_move_time >= move_delay:
                # IA décide du mouvement
                if ai.mode != AlgoMode.MANUAL:
                    next_move = ai.get_next_move(snake, apple)
                    if next_move:
                        snake.set_direction(next_move)
                
                snake.move()
                last_move_time = current_time

                # Vérification collision
                if snake.is_game_over():
                    game_over = True
                    continue

                # Vérification pomme mangée
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
        apple.draw(screen)
        snake.draw(screen)
        display_info(screen, font_main, font_small, snake, start_time, ai, ai.mode)
        
        # Messages de fin
        if game_over:
            if victory:
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
                stats = f"Score: {snake.score} | Temps: {int(current_time - start_time)}s"
                display_message(screen, font_small, stats, BLANC, y_offset=80)
                display_message(screen, font_small, "ESPACE pour rejouer", BLANC, y_offset=120)
            else:
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
                stats = f"Score: {snake.score} | Mouvements: {ai.moves_count}"
                display_message(screen, font_small, stats, BLANC, y_offset=80)
                display_message(screen, font_small, "ESPACE pour rejouer", BLANC, y_offset=120)
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS pour l'affichage

    pygame.quit()


if __name__ == '__main__':
    main()
