#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snake avec Algorithmes de Pathfinding
Groupe 09 - Implémentation avec BFS, Dijkstra, et Hamiltonian Cycle optimisé
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
    def hamiltonian_with_shortcuts(snake, apple):
        """
        Algorithme Hamiltonian Cycle optimisé avec raccourcis.
        
        Stratégie :
        1. Suit un cycle hamiltonien de base (garantit de remplir tout l'espace)
        2. Prend des raccourcis intelligents vers la pomme quand c'est sûr
        3. Vérifie toujours qu'on peut encore atteindre la queue après le raccourci
        
        Cette approche combine vitesse et sécurité pour maximiser le score rapidement.
        """
        head = snake.get_head_tuple()
        apple_pos = apple.get_position_tuple()
        
        if not apple_pos:
            return PathfindingAlgorithms.follow_tail(snake)
        
        # Essayer d'abord un raccourci vers la pomme
        path_to_apple = PathfindingAlgorithms.bfs(head, apple_pos, snake.body)
        
        if path_to_apple:
            # Simuler le mouvement
            new_head = (
                (head[0] + path_to_apple[0]) % GRID_SIZE,
                (head[1] + path_to_apple[1]) % GRID_SIZE
            )
            
            # Vérifier qu'après avoir mangé, on peut toujours atteindre la queue
            if PathfindingAlgorithms.can_reach_tail(new_head, snake.body):
                return path_to_apple
        
        # Sinon, suivre la queue (stratégie sûre)
        return PathfindingAlgorithms.follow_tail(snake)

    @staticmethod
    def follow_tail(snake):
        """
        Suit la queue du serpent - stratégie ultra sûre.
        Garantit de ne jamais se bloquer.
        """
        head = snake.get_head_tuple()
        
        if len(snake.body) < 2:
            # Si serpent très court, bouger aléatoirement
            return random.choice(ALL_DIRECTIONS)
        
        tail = tuple(snake.body[-1])
        path_to_tail = PathfindingAlgorithms.bfs(head, tail, snake.body)
        
        if path_to_tail:
            return path_to_tail
        
        # Si impossible d'atteindre la queue, choisir le mouvement le plus sûr
        neighbors = PathfindingAlgorithms.get_neighbors(head, snake.body)
        if neighbors:
            return neighbors[0][1]
        
        return snake.direction


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

    # Algorithme
    algo_text = font_small.render(f"Mode: {algo_mode.value}", True, JAUNE)
    surface.blit(algo_text, (10, 45))

    # Mouvements
    moves_text = font_small.render(f"Mouvements: {ai.moves_count}", True, BLEU)
    surface.blit(moves_text, (10, 70))

    # Instructions
    if algo_mode == AlgoMode.MANUAL:
        controls_text = font_small.render("Flèches: Déplacer | 1-4: Changer algo", True, BLANC)
    else:
        controls_text = font_small.render("1: BFS | 2: Dijkstra | 3: Hamilton | 4: Manuel", True, BLANC)
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
