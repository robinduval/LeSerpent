"""
SNAKE-ALGO.PY - Version Algorithmique
Utilise A* avec safety check pour pathfinding optimal
Réutilise le jeu de base de serpent.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serpent import Snake, Apple, GRID_SIZE, CELL_SIZE, SCORE_PANEL_HEIGHT
from serpent import SCREEN_WIDTH, SCREEN_HEIGHT, BLANC, NOIR, GRIS_FOND
from serpent import UP, DOWN, LEFT, RIGHT
from serpent import draw_grid, display_info, display_message

import pygame
import heapq
import time
import json


class AStarPathfinder:
    """Algorithme A* avec safety check pour éviter l'auto-piégeage"""

    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size

    def manhattan_distance(self, pos1, pos2):
        """Distance de Manhattan entre deux positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def find_path(self, start, goal, obstacles):
        """
        A* pathfinding
        Returns: list of positions representing the path, or None if no path
        """
        # Priority queue: (f_score, counter, position, path)
        counter = 0
        heap = [(0, counter, tuple(start), [tuple(start)])]
        visited = set()

        while heap:
            f_score, _, current, path = heapq.heappop(heap)

            if current == tuple(goal):
                return [list(p) for p in path]

            if current in visited:
                continue

            visited.add(current)

            # Explore neighbors (4 directions)
            for direction in [UP, DOWN, LEFT, RIGHT]:
                next_x = (current[0] + direction[0]) % self.grid_size
                next_y = (current[1] + direction[1]) % self.grid_size
                next_pos = (next_x, next_y)

                # Check if valid
                if next_pos in obstacles or next_pos in visited:
                    continue

                g_score = len(path)  # Distance from start
                h_score = self.manhattan_distance(next_pos, goal)  # Heuristic to goal
                f_score = g_score + h_score

                counter += 1
                heapq.heappush(heap, (f_score, counter, next_pos, path + [next_pos]))

        return None  # No path found

    def is_safe_move(self, next_head, snake_body):
        """
        Vérifie si le mouvement est sûr en testant si on peut atteindre notre queue après
        C'est le "safety check" qui évite l'auto-piégeage
        """
        tail = tuple(snake_body[-1])

        # Obstacles = corps sans la queue (elle va bouger)
        obstacles = set(tuple(segment) for segment in snake_body[:-1])

        # Peut-on atteindre la queue depuis next_head ?
        escape_path = self.find_path(next_head, tail, obstacles)

        return escape_path is not None

    def get_best_direction(self, snake, apple):
        """
        Retourne la meilleure direction à prendre
        Utilise A* avec safety check
        """
        head = snake.head_pos
        food = list(apple.position)

        # Obstacles = corps du serpent (sauf la queue qui va bouger)
        obstacles = set(tuple(segment) for segment in snake.body[:-1])

        # 1. Trouver le chemin vers la nourriture
        path_to_food = self.find_path(head, food, obstacles)

        if path_to_food and len(path_to_food) > 1:
            next_point = path_to_food[1]  # Premier pas du chemin

            # 2. Safety check: ce mouvement est-il sûr ?
            if self.is_safe_move(next_point, snake.body):
                # Le chemin est sûr, suivons-le
                return self._get_direction_to_point(head, next_point)

        # 3. Si pas de chemin sûr vers la nourriture, suivre notre queue
        tail = snake.body[-1]
        path_to_tail = self.find_path(head, tail, set(tuple(s) for s in snake.body[:-1]))

        if path_to_tail and len(path_to_tail) > 1:
            next_point = path_to_tail[1]
            return self._get_direction_to_point(head, next_point)

        # 4. Dernier recours: prendre n'importe quelle direction valide
        for direction in [UP, DOWN, LEFT, RIGHT]:
            next_x = (head[0] + direction[0]) % self.grid_size
            next_y = (head[1] + direction[1]) % self.grid_size
            if [next_x, next_y] not in snake.body[1:]:
                return direction

        # Vraiment coincé, retourner la direction actuelle
        return snake.direction

    def _get_direction_to_point(self, from_point, to_point):
        """Détermine la direction pour aller de from_point à to_point"""
        dx = (to_point[0] - from_point[0]) % self.grid_size
        dy = (to_point[1] - from_point[1]) % self.grid_size

        # Handle wrapping
        if dx == self.grid_size - 1:
            dx = -1
        if dy == self.grid_size - 1:
            dy = -1

        if dx == 1:
            return RIGHT
        elif dx == -1:
            return LEFT
        elif dy == 1:
            return DOWN
        elif dy == -1:
            return UP

        return snake.direction


def main():
    """Point d'entrée principal pour la version algorithmique"""
    pygame.init()

    # Configuration de l'écran
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake Algorithmique - A* avec Safety Check')
    clock = pygame.time.Clock()

    # Polices
    font_main = pygame.font.Font(None, 40)

    # Pathfinder
    pathfinder = AStarPathfinder()

    # Métriques
    games_played = 0
    total_score = 0
    max_score = 0
    total_moves = 0

    print("=" * 60)
    print("SNAKE ALGORITHMIQUE - A* avec Safety Check")
    print("=" * 60)
    print("Appuyez sur Ctrl+C pour arrêter\n")

    try:
        while True:
            # Initialisation du jeu
            snake = Snake()
            apple = Apple(snake.body)
            start_time = time.time()

            game_over = False
            victory = False
            frame_count = 0

            # Boucle de jeu
            running = True
            while running and not game_over:
                # Gestion des événements
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Get best direction from algorithm
                direction = pathfinder.get_best_direction(snake, apple)
                snake.set_direction(direction)

                # Move snake
                snake.move()
                frame_count += 1
                total_moves += 1

                # Check collisions
                if snake.check_self_collision():  # Wall collision disabled (wrapping)
                    game_over = True
                    continue

                # Check apple eaten
                if snake.head_pos == list(apple.position):
                    snake.grow()
                    if not apple.relocate(snake.body):
                        victory = True
                        game_over = True

                # Timeout protection
                if frame_count > 1000:
                    game_over = True

                # Draw
                screen.fill(GRIS_FOND)
                game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
                pygame.draw.rect(screen, NOIR, game_area_rect)
                draw_grid(screen)
                apple.draw(screen)
                snake.draw(screen)
                display_info(screen, font_main, snake, start_time)

                # Add game counter
                game_info = font_main.render(f"Games: {games_played} | Max: {max_score}", True, BLANC)
                screen.blit(game_info, (10, 50))

                pygame.display.flip()
                clock.tick(20)  # 20 FPS for visibility

            # Game ended
            games_played += 1
            total_score += snake.score
            max_score = max(max_score, snake.score)

            avg_score = total_score / games_played if games_played > 0 else 0

            status = "VICTORY" if victory else "GAME OVER"
            print(f"Game {games_played:3d} | {status:10s} | Score: {snake.score:3d} | "
                  f"Max: {max_score:3d} | Avg: {avg_score:.2f} | Moves: {frame_count}")

            # Save metrics every 10 games
            if games_played % 10 == 0:
                save_metrics(games_played, max_score, avg_score, total_moves)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print(f"Arrêt du programme")
        print(f"Total games: {games_played}")
        print(f"Max score: {max_score}")
        print(f"Average score: {total_score / games_played if games_played > 0 else 0:.2f}")
        print("=" * 60)
        save_metrics(games_played, max_score, total_score / games_played if games_played > 0 else 0, total_moves)
        pygame.quit()


def save_metrics(games, max_score, avg_score, total_moves):
    """Sauvegarde les métriques dans un fichier JSON"""
    metrics = {
        'type': 'algorithm',
        'games_played': games,
        'max_score': max_score,
        'average_score': avg_score,
        'total_moves': total_moves
    }

    with open('results/algo_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
