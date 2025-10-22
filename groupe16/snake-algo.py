# snake-algo.py
# Résolution du Snake par plus court chemin (BFS)
# Objectif : faire bouger le serpent automatiquement jusqu’à la pomme

import sys, os
import pygame
import time
from collections import deque
from serpent import (
    GRID_SIZE, CELL_SIZE, GAME_SPEED,
    SCREEN_WIDTH, SCREEN_HEIGHT, SCORE_PANEL_HEIGHT,
    BLANC, NOIR, VERT, ROUGE, GRIS_FOND, GRIS_GRILLE,
    UP, DOWN, LEFT, RIGHT,
    Snake, Apple,
    draw_grid, display_info, display_message
)

BLEU_CHEMIN = (0, 150, 255)


# --- OUTILS ---

def add_pos(p1, p2):
    """Ajoute deux positions modulo la taille de la grille."""
    return [(p1[0] + p2[0]) % GRID_SIZE, (p1[1] + p2[1]) % GRID_SIZE]

def neighbors(pos):
    """Retourne les 4 voisins possibles (haut, bas, gauche, droite)."""
    dirs = [UP, DOWN, LEFT, RIGHT]
    return [add_pos(pos, d) for d in dirs]

def bfs_path(start, goal, body):
    """
    BFS amélioré : simule le mouvement du serpent pour éviter de se mordre.
    Chaque état garde la position de la tête + du corps à ce tour.
    """
    queue = deque([(start, list(body))])  # (position_tête, corps_complet)
    parents = {tuple(start): None}
    visited = {tuple(start)}

    while queue:
        head, body_state = queue.popleft()

        # Si on a atteint la pomme → reconstruit le chemin
        if head == goal:
            # reconstruction simple via parents
            path = [goal]
            while parents[tuple(path[-1])] is not None:
                path.append(parents[tuple(path[-1])])
            path.reverse()
            return path

        # On simule tous les mouvements possibles
        for direction in [UP, DOWN, LEFT, RIGHT]:
            new_head = [(head[0] + direction[0]) % GRID_SIZE, (head[1] + direction[1]) % GRID_SIZE]

            # Le serpent avance : la tête entre, la queue sort
            new_body = [new_head] + body_state[:-1]

            # Collision avec le corps (si la tête touche le corps)
            if new_head in body_state[:-1]:
                continue

            # Enregistre si non visité
            if tuple(new_head) not in visited:
                visited.add(tuple(new_head))
                parents[tuple(new_head)] = head
                queue.append((new_head, new_body))

    # Aucun chemin trouvé
    return []


def direction_from_to(p1, p2):
    """Renvoie la direction à suivre pour passer de p1 à p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx == 1 or dx < -1:  # droite
        return RIGHT
    if dx == -1 or dx > 1:  # gauche
        return LEFT
    if dy == 1 or dy < -1:  # bas
        return DOWN
    if dy == -1 or dy > 1:  # haut
        return UP
    return (0, 0)

    def is_safe_path_to_apple(snake, apple):
        # On simule le corps après avoir suivi le chemin BFS vers la pomme
        path_to_apple = bfs_path(snake.head_pos, apple.position, snake.body)
        if len(path_to_apple) <= 1:
            return False  # pas de chemin vers la pomme

        # Simule le serpent après avoir mangé
        simulated_body = [list(p) for p in snake.body]
        for step in path_to_apple[1:]:
            simulated_body.insert(0, list(step))
            simulated_body.pop()  # queue bouge
        simulated_body.insert(0, path_to_apple[-1])  # tête finale
        # (optionnel : grow = True → garde la queue)

        # Vérifie si, après ça, la tête peut encore atteindre la queue
        new_head = simulated_body[0]
        new_tail = simulated_body[-1]
        return len(bfs_path(new_head, new_tail, simulated_body)) > 0


# --- ALGO PRINCIPAL ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake - BFS + stratégie de survie (follow-tail)")
    clock = pygame.time.Clock()

    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    snake = Snake()
    apple = Apple(snake.body)

    running = True
    game_over = False
    victory = False
    start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over and not victory:
            start = snake.head_pos
            goal = list(apple.position)

            # --- 1️⃣ Chemin direct vers la pomme ---
            path_to_apple = bfs_path(start, goal, snake.body)

            # --- 2️⃣ Chemin vers la queue (sécurité) ---
            tail = snake.body[-1]
            path_to_tail = bfs_path(start, tail, snake.body)

            next_pos = None

            # Cas 1 : il existe un chemin vers la pomme ET,
            # une fois mangée, on peut encore atteindre la queue
            if len(path_to_apple) > 1:
                # Simule le serpent après avoir suivi le chemin vers la pomme
                simulated_body = [list(p) for p in snake.body]
                for step in path_to_apple[1:]:
                    simulated_body.insert(0, list(step))
                    simulated_body.pop()  # la queue bouge

                # BFS entre tête simulée et queue simulée
                sim_head = simulated_body[0]
                sim_tail = simulated_body[-1]
                path_after = bfs_path(sim_head, sim_tail, simulated_body)

                if len(path_after) > 0:
                    # C’est un chemin “safe” : on fonce sur la pomme
                    next_pos = path_to_apple[1]

            # Cas 2 : sinon, suivre la queue (safe move)
            if next_pos is None and len(path_to_tail) > 1:
                next_pos = path_to_tail[1]

            # Cas 3 : sinon, bouger librement (dernière chance)
            if next_pos is None:
                for d in [UP, DOWN, LEFT, RIGHT]:
                    cand = [(start[0] + d[0]) % GRID_SIZE, (start[1] + d[1]) % GRID_SIZE]
                    if cand not in snake.body[:-1]:
                        next_pos = cand
                        break

            # Applique la direction
            if next_pos:
                new_dir = direction_from_to(start, next_pos)
                snake.set_direction(new_dir)

            # --- Déplacement ---
            snake.move()

            # --- Vérifie collisions ---
            if snake.is_game_over():
                game_over = True

            # --- Mange la pomme ---
            if not game_over and snake.head_pos == list(apple.position):
                snake.grow()
                if not apple.relocate(snake.body):
                    victory = True
                    game_over = True

        # --- DESSIN ---
        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        draw_grid(screen)

        # Affiche le chemin courant (rouge contour)
        if 'path_to_apple' in locals() and len(path_to_apple) > 1:
            for step in path_to_apple[1:]:
                rect = pygame.Rect(
                    step[0] * CELL_SIZE,
                    step[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                    CELL_SIZE,
                    CELL_SIZE
                )
                pygame.draw.rect(screen, (200, 0, 0), rect, width=3)

        apple.draw(screen)
        snake.draw(screen)
        display_info(screen, font_main, snake, start_time)

        if game_over:
            color = VERT if victory else ROUGE
            text = "VICTOIRE !" if victory else "GAME OVER"
            display_message(screen, font_game_over, text, color)
            display_message(screen, font_main, "R pour rejouer | ESC pour quitter", BLANC, y_offset=100)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                snake = Snake()
                apple = Apple(snake.body)
                game_over = False
                victory = False
                start_time = time.time()
            if keys[pygame.K_ESCAPE]:
                running = False

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()
