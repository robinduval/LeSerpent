"""
SNAKE-IA.PY - Version Reinforcement Learning
Utilise Deep Q-Learning pour apprendre à jouer au Snake
Réutilise le jeu de base de serpent.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serpent import Snake, Apple, GRID_SIZE, CELL_SIZE, SCORE_PANEL_HEIGHT
from serpent import SCREEN_WIDTH, SCREEN_HEIGHT, BLANC, NOIR, GRIS_FOND
from serpent import UP, DOWN, LEFT, RIGHT
from serpent import draw_grid, display_info

import pygame
import time
import json
import numpy as np
from agent import Agent

# Mode headless (sans GUI) pour training ultra-rapide
HEADLESS = True  # Mettre à False pour voir le jeu

# Import matplotlib seulement si GUI activé (économise de la RAM)
if not HEADLESS:
    import matplotlib.pyplot as plt
    plt.ion()

# Speed pour training rapide
TRAINING_SPEED = 1000  # FPS très rapide (si GUI activé)


def get_state(snake, apple):
    """
    État SIMPLE en 11 features (version qui marche):
    - 3 dangers (devant, droite, gauche)
    - 4 directions (one-hot)
    - 4 positions pomme (4 bools)
    """
    head = snake.head_pos

    # Points dans chaque direction
    point_l = [(head[0] - 1) % GRID_SIZE, head[1]]
    point_r = [(head[0] + 1) % GRID_SIZE, head[1]]
    point_u = [head[0], (head[1] - 1) % GRID_SIZE]
    point_d = [head[0], (head[1] + 1) % GRID_SIZE]

    # Directions actuelles
    dir_l = snake.direction == LEFT
    dir_r = snake.direction == RIGHT
    dir_u = snake.direction == UP
    dir_d = snake.direction == DOWN

    state = [
        # Danger straight
        (dir_r and point_r in snake.body[1:]) or
        (dir_l and point_l in snake.body[1:]) or
        (dir_u and point_u in snake.body[1:]) or
        (dir_d and point_d in snake.body[1:]),

        # Danger right
        (dir_u and point_r in snake.body[1:]) or
        (dir_d and point_l in snake.body[1:]) or
        (dir_l and point_u in snake.body[1:]) or
        (dir_r and point_d in snake.body[1:]),

        # Danger left
        (dir_d and point_r in snake.body[1:]) or
        (dir_u and point_l in snake.body[1:]) or
        (dir_r and point_u in snake.body[1:]) or
        (dir_l and point_d in snake.body[1:]),

        # Move direction (one-hot encoding)
        dir_l,
        dir_r,
        dir_u,
        dir_d,

        # Food location relative to head
        apple.position[0] < head[0],  # food left
        apple.position[0] > head[0],  # food right
        apple.position[1] < head[1],  # food up
        apple.position[1] > head[1]   # food down
    ]

    return np.array(state, dtype=int)


def action_to_direction(action, current_direction):
    """
    Convertit une action RL [straight, right, left] en direction absolue
    action: [1,0,0] = straight, [0,1,0] = right turn, [0,0,1] = left turn
    """
    clock_wise = [RIGHT, DOWN, LEFT, UP]
    idx = clock_wise.index(current_direction)

    if np.array_equal(action, [1, 0, 0]):
        new_dir = clock_wise[idx]  # no change (straight)
    elif np.array_equal(action, [0, 1, 0]):
        next_idx = (idx + 1) % 4
        new_dir = clock_wise[next_idx]  # right turn
    else:  # [0, 0, 1]
        next_idx = (idx - 1) % 4
        new_dir = clock_wise[next_idx]  # left turn

    return new_dir


def plot_results(scores, mean_scores):
    """Affiche les graphiques de progression"""
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', alpha=0.6)
    plt.plot(mean_scores, label='Mean Score', linewidth=2)
    plt.ylim(ymin=0)
    plt.legend()
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if len(mean_scores) > 0:
        plt.text(len(mean_scores)-1, mean_scores[-1], f'{mean_scores[-1]:.1f}')
    plt.pause(0.1)


def train():
    """Boucle d'entraînement principale"""

    # Initialisation pygame seulement si GUI activé
    if not HEADLESS:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Snake IA - Deep Q-Learning Training')
        clock = pygame.time.Clock()
        font_main = pygame.font.Font(None, 40)
    else:
        screen = None
        clock = None
        font_main = None

    # Métriques
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # Agent RL
    agent = Agent()

    print("=" * 60)
    print("SNAKE IA - Deep Q-Learning Training")
    print("=" * 60)
    if HEADLESS:
        print("Mode: HEADLESS (ultra-rapide)")
    else:
        print("Mode: GUI (visualisation)")
    print("L'agent va apprendre à jouer par essai-erreur")
    print("Attendez ~100 games pour voir les premiers progrès")
    print("Appuyez sur Ctrl+C pour arrêter\n")

    # Timer pour mesurer la vitesse
    start_training_time = time.time()

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
            while not game_over:
                # Gestion des événements (seulement si GUI)
                if not HEADLESS:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return

                # Get current state
                state_old = get_state(snake, apple)

                # Get action from agent
                final_move = agent.get_action(state_old)

                # Convert action to direction and apply
                direction = action_to_direction(final_move, snake.direction)
                snake.set_direction(direction)

                # Move snake
                snake.move()
                frame_count += 1

                # Calculate reward (selon le sujet)
                reward = 0.05  # Récompense de base pour se déplacer

                # Check collisions
                if snake.check_self_collision():  # Wall collision disabled
                    game_over = True
                    reward = -15  # Perdre
                elif snake.head_pos == list(apple.position):
                    # Apple eaten
                    snake.grow()
                    reward = 10  # Attraper la pomme
                    if not apple.relocate(snake.body):
                        victory = True
                        game_over = True
                        reward = 100  # Finir le jeu (victoire)

                # Timeout protection
                if frame_count > 100 * len(snake.body):
                    game_over = True
                    reward = -10

                # Get new state
                state_new = get_state(snake, apple)

                # Train short memory (immediate learning)
                agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

                # Remember
                agent.remember(state_old, final_move, reward, state_new, game_over)

                # Draw (seulement si GUI activé)
                if not HEADLESS and frame_count % 5 == 0:
                    screen.fill(GRIS_FOND)
                    game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
                    pygame.draw.rect(screen, NOIR, game_area_rect)
                    draw_grid(screen)
                    apple.draw(screen)
                    snake.draw(screen)
                    display_info(screen, font_main, snake, start_time)

                    # Add training info
                    training_info = font_main.render(
                        f"Game: {agent.n_games} | Record: {record} | Epsilon: {agent.epsilon}",
                        True, BLANC
                    )
                    screen.blit(training_info, (10, 50))

                    pygame.display.flip()

                # Limite FPS seulement si GUI activé
                if not HEADLESS:
                    clock.tick(TRAINING_SPEED)

            # Game ended - train long memory (experience replay)
            agent.n_games += 1
            agent.train_long_memory()

            if snake.score > record:
                record = snake.score
                agent.model.save()

            # Log results (print tous les 10 games en mode headless pour accélérer)
            if not HEADLESS or agent.n_games % 10 == 0:
                status = "VICTORY" if victory else "GAME OVER"

                # Afficher la vitesse tous les 50 games
                if agent.n_games % 50 == 0:
                    elapsed = time.time() - start_training_time
                    games_per_sec = agent.n_games / elapsed if elapsed > 0 else 0
                    print(f'Game {agent.n_games:4d} | {status:10s} | Score: {snake.score:3d} | '
                          f'Record: {record:3d} | Epsilon: {agent.epsilon:3d} | Speed: {games_per_sec:.1f} g/s')
                else:
                    print(f'Game {agent.n_games:4d} | {status:10s} | Score: {snake.score:3d} | '
                          f'Record: {record:3d} | Epsilon: {agent.epsilon:3d}')

            # Track metrics
            plot_scores.append(snake.score)
            total_score += snake.score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # Plot every 10 games (seulement si GUI)
            if not HEADLESS and agent.n_games % 10 == 0:
                plot_results(plot_scores, plot_mean_scores)

            # Save metrics (every 50 games en GUI, every 100 en headless)
            save_interval = 50 if not HEADLESS else 100
            if agent.n_games % save_interval == 0:
                save_metrics(agent.n_games, record, mean_score, plot_scores)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print(f"Arrêt de l'entraînement")
        print(f"Total games: {agent.n_games}")
        print(f"Record: {record}")
        print(f"Average score: {total_score / agent.n_games if agent.n_games > 0 else 0:.2f}")

        # Afficher la vitesse moyenne finale
        elapsed = time.time() - start_training_time
        games_per_sec = agent.n_games / elapsed if elapsed > 0 else 0
        print(f"Training time: {elapsed:.1f}s")
        print(f"Average speed: {games_per_sec:.2f} games/sec")
        print("=" * 60)
        save_metrics(agent.n_games, record, total_score / agent.n_games if agent.n_games > 0 else 0, plot_scores)

        # Save final plot (seulement si GUI)
        if not HEADLESS:
            plt.savefig('results/training_plot.png')
            print("\nGraphique sauvegardé dans results/training_plot.png")
            pygame.quit()


def save_metrics(games, max_score, avg_score, scores):
    """Sauvegarde les métriques d'entraînement"""
    metrics = {
        'type': 'reinforcement_learning',
        'games_played': games,
        'max_score': max_score,
        'average_score': avg_score,
        'scores_history': scores[-100:] if len(scores) > 100 else scores  # Derniers 100
    }

    with open('results/rl_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    train()
