#!/usr/bin/env python3
"""Simple evaluator/replayer for the Snake agents.

This script runs N episodes (headless by default) using the best available agent
(DQN checkpoint in ./checkpoints or fallback q_table.pkl). It keeps the best
episode (highest score) and at the end asks the user whether to replay that
best episode visually.

Features:
- Toroidal wrap (edges connect).
- Loop-detection to stop when repeating states.
"""

import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pygame  # utilisé pour le replay GUI

# ==== Import ROBUSTE du serpent.py local (évite d'importer le paquet PyPI "serpent") ====
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import serpent as serpent_mod  # tente l'import local
    # Sanity check: doit exposer Snake/Apple/constantes
    _ = serpent_mod.Snake
    _ = serpent_mod.UP
except Exception:
    # Chargement direct par chemin si l'import normal ne pointe pas le bon module
    import importlib.util
    serpent_path = os.path.join(REPO_ROOT, 'serpent.py')
    if not os.path.exists(serpent_path):
        raise ImportError("Impossible de trouver serpent.py au chemin attendu: " + serpent_path)
    spec = importlib.util.spec_from_file_location('serpent_local', serpent_path)
    serpent_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serpent_mod)

# On l'appelle "serpent" partout dans ce script
serpent = serpent_mod

MODEL_DQN_GLOB = "checkpoints/*.pt"
MODEL_Q = "q_table.pkl"


def find_best_dqn():
    files = sorted(glob.glob(MODEL_DQN_GLOB), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def load_agent():
    """Return (agent_act, state_fn, name).

    agent_act(state)-> action_idx ; state_fn(env)-> state
    Preference: DQN checkpoint. Fallback: Q-table. Else random.
    """
    # Try DQN if available
    best = find_best_dqn()
    if best:
        try:
            import torch
            from dqn_agent import DQNAgent

            device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
            agent = DQNAgent(state_dim=11, action_dim=4, device=device)
            agent.load(best)
            agent.net.eval()

            def act(s):
                return int(agent.act(np.array(s, dtype=np.float32), epsilon=0.0))

            return act, state_from_env, f"DQN:{os.path.basename(best)}"
        except Exception:
            pass

    # Try Q-table file
    if os.path.exists(MODEL_Q):
        try:
            import pickle
            with open(MODEL_Q, 'rb') as f:
                q = pickle.load(f)

            def act(s):
                key = tuple(s)
                v = q.get(key)
                if v is None:
                    return int(np.random.randint(0, 4))
                if isinstance(v, int):
                    return int(v)
                if isinstance(v, (list, tuple, np.ndarray)):
                    return int(np.argmax(v))
                return int(np.random.randint(0, 4))

            return act, state_from_env, f"QTable:{MODEL_Q}"
        except Exception:
            pass

    # Fallback: random
    def rand_act(s):
        return int(np.random.randint(0, 4))
    return rand_act, state_from_env, "Random"


def state_from_env(snake, apple):
    """Encode environment into the 11-element state used by agents.

    Order: danger straight/left/right (3),
           direction one-hot (4),
           apple left/right/up/down (4) -> total 11.
    """
    head = snake.head_pos
    dirx, diry = snake.direction

    def danger_in_direction(dx, dy):
        nx = (head[0] + dx) % serpent.GRID_SIZE
        ny = (head[1] + dy) % serpent.GRID_SIZE
        return [nx, ny] in snake.body

    # Danger relative to current heading
    if (dirx, diry) == (1, 0):         # RIGHT
        straight = danger_in_direction(1, 0)
        left = danger_in_direction(0, -1)
        right = danger_in_direction(0, 1)
    elif (dirx, diry) == (-1, 0):      # LEFT
        straight = danger_in_direction(-1, 0)
        left = danger_in_direction(0, 1)
        right = danger_in_direction(0, -1)
    elif (dirx, diry) == (0, 1):       # DOWN
        straight = danger_in_direction(0, 1)
        left = danger_in_direction(1, 0)
        right = danger_in_direction(-1, 0)
    else:                              # UP
        straight = danger_in_direction(0, -1)
        left = danger_in_direction(-1, 0)
        right = danger_in_direction(1, 0)

    dir_up = 1 if (dirx, diry) == (0, -1) else 0
    dir_down = 1 if (dirx, diry) == (0, 1) else 0
    dir_left = 1 if (dirx, diry) == (-1, 0) else 0
    dir_right = 1 if (dirx, diry) == (1, 0) else 0

    apple_left = 1 if apple.position[0] < head[0] else 0
    apple_right = 1 if apple.position[0] > head[0] else 0
    apple_up = 1 if apple.position[1] < head[1] else 0
    apple_down = 1 if apple.position[1] > head[1] else 0

    return [
        int(straight), int(left), int(right),
        dir_up, dir_down, dir_left, dir_right,
        apple_left, apple_right, apple_up, apple_down
    ]


def wrap_position(pos):
    """Wrap a (x,y) pair around the GRID_SIZE."""
    x, y = pos
    x %= serpent.GRID_SIZE
    y %= serpent.GRID_SIZE
    return [x, y]


def run_episode(agent_act, state_fn, max_steps=1000, loop_patience=200):
    """Run one episode headless, return (score, steps, trajectory).

    trajectory: list of (snake_head, apple_pos, body) for replay.
    loop_patience: stop if (head,direction,apple) repeats too many times.
    """
    snake = serpent.Snake()
    apple = serpent.Apple(snake.body)

    def step_move(snake_obj):
        new_x = snake_obj.head_pos[0] + snake_obj.direction[0]
        new_y = snake_obj.head_pos[1] + snake_obj.direction[1]
        snake_obj.head_pos = wrap_position([new_x, new_y])
        snake_obj.body.insert(0, list(snake_obj.head_pos))
        if not snake_obj.grow_pending:
            snake_obj.body.pop()
        else:
            snake_obj.grow_pending = False

    seen = defaultdict(int)
    trajectory = []
    steps = 0

    while steps < max_steps:
        state = state_fn(snake, apple)
        action_idx = agent_act(state)
        # Action mapping: (UP, RIGHT, DOWN, LEFT)
        ACTION_LIST = [serpent.UP, serpent.RIGHT, serpent.DOWN, serpent.LEFT]
        action = ACTION_LIST[action_idx % 4]
        snake.set_direction(action)

        step_move(snake)
        steps += 1

        key = (tuple(snake.head_pos), tuple(snake.direction), tuple(apple.position))
        seen[key] += 1
        if seen[key] > loop_patience:
            # suspected loop; terminate episode
            break

        # apple eaten
        if snake.head_pos == list(apple.position):
            snake.grow()
            if not apple.relocate(snake.body):
                # victory: filled
                break

        # self-collision (should still be checked)
        if snake.check_self_collision():
            break

        # snapshot for replay
        trajectory.append((
            list(snake.head_pos),
            list(apple.position) if apple.position else None,
            [list(p) for p in snake.body]
        ))

    return snake.score, steps, trajectory


def replay_trajectory(trajectory, speed=12):
    """Open pygame and replay the saved trajectory visually."""
    # Prépare un driver vidéo raisonnable (WSLg: wayland/x11)
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if os.path.exists("/mnt/wslg/runtime-dir/wayland-0") or os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "wayland")
        os.environ.setdefault("WAYLAND_DISPLAY", os.environ.get("WAYLAND_DISPLAY", "wayland-0"))
    else:
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")

    pygame.init()
    screen = pygame.display.set_mode((serpent.SCREEN_WIDTH, serpent.SCREEN_HEIGHT))
    pygame.display.set_caption("Snake - Best Run Replay")
    clock = pygame.time.Clock()

    font_main = pygame.font.Font(None, 40)
    running = True
    idx = 0

    while running and idx < len(trajectory):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        head_pos, apple_pos, body = trajectory[idx]

        # draw
        screen.fill(serpent.GRIS_FOND)
        game_area_rect = pygame.Rect(0, serpent.SCORE_PANEL_HEIGHT, serpent.SCREEN_WIDTH, serpent.SCREEN_WIDTH)
        pygame.draw.rect(screen, serpent.NOIR, game_area_rect)
        serpent.draw_grid(screen)

        # apple
        if apple_pos:
            rect = pygame.Rect(
                apple_pos[0] * serpent.CELL_SIZE,
                apple_pos[1] * serpent.CELL_SIZE + serpent.SCORE_PANEL_HEIGHT,
                serpent.CELL_SIZE, serpent.CELL_SIZE
            )
            pygame.draw.rect(screen, serpent.ROUGE, rect, border_radius=5)

        # snake body
        for segment in body[1:]:
            rect = pygame.Rect(
                segment[0] * serpent.CELL_SIZE,
                segment[1] * serpent.CELL_SIZE + serpent.SCORE_PANEL_HEIGHT,
                serpent.CELL_SIZE, serpent.CELL_SIZE
            )
            pygame.draw.rect(screen, serpent.VERT, rect)
            pygame.draw.rect(screen, serpent.NOIR, rect, 1)

        # head
        head_rect = pygame.Rect(
            head_pos[0] * serpent.CELL_SIZE,
            head_pos[1] * serpent.CELL_SIZE + serpent.SCORE_PANEL_HEIGHT,
            serpent.CELL_SIZE, serpent.CELL_SIZE
        )
        pygame.draw.rect(screen, serpent.ORANGE, head_rect)

        # info (utilise display_info du module pour garder le style)
        class _S:
            def __init__(self, body, score):
                self.body = body
                self.score = score

        serpent.display_info(screen, font_main, _S(body, idx), time.time())

        pygame.display.flip()
        clock.tick(max(1, int(speed)))
        idx += 1

    pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes to run headless')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--loop-patience', type=int, default=200, help='Repeat count before cutting loop')
    args = parser.parse_args()

    agent_act, state_fn, name = load_agent()
    print('Using agent:', name)

    best_score = -1
    best_traj = None
    best_steps = 0

    for i in range(args.episodes):
        score, steps, traj = run_episode(agent_act, state_fn,
                                         max_steps=args.max_steps,
                                         loop_patience=args.loop_patience)
        print(f'Episode {i+1}/{args.episodes}: score={score} steps={steps} traj_len={len(traj)}')
        if score > best_score:
            best_score = score
            best_traj = traj
            best_steps = steps

    if best_traj is None:
        print('No valid episodes produced.')
        return

    print(f'Best score: {best_score} (steps={best_steps}).')
    try:
        ans = input('Replay best episode now? [y/N]: ').strip().lower()
    except Exception:
        ans = 'n'
    if ans in ('y', 'yes'):
        replay_trajectory(best_traj)


if __name__ == '__main__':
    main()
