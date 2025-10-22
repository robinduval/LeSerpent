#!/usr/bin/env python3
"""
Single-file Snake DQN implementation (game + model + agent + training)
Usage:
    python3 snake-ia.py [--vis] [--fps N] [--spf M]

Options:
    --vis, -v     Enable pygame visualization
    --fps N       Visualization FPS (default 8)
    --spf M       Steps per frame (agent steps per displayed frame) (default 1)

This file bundles the previous project into one runnable script.
"""

import sys
import time
import random
from collections import deque
import numpy as np

# optional imports
try:
    import pygame
except Exception:
    pygame = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except Exception:
    torch = None

# ---------------------- Game constants & classes ----------------------
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 60
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Colors
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
LIGHT_BLUE = (173, 216, 230)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos[:], [self.head_pos[0] - 1, self.head_pos[1]], [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
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
        self.grow_pending = True
        self.score += 1

    def check_wall_collision(self):
        return False

    def check_self_collision(self):
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        return self.check_wall_collision() or self.check_self_collision()

    def draw(self, surface):
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)
        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)

class Apple:
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(pos) not in occupied_positions]
        if not available_positions:
            return None
        return random.choice(available_positions)

    def relocate(self, snake_body):
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False

    def draw(self, surface):
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            pygame.draw.circle(surface, BLANC, (int(rect.x + CELL_SIZE * 0.7), int(rect.y + CELL_SIZE * 0.3)), CELL_SIZE // 8)

def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

def display_info(surface, font, snake, start_time):
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 10))
    max_cells = GRID_SIZE * GRID_SIZE
    fill_rate = (len(snake.body) / max_cells) * 100
    fill_text = font.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
    surface.blit(fill_text, (10, 10 + score_text.get_height() + 6))
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_text = font.render(f"Temps: {minutes:02d}:{seconds:02d}", True, BLANC)
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 10))

# ---------------------- Model / Trainer ----------------------
if torch is None:
    # Minimal placeholders to allow syntax-checking even without torch installed
    class Linear_QNet:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return None

    class QTrainer:
        def __init__(self, *args, **kwargs):
            pass
        def train_step(self, *args, **kwargs):
            return
else:
    class Linear_QNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    class QTrainer:
        def __init__(self, model, lr, gamma):
            self.model = model
            self.lr = lr
            self.gamma = gamma
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
            self.criterion = nn.MSELoss()

        def train_step(self, state, action, reward, next_state, done):
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)

            if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done,)

            pred = self.model(state)
            target = pred.clone()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_next = self.model(next_state[idx])
                    Q_new = reward[idx] + self.gamma * torch.max(Q_next)
                target[idx][action[idx].item()] = Q_new

            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()
            self.optimizer.step()

# ---------------------- Agent ----------------------
MAX_MEMORY = 100_000
BATCH_SIZE = 128

class Agent:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001, gamma=0.9):
        self.n_games = 0
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 5000.0
        self.gamma = gamma
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma)

    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.n_games / self.epsilon_decay)

    def get_state(self, game):
        head = game.snake.head_pos
        point_l = [head[0] - 1, head[1]]
        point_r = [head[0] + 1, head[1]]
        point_u = [head[0], head[1] - 1]
        point_d = [head[0], head[1] + 1]

        dir_l = game.snake.direction == (-1,0)
        dir_r = game.snake.direction == (1,0)
        dir_u = game.snake.direction == (0,-1)
        dir_d = game.snake.direction == (0,1)

        def danger_at(point):
            x, y = point
            if x < 0 or x >= game.GRID_SIZE or y < 0 or y >= game.GRID_SIZE:
                return True
            if list(point) in game.snake.body:
                return True
            return False

        if dir_r:
            danger_straight = danger_at(point_r)
            danger_right = danger_at(point_d)
            danger_left = danger_at(point_u)
        elif dir_l:
            danger_straight = danger_at(point_l)
            danger_right = danger_at(point_u)
            danger_left = danger_at(point_d)
        elif dir_u:
            danger_straight = danger_at(point_u)
            danger_right = danger_at(point_r)
            danger_left = danger_at(point_l)
        else:
            danger_straight = danger_at(point_d)
            danger_right = danger_at(point_l)
            danger_left = danger_at(point_r)

        apple = game.apple.position
        apple_left = apple[0] < head[0]
        apple_right = apple[0] > head[0]
        apple_up = apple[1] < head[1]
        apple_down = apple[1] > head[1]

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(apple_left),
            int(apple_right),
            int(apple_up),
            int(apple_down)
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        epsilon = self.get_epsilon()
        final_move = [0, 0, 0, 0]
        if random.random() < epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            if torch is None:
                move = random.randint(0,3)
                final_move[move] = 1
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                with torch.no_grad():
                    prediction = self.model(state0)
                    move = torch.argmax(prediction).item()
                    final_move[move] = 1
        return final_move

# ---------------------- Game wrapper ----------------------
class Game:
    def __init__(self):
        self.GRID_SIZE = GRID_SIZE
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.score = 0

    def reset(self):
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.score = 0

    def play_step(self, action):
        if isinstance(action, list) or isinstance(action, tuple):
            if len(action) == 4:
                idx = action.index(1)
            else:
                idx = max(range(len(action)), key=lambda i: action[i])
        else:
            idx = int(action)

        if idx == 0:
            self.snake.set_direction((0, -1))
        elif idx == 1:
            self.snake.set_direction((0, 1))
        elif idx == 2:
            self.snake.set_direction((-1, 0))
        elif idx == 3:
            self.snake.set_direction((1, 0))

        self.snake.move()

        reward = 0
        done = False

        if self.snake.is_game_over():
            reward = -10
            done = True
            return reward, done, self.score

        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            self.score += 1
            reward = 10
            if not self.apple.relocate(self.snake.body):
                reward = 100
                done = True
                return reward, done, self.score

        reward -= 0.1
        return reward, done, self.score

# ---------------------- Training loop ----------------------

def train(visualize: bool = False, vis_fps: int = 8, steps_per_frame: int = 1):
    input_size = 11
    hidden_size = 128
    output_size = 4

    agent = Agent(input_size, hidden_size, output_size)
    game = Game()

    if visualize and pygame is None:
        print("pygame is not available; run without --vis or install pygame.")
        return

    if visualize:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Snake - DQN visualization')
        font_main = pygame.font.Font(None, 40)
        clock = pygame.time.Clock()

    n_epochs = 1000
    best_score_overall = 0
    best_score_duration = 0.0

    for epoch in range(n_epochs):
        game.reset()
        state_old = agent.get_state(game)
        done = False
        total_reward = 0
        score = 0
        start_time = time.time()

        while not done:
            if visualize:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        return

            for _ in range(steps_per_frame):
                action = agent.get_action(state_old)
                reward, done, score = game.play_step(action)
                state_new = agent.get_state(game)
                action_idx = action.index(1) if isinstance(action, list) else int(action)
                agent.train_short_memory(state_old, action_idx, reward, state_new, done)
                agent.remember(state_old, action_idx, reward, state_new, done)
                state_old = state_new
                total_reward += reward
                if done:
                    break

            if visualize:
                screen.fill(GRIS_FOND)
                game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
                pygame.draw.rect(screen, NOIR, game_area_rect)
                draw_grid(screen)
                # highlight checked cells
                head = game.snake.head_pos
                dir_l = game.snake.direction == (-1,0)
                dir_r = game.snake.direction == (1,0)
                dir_u = game.snake.direction == (0,-1)
                dir_d = game.snake.direction == (0,1)
                point_l = [head[0] - 1, head[1]]
                point_r = [head[0] + 1, head[1]]
                point_u = [head[0], head[1] - 1]
                point_d = [head[0], head[1] + 1]
                if dir_r:
                    pts = [point_r, point_d, point_u]
                elif dir_l:
                    pts = [point_l, point_u, point_d]
                elif dir_u:
                    pts = [point_u, point_r, point_l]
                else:
                    pts = [point_d, point_l, point_r]
                highlight_surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                highlight_surf.fill((*LIGHT_BLUE, 120))
                for p in pts:
                    x = (p[0] % game.GRID_SIZE) * CELL_SIZE
                    y = (p[1] % game.GRID_SIZE) * CELL_SIZE + SCORE_PANEL_HEIGHT
                    screen.blit(highlight_surf, (x, y))
                game.apple.draw(screen)
                game.snake.draw(screen)
                display_info(screen, font_main, game.snake, start_time)
                pygame.display.flip()
                clock.tick(vis_fps)

        agent.n_games += 1
        agent.train_long_memory()

        epoch_duration = time.time() - start_time
        if score > best_score_overall:
            best_score_overall = score
            best_score_duration = epoch_duration

        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Game {agent.n_games} - Score {score} - TotalReward {total_reward:.2f} - Duration {epoch_duration:.2f}s - BestScore {best_score_overall} (best duration {best_score_duration:.2f}s)")
            if torch is not None:
                torch.save(agent.model.state_dict(), 'model.pth')

# ---------------------- CLI ----------------------
if __name__ == '__main__':
    visualize = False
    if '--vis' in sys.argv or '-v' in sys.argv:
        visualize = True
    fps = 8
    spf = 1
    if '--fps' in sys.argv:
        try:
            fps = int(sys.argv[sys.argv.index('--fps') + 1])
        except Exception:
            pass
    if '--spf' in sys.argv:
        try:
            spf = int(sys.argv[sys.argv.index('--spf') + 1])
        except Exception:
            pass
    train(visualize=visualize, vis_fps=fps, steps_per_frame=spf)
