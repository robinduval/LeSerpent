# snake-ia.py
# Version simple DQN pour Snake (mode texte seulement)
# Respecte les états et rewards demandés dans le PDF (Danger, Direction, Pomme; rewards: +10/-10/step).
# Nécessite: torch, numpy
# Usage:
#   python snake-ia.py               -> entraîne (par défaut 500 épisodes)
#   python snake-ia.py 1000          -> entraîne 1000 épisodes
#   python snake-ia.py play          -> charge snake_model.pth et joue (mode texte, affiche actions/scores)

import random
import time
import sys
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pygame
from serpent import GAME_SPEED

# ---------- CONFIGURATION (modifiable) ----------
GRID_SIZE = 15  # grille GRID_SIZE x GRID_SIZE
MEMORY_SIZE = 100_000
BATCH_SIZE = 512
LR = 0.001
GAMMA = 0.9
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE_EPISODES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rewards (conformes à la consigne du PDF)
REWARD_EAT = 10.0
REWARD_DEAD = -10.0
REWARD_STEP = -0.1       # petit coût pour chaque pas (encourage trouver la pomme rapidement)
REWARD_VICTORY = 100.0  # si toute la grille est remplie

# Directions (absolues)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
CLOCKWISE = [RIGHT, DOWN, LEFT, UP]  # ordre pour tourner relative

# Actions (relatives à la direction actuelle)
# 0 = straight, 1 = right, 2 = left
ACTIONS = [0, 1, 2]

# ---------- VISUALIZATION CONSTANTS ----------
CELL_SIZE = 30
SCORE_PANEL_HEIGHT = 80
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Colors
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# ---------- ENVIRONNEMENT AVEC AFFICHAGE ----------
class SnakeGame:
    def __init__(self, w=GRID_SIZE, h=GRID_SIZE, render=False):
        self.w = w
        self.h = h
        self.render_mode = render
        
        # Initialize pygame if rendering
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake AI Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 30)
        else:
            self.screen = None
            self.clock = None
            self.font = None
        
        self.reset()

    def reset(self):
        self.head = [self.w // 4, self.h // 2]
        self.direction = RIGHT
        self.body = [self.head[:], [self.head[0]-1, self.head[1]], [self.head[0]-2, self.head[1]]]
        self.score = 0
        self.place_apple()
        self.frame_iteration = 0
        self.game_over = False
        
        # Render initial state
        if self.render_mode:
            self.render()
        
        return self.get_state()

    def place_apple(self):
        all_pos = [(x, y) for x in range(self.w) for y in range(self.h)]
        occupied = set(tuple(p) for p in self.body)
        choices = [p for p in all_pos if p not in occupied]
        if not choices:
            self.apple = None
        else:
            self.apple = list(random.choice(choices))

    def play_step(self, action):
        """
        Applique action (0/1/2) relative à la dir actuelle.
        Retourne (reward, done, score)
        """
        self.frame_iteration += 1
        # compute new direction & head
        idx = CLOCKWISE.index(self.direction)
        if action == 0:
            new_dir = CLOCKWISE[idx]
        elif action == 1:
            new_dir = CLOCKWISE[(idx + 1) % 4]
        else:
            new_dir = CLOCKWISE[(idx - 1) % 4]
        self.direction = new_dir
        self.head = [self.head[0] + self.direction[0], self.head[1] + self.direction[1]]
        self.body.insert(0, self.head[:])

        # collision?
        if self._is_collision(self.head):
            self.game_over = True
            return REWARD_DEAD, True, self.score

        # apple?
        if self.apple and self.head == self.apple:
            self.score += 1
            reward = REWARD_EAT
            self.place_apple()
            # victory: plus de place pour la pomme
            if self.apple is None:
                # rempli entièrement
                self.game_over = True
                return REWARD_VICTORY, True, self.score
            # don't pop tail (snake grows)
        else:
            # normal move: pop tail
            self.body.pop()
            reward = REWARD_STEP

        # optional: limit frames to avoid infinite wandering
        if self.frame_iteration > 100 * len(self.body):
            self.game_over = True
            return REWARD_DEAD, True, self.score

        # Render after each step
        if self.render_mode:
            self.render()

        return reward, False, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # walls
        if pt[0] < 0 or pt[0] >= self.w or pt[1] < 0 or pt[1] >= self.h:
            return True
        # self body collision (ignore head)
        if pt in self.body[1:]:
            return True
        return False

    def get_state(self):
        """
        État selon la consigne:
        - danger en face, à droite, à gauche
        - direction one-hot (left, right, up, down)
        - position de la pomme relative (left,right,up,down)
        -> vecteur binaire de longueur 11 (comme dans le PDF)
        """
        head = self.head
        dir_now = self.direction

        # compute relative points
        # forward point
        fpt = [head[0] + dir_now[0], head[1] + dir_now[1]]
        # right: rotate clockwise
        cw = {RIGHT: DOWN, DOWN: LEFT, LEFT: UP, UP: RIGHT}
        ccw = {RIGHT: UP, UP: LEFT, LEFT: DOWN, DOWN: RIGHT}
        rdir = cw[dir_now]
        ldir = ccw[dir_now]
        rpt = [head[0] + rdir[0], head[1] + rdir[1]]
        lpt = [head[0] + ldir[0], head[1] + ldir[1]]

        danger_straight = int(self._is_collision(fpt))
        danger_right = int(self._is_collision(rpt))
        danger_left = int(self._is_collision(lpt))

        dir_left_flag = int(dir_now == LEFT)
        dir_right_flag = int(dir_now == RIGHT)
        dir_up_flag = int(dir_now == UP)
        dir_down_flag = int(dir_now == DOWN)

        apple_left = int(self.apple is not None and self.apple[0] < head[0])
        apple_right = int(self.apple is not None and self.apple[0] > head[0])
        apple_up = int(self.apple is not None and self.apple[1] < head[1])
        apple_down = int(self.apple is not None and self.apple[1] > head[1])

        state = np.array([
            danger_straight,
            danger_right,
            danger_left,
            dir_left_flag,
            dir_right_flag,
            dir_up_flag,
            dir_down_flag,
            apple_left,
            apple_right,
            apple_up,
            apple_down
        ], dtype=int)
        return state

    def render(self):
        """Render the game state using pygame"""
        if not self.render_mode or self.screen is None:
            return
        
        # Handle pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Fill background
        self.screen.fill(GRIS_FOND)
        
        # Draw game area
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.screen, NOIR, game_area_rect)
        
        # Draw grid
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
        for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))
        
        # Draw apple
        if self.apple:
            apple_rect = pygame.Rect(
                self.apple[0] * CELL_SIZE,
                self.apple[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(self.screen, ROUGE, apple_rect, border_radius=5)
            pygame.draw.circle(self.screen, BLANC, 
                             (apple_rect.x + CELL_SIZE * 0.7, apple_rect.y + CELL_SIZE * 0.3), 
                             CELL_SIZE // 8)
        
        # Draw snake body
        for segment in self.body[1:]:
            body_rect = pygame.Rect(
                segment[0] * CELL_SIZE,
                segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(self.screen, VERT, body_rect)
            pygame.draw.rect(self.screen, NOIR, body_rect, 1)
        
        # Draw snake head
        head_rect = pygame.Rect(
            self.head[0] * CELL_SIZE,
            self.head[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
            CELL_SIZE,
            CELL_SIZE
        )
        pygame.draw.rect(self.screen, ORANGE, head_rect)
        pygame.draw.rect(self.screen, NOIR, head_rect, 2)
        
        # Draw score panel
        pygame.draw.rect(self.screen, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
        pygame.draw.line(self.screen, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)
        
        # Display score
        score_text = self.font.render(f"Score: {self.score}", True, BLANC)
        self.screen.blit(score_text, (10, 20))
        
        # Display frame count
        frame_text = self.font.render(f"Frames: {self.frame_iteration}", True, BLANC)
        self.screen.blit(frame_text, (SCREEN_WIDTH - frame_text.get_width() - 10, 20))
        
        # Update display
        pygame.display.flip()
        
        # Control speed using GAME_SPEED from serpent.py
        if self.clock:
            self.clock.tick(GAME_SPEED)

# ---------- MODEL (Simple Linear Network) ----------
class LinearQNet(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ---------- AGENT (replay, greedy, train) ----------
class Agent:
    def __init__(self):
        self.epsilon = EPS_START
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = LinearQNet().to(DEVICE)
        self.target = LinearQNet().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        self.update_target()
        self.steps = 0

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def act(self, state):
        # epsilon-greedy
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = self.model(state_t)
        return int(torch.argmax(q).cpu().item())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train_short(self, s, a, r, s2, done):
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        s2_t = torch.tensor(s2, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        r_t = torch.tensor([r], dtype=torch.float32).to(DEVICE)
        a_t = torch.tensor([a], dtype=torch.long).to(DEVICE)

        pred = self.model(s_t)
        target = pred.clone().detach()
        with torch.no_grad():
            next_q = self.target(s2_t).max(1)[0]
        if done:
            target[0, a] = r_t
        else:
            target[0, a] = r_t + GAMMA * next_q
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_long(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(DEVICE)

        q_pred = self.model(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + (~dones).float() * GAMMA * q_next

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path="snake_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="snake_model.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=DEVICE))
            self.update_target()
            print("Model loaded:", path)
        else:
            print("No model found at", path)

# ---------- TRAIN / PLAY LOOP (with live visualization) ----------
def train(n_episodes=500, render=True):
    """
    Train the agent with live visualization.
    Args:
        n_episodes: Number of episodes to train
        render: If True, shows live visualization (speed controlled by GAME_SPEED in serpent.py)
    """
    env = SnakeGame(render=render)
    agent = Agent()
    scores = []
    best = 0
    start = time.time()

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        
        # Display episode info on pygame window
        if render and env.screen:
            ep_text = env.font.render(f"Episode: {ep}/{n_episodes}", True, BLANC)
            env.screen.blit(ep_text, (SCREEN_WIDTH // 2 - ep_text.get_width() // 2, 50))
            pygame.display.flip()
        
        while True:
            action = agent.act(state)
            reward, done, score = env.play_step(action)
            next_state = env.get_state()
            agent.remember(state, action, reward, next_state, done)
            agent.train_short(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            if done:
                agent.train_long()
                if ep % TARGET_UPDATE_EPISODES == 0:
                    agent.update_target()
                scores.append(score)
                avg_last_50 = np.mean(scores[-50:]) if scores else 0
                if score > best:
                    best = score
                    agent.save()
                # Print text summary per episode
                print(f"Ep {ep:4d} | Score: {score:2d} | Steps: {steps:4d} | TotRew: {total_reward:6.1f} | Eps: {agent.epsilon:.3f} | Best: {best} | Avg50: {avg_last_50:.2f}")
                break

    duration = time.time() - start
    print(f"Training done ({n_episodes} eps) in {duration/60:.2f} min. Best score: {best}")
    agent.save()
    
    if render:
        pygame.quit()
    
    return agent, scores

def play(agent, max_steps=10000, render=True):
    """
    Play a game with the trained agent.
    Args:
        agent: Trained agent
        max_steps: Maximum steps per game
        render: If True, shows live visualization (speed controlled by GAME_SPEED in serpent.py)
    """
    env = SnakeGame(render=render)
    state = env.reset()
    total_reward = 0.0
    steps = 0
    print("Play mode: watching trained agent play")
    while True:
        action = agent.act(state)  # note: act() uses epsilon decay; to force deterministic, set agent.epsilon=0 before calling
        reward, done, score = env.play_step(action)
        state = env.get_state()
        total_reward += reward
        steps += 1
        if not render:
            print(f"Step {steps:4d} | Action: {action} | Reward: {reward:+5.1f} | Score: {score}")
        if done or steps >= max_steps:
            print(f"Game over. Final Score: {score} | Steps: {steps} | Total reward: {total_reward:.1f}")
            break
    
    if render:
        # Wait a bit before closing
        time.sleep(2)
        pygame.quit()

# ---------- CLI ----------
if __name__ == "__main__":
    # default: train 500 episodes
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ("play", "test"):
            agent = Agent()
            agent.load()
            # deterministic play
            agent.epsilon = 0.0
            play(agent)
            sys.exit(0)
        else:
            try:
                n = int(arg)
            except:
                n = 500
            train(n_episodes=n)
    else:
        train(500)
