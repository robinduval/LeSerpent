#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snake - Version Apprentissage par Renforcement (DQN)
---------------------------------------------------
- Entraîne un agent Deep Q-Learning à jouer à Snake.
- Trois blocs : Game (environnement PyGame minimal), Model (réseau Torch),
  Agent (mémoire d'expérience, epsilon-greedy, entraînement).
- Inspiré des patterns DQN classiques.

Dépendances : torch, pygame, numpy
    pip install torch pygame numpy
Exécution (entraînement sans rendu) :
    python snake-ia.py
Affichage temps réel (lent, pour démo) :
    python snake-ia.py --render
"""

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Tuple

import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.optim as optim

# Pygame (uniquement si rendu demandé)
import pygame


# ======================
# ====== CONSTANTES ====
# ======================

GRID_SIZE = 15              # 15x15
CELL_SIZE = 30              # pixels (pour rendu)
SCORE_PANEL_HEIGHT = 80     # pixels (pour rendu)
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (pour rendu)
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Récompenses (guidelines issues des slides)
REWARD_EAT = 10.0
REWARD_DIE = -10.0
REWARD_STEP = -0.01           # petit bonus pour avancer (peut être mis à -0.01 si vous observez du looping)
REWARD_WIN = 100.0

# DQN
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9

# Actions relatives (plus stable que 4 actions absolues) :
# 0 -> tout droit, 1 -> tourner à droite, 2 -> tourner à gauche
N_ACTIONS = 3

# État (11 features standards) :
# [Danger avant, Danger droite, Danger gauche,
#  Dir_gauche, Dir_droite, Dir_haut, Dir_bas,
#  Food_gauche, Food_droite, Food_haut, Food_bas]
N_STATES = 11


# ======================
# ====== UTILITAIRES ===
# ======================

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


@dataclass
class Point:
    x: int
    y: int


# ======================
# ====== GAME (Env) ====
# ======================

class SnakeGameAI:
    """ Environnement Snake simplifié pour DQN (avec PyGame optionnel pour rendu). """
    def __init__(self, grid_size=GRID_SIZE, render=False, seed=None):
        self.grid_size = grid_size
        self.render = render

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.reset()

        # Pour rendu
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake DQN - Training/Play")
            self.clock = pygame.time.Clock()
            self.font_main = pygame.font.Font(None, 40)

    def reset(self):
        self.direction = Direction.RIGHT
        # Snake initial : 3 segments alignés horizontalement
        x = self.grid_size // 4
        y = self.grid_size // 2
        self.head = Point(x, y)
        self.snake: List[Point] = [self.head,
                                   Point(self.head.x - 1, self.head.y),
                                   Point(self.head.x - 2, self.head.y)]
        self.score = 0
        self.frame_iteration = 0  # sécurité anti-boucles
        self._place_food()
        self.game_over = False
        self.victory = False

    def _place_food(self):
        positions = {(p.x, p.y) for p in self.snake}
        free = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                if (x, y) not in positions]
        if not free:
            # Plus d'espace dispo : victoire
            self.food = None
            self.victory = True
            return
        self.food = Point(*random.choice(free))

    def _is_collision(self, p: Point = None) -> bool:
        """ Collision si murs (pas de wrap) ou si on mord son corps. """
        if p is None:
            p = self.head
        if p.x < 0 or p.x >= self.grid_size or p.y < 0 or p.y >= self.grid_size:
            return True
        if p in self.snake[1:]:
            return True
        return False

    def _move(self, action: List[int]):
        """
        Action one-hot [straight, right, left] relative à la direction actuelle.
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):       # straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):     # right turn
            new_dir = clock_wise[(idx + 1) % 4]
        else:                                       # left turn
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.head = Point(x, y)

    def play_step(self, action: List[int]) -> Tuple[float, bool, int]:
        """
        Avance le jeu d'un pas.
        :param action: [1,0,0]=straight, [0,1,0]=right, [0,0,1]=left
        :return: reward, game_over, score
        """
        self.frame_iteration += 1

        # Gérer événements (pour fermer la fenêtre en mode render)
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        # 1) Bouger
        old_head = Point(self.head.x, self.head.y)
        self._move(action)

        # 2) Vérifier collision
        reward = REWARD_STEP
        if self._is_collision():
            self.game_over = True
            return REWARD_DIE, True, self.score

        # 3) Vérifier si pomme mangée
        self.snake.insert(0, self.head)
        if self.food is not None and self.head.x == self.food.x and self.head.y == self.food.y:
            self.score += 1
            reward = REWARD_EAT
            self._place_food()
            if self.victory:
                # Plus de cellules libres : victoire totale
                return REWARD_WIN, True, self.score
        else:
            self.snake.pop()

        # 4) Anti-boucle : si trop de frames sans progrès -> terminer (empêche les spirales infinies)
        if self.frame_iteration > 100 * len(self.snake):
            self.game_over = True
            reward = REWARD_DIE
            return reward, True, self.score

        # 5) Rendu optionnel
        if self.render:
            self._draw()

        return reward, False, self.score

    # ---------- État / Observations ----------
    def get_state(self) -> np.ndarray:
        """
        11 features binaires/0-1 :
        - danger straight/right/left
        - direction (one-hot)
        - food relative pos (left/right/up/down)
        """
        head = self.head
        # Points autour selon orientation courante (utilisés pour danger R/L/Front)
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # Danger relatif
        def danger_ahead():
            if dir_r:
                return self._is_collision(point_r)
            if dir_l:
                return self._is_collision(point_l)
            if dir_u:
                return self._is_collision(point_u)
            if dir_d:
                return self._is_collision(point_d)

        def danger_right():
            if dir_u:
                return self._is_collision(point_r)
            if dir_d:
                return self._is_collision(point_l)
            if dir_l:
                return self._is_collision(point_u)
            if dir_r:
                return self._is_collision(point_d)

        def danger_left():
            if dir_u:
                return self._is_collision(point_l)
            if dir_d:
                return self._is_collision(point_r)
            if dir_l:
                return self._is_collision(point_d)
            if dir_r:
                return self._is_collision(point_u)

        food_left = self.food is not None and self.food.x < self.head.x
        food_right = self.food is not None and self.food.x > self.head.x
        food_up = self.food is not None and self.food.y < self.head.y
        food_down = self.food is not None and self.food.y > self.head.y

        state = [
            # Danger
            1.0 if danger_ahead() else 0.0,
            1.0 if danger_right() else 0.0,
            1.0 if danger_left() else 0.0,
            # Direction
            1.0 if dir_l else 0.0,
            1.0 if dir_r else 0.0,
            1.0 if dir_u else 0.0,
            1.0 if dir_d else 0.0,
            # Food relative position
            1.0 if food_left else 0.0,
            1.0 if food_right else 0.0,
            1.0 if food_up else 0.0,
            1.0 if food_down else 0.0,
        ]
        return np.array(state, dtype=np.float32)

    # ---------- Rendu optionnel ----------
    def _draw(self):
        self.screen.fill(GRIS_FOND)
        # panneau du haut
        pygame.draw.rect(self.screen, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
        pygame.draw.line(self.screen, BLANC, (0, SCORE_PANEL_HEIGHT - 2),
                         (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)
        score_text = self.font_main.render(f"Score: {self.score}", True, BLANC)
        self.screen.blit(score_text, (10, 20))

        # zone de jeu
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.screen, NOIR, game_area_rect)

        # grille
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
        for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

        # food
        if self.food:
            rect = pygame.Rect(self.food.x * CELL_SIZE,
                               self.food.y * CELL_SIZE + SCORE_PANEL_HEIGHT,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, ROUGE, rect, border_radius=5)

        # snake
        # corps
        for p in self.snake[1:]:
            rect = pygame.Rect(p.x * CELL_SIZE, p.y * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, VERT, rect)
            pygame.draw.rect(self.screen, NOIR, rect, 1)
        # tête
        head_rect = pygame.Rect(self.snake[0].x * CELL_SIZE,
                                self.snake[0].y * CELL_SIZE + SCORE_PANEL_HEIGHT,
                                CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, ORANGE, head_rect)
        pygame.draw.rect(self.screen, NOIR, head_rect, 2)

        pygame.display.flip()
        # limiter un peu pour pouvoir suivre (ajustez si besoin)
        self.clock.tick(60)


# ======================
# ====== MODEL (DQN) ===
# ======================

class LinearQNet(nn.Module):
    def __init__(self, input_size=N_STATES, hidden_size=256, output_size=N_ACTIONS):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

        # Xavier init
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)


class QTrainer:
    def __init__(self, model: nn.Module, lr=LR, gamma=GAMMA):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Peut accepter batchs (numpy/torch) ou scalaires.
        """
        state = torch.tensor(np.array(state), dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float32)

        if len(state.shape) == 1:
            # ajouter la dimension batch
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # Q valeurs
        pred = self.model(state)
        # Q cible
        target = pred.clone().detach()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][action[idx].argmax().item()] = Q_new

        # Optim step
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()


# ======================
# ====== AGENT =========
# ======================

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # exploration
        self.gamma = GAMMA
        self.memory: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI) -> np.ndarray:
        return game.get_state()

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

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Epsilon-greedy : prob d'explorer diminue avec n_games.
        Retourne un one-hot sur 3 actions.
        """
        # décroissance epsilon (linéaire)
        self.epsilon = max(0, 80 - self.n_games)  # commence haut, descend vers 0
        final_move = [0, 0, 0]

        if random.random() < self.epsilon / 100.0:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return np.array(final_move, dtype=np.float32)


# ======================
# ===== TRAINING =======
# ======================

def train(render=False, max_games=500):
    game = SnakeGameAI(render=render)
    agent = Agent()

    best_score = 0
    scores = []
    mean_scores = []
    total_score = 0

    while agent.n_games < max_games:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                # sauvegarde du modèle (best)
                torch.save(agent.model.state_dict(), "best_model.pth")

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)

            print(f"Game {agent.n_games} | Score: {score} | Best: {best_score} | Mean: {mean_score:.2f} | Epsilon: {agent.epsilon}")

    # sauvegarde finale
    torch.save(agent.model.state_dict(), "last_model.pth")
    return scores, mean_scores


def play_trained(render=True, weights_path="best_model.pth"):
    """ Joue avec un modèle entraîné (inférence seulement). """
    game = SnakeGameAI(render=render)
    agent = Agent()
    try:
        agent.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        agent.model.eval()
    except Exception as e:
        print("Impossible de charger le modèle, joue aléatoire :", e)

    while True:
        state_old = agent.get_state(game)
        with torch.no_grad():
            prediction = agent.model(torch.tensor(state_old, dtype=torch.float32))
            move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1

        reward, done, score = game.play_step(final_move)
        if done:
            print("Score:", score)
            game.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Affiche le jeu pendant l'entraînement (lent).")
    parser.add_argument("--play", action="store_true", help="Joue avec le modèle entraîné (best_model.pth).")
    parser.add_argument("--games", type=int, default=500, help="Nombre de parties pour l'entraînement.")
    args = parser.parse_args()

    if args.play:
        play_trained(render=True)
    else:
        train(render=args.render, max_games=args.games)


if __name__ == "__main__":
    main()