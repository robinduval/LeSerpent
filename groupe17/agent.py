"""
AGENT.PY - RL Agent avec Deep Q-Learning
L'extraction de l'état (get_state) est maintenant dans snake-ia.py
"""
import torch
import random
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 500
LR = 0.001  # Learning rate

class Agent:
    """
    Agent d'apprentissage par renforcement utilisant Deep Q-Learning
    """

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Contrôle l'exploration (randomness)
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience replay memory

        # Modèle: 11 inputs -> 256 hidden -> 3 outputs
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        """Stocke l'expérience dans la mémoire (experience replay)"""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Entraîne sur un batch d'expériences de la mémoire
        C'est l'experience replay qui stabilise l'apprentissage
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Entraîne sur une seule expérience (step actuel)"""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Décide de l'action à prendre
        Utilise epsilon-greedy: exploration vs exploitation
        """
        # Exploration vs exploitation
        # Plus on joue, moins on explore (epsilon diminue)
        self.epsilon = max(0, 150 - self.n_games)  # Exploration prolongée
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # Exploration: action aléatoire
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation: utiliser le modèle
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
