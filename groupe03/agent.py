import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, UP, DOWN, LEFT, RIGHT, GRID_SIZE
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    """Agent Deep Q-Learning pour jouer au Snake"""
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Paramètre de randomness
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() si dépassement
        self.model = Linear_QNet(11, 256, 3)  # 11 états en entrée, 3 actions en sortie
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, game):
        """
        Obtient l'état actuel du jeu
        
        État composé de 11 valeurs:
        - Danger tout droit, à droite, à gauche (3)
        - Direction actuelle (4 booléens)
        - Position de la nourriture (4 booléens)
        """
        head = game.snake[0]
        point_l = [head[0] - 1, head[1]]
        point_r = [head[0] + 1, head[1]]
        point_u = [head[0], head[1] - 1]
        point_d = [head[0], head[1] + 1]
        
        dir_l = game.direction == LEFT
        dir_r = game.direction == RIGHT
        dir_u = game.direction == UP
        dir_d = game.direction == DOWN
        
        state = [
            # Danger tout droit
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            # Danger à droite
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            
            # Danger à gauche
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            # Direction actuelle
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Position de la nourriture
            game.food[0] < game.head[0],  # Nourriture à gauche
            game.food[0] > game.head[0],  # Nourriture à droite
            game.food[1] < game.head[1],  # Nourriture en haut
            game.food[1] > game.head[1]   # Nourriture en bas
        ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        """Stocke l'expérience dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        """Entraîne sur un batch d'expériences de la mémoire"""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Liste de tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        """Entraîne sur une seule étape"""
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        """
        Détermine l'action à prendre
        Utilise l'exploration (random) vs exploitation (modèle)
        """
        # Random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games  # Plus on joue, moins on explore
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move


def train():
    """Fonction principale d'entraînement"""
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(display=False)  # Mode rapide sans affichage graphique
    
    print("=" * 70)
    print("🐍 ENTRAÎNEMENT DEEP Q-LEARNING - SNAKE AI")
    print("=" * 70)
    print(f"📊 Paramètres:")
    print(f"   - Learning Rate: {LR}")
    print(f"   - Gamma (discount): {agent.gamma}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Max Memory: {MAX_MEMORY}")
    print(f"   - Architecture: 11 → 256 → 3")
    print("=" * 70)
    print()
    
    while True:
        # Obtenir l'état actuel
        state_old = agent.get_state(game)
        
        # Obtenir l'action
        final_move = agent.get_action(state_old)
        
        # Effectuer l'action et obtenir le nouvel état
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Entraîner la mémoire courte
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Se souvenir
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # Entraîner la mémoire longue (replay memory)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save('best_model_dqn.pth')
                print(f'🏆 NOUVEAU RECORD! Game {agent.n_games} | Score: {score} ⭐')
            else:
                print(f'Game {agent.n_games:4d} | Score: {score:3d} | Record: {record:3d}', end='')
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            if score < record:
                print(f' | Moyenne: {mean_score:.2f} | Epsilon: {max(0, agent.epsilon)}')
            
            # Optionnel: sauvegarder tous les 10 jeux
            if agent.n_games % 10 == 0:
                agent.model.save(f'model_game_{agent.n_games}.pth')


if __name__ == '__main__':
    train()
