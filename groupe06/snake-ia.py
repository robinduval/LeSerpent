"""
snake-ia.py - Agent IA basé sur Deep Q-Learning
S'intègre avec le jeu Snake de serpent.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import argparse
import sys
import time
import pygame
from serpent import (
    Snake, Apple, GRID_SIZE, UP, DOWN, LEFT, RIGHT,
    SCREEN_WIDTH, SCREEN_HEIGHT, SCORE_PANEL_HEIGHT,
    draw_grid, display_info, display_message,
    GRIS_FOND, NOIR
)

# ============================================================================
# PARTIE 1: DEEP Q-LEARNING
# ============================================================================

class Linear_QNet(nn.Module):
    """
    Réseau neuronal pour l'approximation de la fonction Q.
    Input: vecteur d'état amélioré avec distances normalisées
    Output: 4 Q-valeurs (une par direction)
    """
    def __init__(self, input_size=15, hidden_size=512, output_size=4):
        super(Linear_QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, 256)
        self.fc4 = nn.Linear(256, output_size)
        
    def forward(self, x):
        """Forward pass du réseau avec dropout pour éviter l'overfitting."""
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class QTrainer:
    """
    Entraîneur pour le modèle Q-Learning avec PyTorch.
    Implémente la boucle d'entraînement et les mises à jour du réseau.
    """
    def __init__(self, model, lr=0.001, gamma=0.9):
        """
        Args:
            model: Réseau neuronal Linear_QNet
            lr: Learning rate
            gamma: Facteur de discount
        """
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, game_over):
        """
        Effectue une étape d'entraînement sur une transition.
        
        Args:
            state: État actuel (tenseur)
            action: Action prise (index 0-3)
            reward: Récompense reçue
            next_state: État suivant (tenseur)
            game_over: Booléen indiquant si le jeu est terminé
        """
        # Conversion en tenseurs si nécessaire
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32)
        
        # Prédictions Q pour l'état actuel
        prediction = self.model(state)
        target = prediction.clone()
        
        # Calcul de la Q-valeur cible
        if game_over:
            q_new = reward
        else:
            q_new = reward + self.gamma * torch.max(self.model(next_state))
        
        # Mise à jour de la Q-valeur pour l'action prise
        target[action] = q_new
        
        # Calcul de la perte et optimisation
        loss = self.criterion(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Agent:
    """
    Agent d'apprentissage par renforcement utilisant Deep Q-Learning.
    Gère la mémoire, l'exploration et l'exploitation, l'entraînement.
    """
    def __init__(self, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, learning_rate=0.001, batch_size=32, max_memory=100000):
        """
        Initialisation de l'agent IA.
        
        Args:
            gamma: Facteur de discount
            epsilon: Paramètre d'exploration (epsilon-greedy)
            epsilon_decay: Taux de décroissance de epsilon
            epsilon_min: Valeur minimale de epsilon
            learning_rate: Taux d'apprentissage
            batch_size: Taille des batches d'entraînement
            max_memory: Taille maximale de la mémoire de replay
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_memory = max_memory
        
        # Modèle et entraîneur
        self.model = Linear_QNet()
        self.trainer = QTrainer(self.model, lr=learning_rate, gamma=gamma)
        
        # Mémoire de replay (state, action, reward, next_state, game_over)
        self.memory = deque(maxlen=max_memory)
        
    def get_state(self, snake, apple):
        """
        Construit un vecteur d'état enrichi pour ignorer le corps et se concentrer sur la pomme.
        
        État: 15 caractéristiques
        - 3 dangers (face, droite, gauche)
        - 4 directions actuelles
        - 4 positions relatives de la pomme
        - Distance normalisée à la pomme
        - Taille du serpent (normalisée)
        
        Args:
            snake: Objet Snake
            apple: Objet Apple
            
        Returns:
            np.array: Vecteur d'état de 15 éléments
        """
        head_x, head_y = snake.head_pos
        apple_x, apple_y = apple.position
        direction = snake.direction
        
        # Fonction pour vérifier les collisions (strictement les murs et l'auto-morsure)
        def is_collision(x, y):
            if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
                return True
            if [x, y] in snake.body[1:]:
                return True
            return False
        
        # Détection des dangers dans chaque direction
        danger_up = is_collision(head_x + UP[0], head_y + UP[1])
        danger_down = is_collision(head_x + DOWN[0], head_y + DOWN[1])
        danger_left = is_collision(head_x + LEFT[0], head_y + LEFT[1])
        danger_right = is_collision(head_x + RIGHT[0], head_y + RIGHT[1])
        
        # Danger face (selon la direction actuelle)
        if direction == UP:
            danger_face = danger_up
            danger_right_side = danger_right
            danger_left_side = danger_left
        elif direction == DOWN:
            danger_face = danger_down
            danger_right_side = danger_left
            danger_left_side = danger_right
        elif direction == LEFT:
            danger_face = danger_left
            danger_right_side = danger_up
            danger_left_side = danger_down
        else:  # RIGHT
            danger_face = danger_right
            danger_right_side = danger_down
            danger_left_side = danger_up
        
        # Calcul de la distance à la pomme (distance de Manhattan)
        manhattan_distance = abs(apple_x - head_x) + abs(apple_y - head_y)
        # Normalisation entre 0 et 1
        max_distance = 2 * GRID_SIZE
        normalized_distance = manhattan_distance / max_distance
        
        # Taille du serpent normalisée
        snake_size_normalized = len(snake.body) / (GRID_SIZE * GRID_SIZE)
        
        # Construction du vecteur d'état amélioré
        state = np.array([
            danger_face,
            danger_right_side,
            danger_left_side,
            direction == UP,
            direction == DOWN,
            direction == LEFT,
            direction == RIGHT,
            apple_x < head_x,
            apple_x > head_x,
            apple_y < head_y,
            apple_y > head_y,
            normalized_distance,          # Distance à la pomme
            snake_size_normalized,         # Taille du serpent
            1.0 if manhattan_distance > 1 else 0.0,  # Est loin de la pomme
            1.0 if [apple_x, apple_y] in snake.body else 0.0,  # Pomme accessible
        ], dtype=np.float32)
        
        return state
    
    def remember(self, state, action, reward, next_state, game_over):
        """Stocke une transition dans la mémoire de replay."""
        self.memory.append((state, action, reward, next_state, game_over))
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        """Entraîne le modèle sur une seule transition."""
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        self.trainer.train_step(state_tensor, action, reward, next_state_tensor, game_over)
    
    def train_long_memory(self):
        """Entraîne le modèle sur un batch aléatoire de la mémoire."""
        if len(self.memory) < self.batch_size:
            mini_sample = list(self.memory)
        else:
            mini_sample = random.sample(list(self.memory), self.batch_size)
        
        # Conversion efficace en numpy d'abord, puis en tenseur (beaucoup plus rapide)
        states = np.array([sample[0] for sample in mini_sample], dtype=np.float32)
        actions = np.array([sample[1] for sample in mini_sample], dtype=np.int64)
        rewards = np.array([sample[2] for sample in mini_sample], dtype=np.float32)
        next_states = np.array([sample[3] for sample in mini_sample], dtype=np.float32)
        game_overs = np.array([sample[4] for sample in mini_sample], dtype=bool)
        
        # Conversion en tenseurs PyTorch (une seule fois, très rapide)
        states_tensor = torch.from_numpy(states)
        next_states_tensor = torch.from_numpy(next_states)
        rewards_tensor = torch.from_numpy(rewards)
        
        # Prédictions pour les états actuels
        predictions = self.model(states_tensor)
        targets = predictions.clone()
        
        for idx in range(len(mini_sample)):
            if game_overs[idx]:
                q_new = rewards_tensor[idx]
            else:
                q_new = rewards_tensor[idx] + self.gamma * torch.max(self.model(next_states_tensor[idx:idx+1]))
            targets[idx, actions[idx]] = q_new
        
        # Entraînement
        loss = self.trainer.criterion(predictions, targets)
        self.trainer.optimizer.zero_grad()
        loss.backward()
        self.trainer.optimizer.step()
        
        # Décroissance de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_action(self, state, valid_actions=None):
        """Choisit une action selon la stratégie epsilon-greedy."""
        if valid_actions is None:
            valid_actions = [0, 1, 2, 3]
        
        # Exploration (aléatoire)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation (meilleure action connue)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(state_tensor)
        
        # Filtre les actions non valides
        for i in range(len(predictions)):
            if i not in valid_actions:
                predictions[i] = float('-inf')
        
        return torch.argmax(predictions).item()
    
    def save_model(self, path="snake_model.pth"):
        """Sauvegarde le modèle sur disque."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path="snake_model.pth"):
        """Charge le modèle depuis disque."""
        self.model.load_state_dict(torch.load(path))


# ============================================================================
# PARTIE 2: BOUCLE PRINCIPALE ET INTÉGRATION
# ============================================================================

def get_valid_actions(snake):
    """Retourne les actions valides (sans inversion de direction)."""
    current_dir = snake.direction
    valid = []
    
    if current_dir != DOWN:
        valid.append(0)  # UP
    if current_dir != UP:
        valid.append(1)  # DOWN
    if current_dir != RIGHT:
        valid.append(2)  # LEFT
    if current_dir != LEFT:
        valid.append(3)  # RIGHT
    
    return valid


def train_dqn(episodes=1000, render=True):
    """Entraîne l'agent Deep Q-Learning avec des récompenses améliorées."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake IA - Entraînement DQN")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    
    agent = Agent(learning_rate=0.0005, gamma=0.95, epsilon_decay=0.9995)
    
    print(f"Entraînement DQN pour {episodes} épisodes...")
    
    start_time = time.time()
    
    for episode in range(episodes):
        snake = Snake()
        apple = Apple(snake.body)
        state_old = agent.get_state(snake, apple)
        
        total_reward = 0
        steps = 0
        max_steps = 1000
        move_counter = 0
        game_over = False
        prev_distance = abs(apple.position[0] - snake.head_pos[0]) + abs(apple.position[1] - snake.head_pos[1])
        
        while not game_over and steps < max_steps:
            # Vérifie les événements pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Choix d'une action valide
            valid_actions = get_valid_actions(snake)
            action = agent.get_action(state_old, valid_actions)
            
            # Exécution de l'action
            action_map = [UP, DOWN, LEFT, RIGHT]
            snake.set_direction(action_map[action])
            snake.move()
            
            reward = 0
            
            # Vérification des collisions
            if snake.is_game_over():
                reward = -10
                game_over = True
            # Vérification de la pomme mangée
            elif snake.head_pos == list(apple.position):
                snake.grow()
                reward = 50  # Grosse récompense pour la pomme
                if not apple.relocate(snake.body):
                    game_over = True
            else:
                # Récompenses intermédiaires : encourager à se rapprocher de la pomme
                current_distance = abs(apple.position[0] - snake.head_pos[0]) + abs(apple.position[1] - snake.head_pos[1])
                
                if current_distance < prev_distance:
                    reward = 1  # Petit bonus pour se rapprocher
                elif current_distance > prev_distance:
                    reward = -0.5  # Petite pénalité pour s'éloigner
                else:
                    reward = -0.1  # Légère pénalité pour ne rien faire
                
                prev_distance = current_distance
            
            total_reward += reward
            
            # Nouvel état
            state_new = agent.get_state(snake, apple)
            
            # Entraînement court terme
            agent.train_short_memory(state_old, action, reward, state_new, game_over)
            
            # Mémorisation
            agent.remember(state_old, action, reward, state_new, game_over)
            
            state_old = state_new
            steps += 1
            
            # Affichage occasionnel
            if render and episode % 50 == 0:
                move_counter += 1
                if move_counter >= 5:
                    # Dessiner
                    screen.fill(GRIS_FOND)
                    game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
                    pygame.draw.rect(screen, NOIR, game_area_rect)
                    draw_grid(screen)
                    apple.draw(screen)
                    snake.draw(screen)
                    display_info(screen, font_main, snake, time.time())
                    pygame.display.flip()
                    move_counter = 0
        
        # Entraînement long terme
        agent.train_long_memory()
        
        if (episode + 1) % 50 == 0:
            print(f"Épisode {episode + 1}/{episodes} - Score: {snake.score} - "
                  f"Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.4f}")
    
    # Sauvegarde du modèle
    agent.save_model("snake_model.pth")
    print("✓ Modèle sauvegardé dans 'snake_model.pth'")
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"📊 Entraînement terminé en {minutes}m {seconds:.1f}s")
    pygame.quit()


def play_dqn(episodes=10, render=True, model_path="snake_model.pth"):
    """Joue avec le modèle DQN entraîné."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake IA - Mode Jeu DQN")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    
    agent = Agent(epsilon=0.0)  # Pas d'exploration en mode jeu
    try:
        agent.load_model(model_path)
        print(f"✓ Modèle chargé depuis '{model_path}'")
    except FileNotFoundError:
        print(f"⚠️  Modèle non trouvé: {model_path}")
        print("💡 Entraîner d'abord: python3 snake-ia.py --train")
        pygame.quit()
        return
    
    print(f"Jeu avec DQN pour {episodes} parties...")
    
    global_start_time = time.time()
    
    for episode in range(episodes):
        snake = Snake()
        apple = Apple(snake.body)
        state = agent.get_state(snake, apple)
        
        total_reward = 0
        steps = 0
        max_steps = 1000
        move_counter = 0
        game_over = False
        
        start_time = time.time()
        
        while not game_over and steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            valid_actions = get_valid_actions(snake)
            action = agent.get_action(state, valid_actions)
            
            action_map = [UP, DOWN, LEFT, RIGHT]
            snake.set_direction(action_map[action])
            snake.move()
            
            reward = 0
            if snake.is_game_over():
                reward = -10
                game_over = True
            elif snake.head_pos == list(apple.position):
                snake.grow()
                reward = 10
                if not apple.relocate(snake.body):
                    game_over = True
            
            total_reward += reward
            state = agent.get_state(snake, apple)
            steps += 1
            
            if render:
                move_counter += 1
                if move_counter >= 2:
                    screen.fill(GRIS_FOND)
                    game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
                    pygame.draw.rect(screen, NOIR, game_area_rect)
                    draw_grid(screen)
                    apple.draw(screen)
                    snake.draw(screen)
                    display_info(screen, font_main, snake, start_time)
                    pygame.display.flip()
                    clock.tick(10)
                    move_counter = 0
        
        print(f"Partie {episode + 1}/{episodes} - Score: {snake.score} - Steps: {steps}")
    
    elapsed_time = time.time() - global_start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"🎮 Jeu terminé en {minutes}m {seconds:.1f}s")
    pygame.quit()


def main():
    """Fonction principale avec comportement intelligent."""
    import os
    
    parser = argparse.ArgumentParser(
        description='Snake IA - Deep Q-Learning',
        usage='python3 snake-ia.py [--train EPISODES] [--play EPISODES] [--no-render]'
    )
    
    # Déterminer le comportement par défaut
    model_exists = os.path.exists("snake_model.pth")
    
    parser.add_argument('--train', type=int, default=None, metavar='N',
                        help='Entraîner pour N épisodes (défaut: 1000 si pas de modèle)')
    parser.add_argument('--play', type=int, default=None, metavar='N',
                        help='Jouer N parties (défaut: 10 si modèle existe)')
    parser.add_argument('--no-render', action='store_true',
                        help='Désactiver l\'affichage')
    
    args = parser.parse_args()
    
    # Comportement intelligent
    train_mode = args.train is not None
    play_mode = args.play is not None
    
    # Si aucun mode spécifié: entraîner si pas de modèle, sinon jouer
    if not train_mode and not play_mode:
        if model_exists:
            play_mode = True
            args.play = 10
            print("🎮 Mode jeu (modèle trouvé)")
        else:
            train_mode = True
            args.train = 1000
            print("🤖 Mode entraînement (modèle absent)")
    
    render = not args.no_render
    
    if train_mode:
        episodes = args.train or 1000
        print(f"📚 Entraînement pendant {episodes} épisodes...")
        train_dqn(episodes=episodes, render=render)
    
    if play_mode:
        episodes = args.play or 10
        print(f"🎮 Jeu pendant {episodes} parties...")
        play_dqn(episodes=episodes, render=render)


if __name__ == '__main__':
    main()
