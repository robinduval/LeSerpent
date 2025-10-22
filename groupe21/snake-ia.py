"""
Snake Game AI using Deep Q-Learning (DQN)
Groupe 21
"""

import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import math
import os
from typing import List, Tuple

# --- CONSTANTES DE JEU ---
# Taille de la grille (15x15)
GRID_SIZE = 15
# Taille d'une cellule en pixels
CELL_SIZE = 30
# Vitesse de jeu (images par seconde)
GAME_SPEED = 60

# Dimensions de l'écran (avec espace pour le score/timer)
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (en RGB)
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)  # Tête du serpent
VERT = (0, 200, 0)     # Corps du serpent
ROUGE = (200, 0, 0)    # Pomme
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)
BLEU_CLAIR = (135, 206, 250)  # Pour la visualisation des états
BLEU_FONCE = (0, 0, 139)      # Pour la visualisation des actions

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Paramètres DQN
MEMORY_SIZE = 200_000  # Plus grande mémoire pour plus d'expérience
BATCH_SIZE = 128      # Plus grand batch pour un apprentissage plus stable
GAMMA = 0.99          # Facteur d'actualisation plus élevé pour valoriser les récompenses futures
EPSILON = 1.0         # Taux d'exploration initial
EPSILON_MIN = 0.01    # Taux d'exploration minimal
EPSILON_DECAY = 0.997 # Décroissance plus lente pour plus d'exploration
LEARNING_RATE = 0.0005  # Learning rate plus petit pour un apprentissage plus stable
TARGET_UPDATE = 5     # Mise à jour plus fréquente du réseau cible

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.memory)

class SnakeAI:
    def __init__(self):
        """Initialise le serpent avec une position de départ."""
        self.reset()
    
    def reset(self):
        """Réinitialise le serpent à sa position de départ."""
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos, 
                    [self.head_pos[0] - 1, self.head_pos[1]],
                    [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.score = 0
        self.steps_without_apple = 0
    
    def move(self, action: int) -> bool:
        """
        Déplace le serpent selon l'action choisie.
        Retourne True si le mouvement est valide, False si collision.
        """
        # Convertit l'action en direction
        if action == UP and self.direction != DOWN:
            self.direction = UP
        elif action == RIGHT and self.direction != LEFT:
            self.direction = RIGHT
        elif action == DOWN and self.direction != UP:
            self.direction = DOWN
        elif action == LEFT and self.direction != RIGHT:
            self.direction = LEFT
        
        # Calcule la nouvelle position
        dx = 1 if self.direction == RIGHT else -1 if self.direction == LEFT else 0
        dy = 1 if self.direction == DOWN else -1 if self.direction == UP else 0
        
        new_head = [
            (self.head_pos[0] + dx) % GRID_SIZE,
            (self.head_pos[1] + dy) % GRID_SIZE
        ]
        
        # Vérifie les collisions
        if new_head in self.body[:-1]:
            return False
        
        self.body.insert(0, new_head)
        self.head_pos = new_head
        self.body.pop()
        self.steps_without_apple += 1
        
        return True
    
    def grow(self):
        """Fait grandir le serpent."""
        self.body.append(self.body[-1])
        self.score += 1
        self.steps_without_apple = 0
    
    def get_state(self, apple_pos):
        """
        Retourne l'état du jeu sous forme de vecteur pour le DQN.
        État amélioré avec plus d'informations sur l'environnement.
        """
        state = []
        
        # Directions possibles relatives à la direction actuelle
        directions = self._get_relative_directions()
        
        # Danger dans chaque direction relative (8 directions au lieu de 3)
        for dx, dy in [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]:
            next_pos = [
                (self.head_pos[0] + dx) % GRID_SIZE,
                (self.head_pos[1] + dy) % GRID_SIZE
            ]
            state.append(1.0 if next_pos in self.body[:-1] else 0.0)
        
        # Direction actuelle one-hot encoded
        for i in range(4):
            state.append(1.0 if self.direction == i else 0.0)
        
        # Position relative de la pomme avec distances normalisées
        apple_dir = [
            apple_pos[0] - self.head_pos[0],
            apple_pos[1] - self.head_pos[1]
        ]
        
        # Gestion du wraparound pour la distance à la pomme
        if apple_dir[0] > GRID_SIZE // 2:
            apple_dir[0] = apple_dir[0] - GRID_SIZE
        elif apple_dir[0] < -GRID_SIZE // 2:
            apple_dir[0] = apple_dir[0] + GRID_SIZE
            
        if apple_dir[1] > GRID_SIZE // 2:
            apple_dir[1] = apple_dir[1] - GRID_SIZE
        elif apple_dir[1] < -GRID_SIZE // 2:
            apple_dir[1] = apple_dir[1] + GRID_SIZE
        
        # Distance normalisée à la pomme
        dist_to_apple = math.sqrt(apple_dir[0]**2 + apple_dir[1]**2) / GRID_SIZE
        state.append(dist_to_apple)
        
        # Direction relative de la pomme (angles)
        angle_to_apple = math.atan2(apple_dir[1], apple_dir[0]) / math.pi
        state.append(angle_to_apple)
        
        # Proximité aux murs (distances normalisées)
        state.append(self.head_pos[0] / GRID_SIZE)  # distance au mur gauche
        state.append((GRID_SIZE - self.head_pos[0]) / GRID_SIZE)  # distance au mur droit
        state.append(self.head_pos[1] / GRID_SIZE)  # distance au mur haut
        state.append((GRID_SIZE - self.head_pos[1]) / GRID_SIZE)  # distance au mur bas
        
        # Densité du serpent dans chaque quadrant
        quadrants = [[0,0], [0,0], [0,0], [0,0]]
        for segment in self.body:
            quad_x = 0 if segment[0] < GRID_SIZE//2 else 1
            quad_y = 0 if segment[1] < GRID_SIZE//2 else 1
            quadrants[quad_y*2 + quad_x][1] += 1
            quadrants[quad_y*2 + quad_x][0] = (quad_y*2 + quad_x)
        
        # Normaliser et ajouter les densités
        total_length = len(self.body)
        for quad in quadrants:
            state.append(quad[1] / total_length)
        
        # Ajout : direction de la queue (one-hot)
        tail_dir = [0, 0, 0, 0]
        if len(self.body) > 1:
            tail_vec = [self.body[-1][0] - self.body[-2][0], self.body[-1][1] - self.body[-2][1]]
            if tail_vec == [0, -1]: tail_dir[0] = 1  # UP
            elif tail_vec == [1, 0]: tail_dir[1] = 1  # RIGHT
            elif tail_vec == [0, 1]: tail_dir[2] = 1  # DOWN
            elif tail_vec == [-1, 0]: tail_dir[3] = 1  # LEFT
        state.extend(tail_dir)

        # Ajout : distance à la queue (euclidienne, normalisée)
        tail_dist = math.sqrt((self.head_pos[0] - self.body[-1][0])**2 + (self.head_pos[1] - self.body[-1][1])**2) / GRID_SIZE
        state.append(tail_dist)

        # Ajout : longueur du serpent (normalisée)
        state.append(len(self.body) / (GRID_SIZE * GRID_SIZE))
        
        return np.array(state, dtype=np.float32)
    
    def _get_relative_directions(self):
        """Retourne les directions relatives (devant, droite, gauche) selon la direction actuelle."""
        if self.direction == UP:
            return [(0, -1), (1, 0), (-1, 0)]  # devant, droite, gauche
        elif self.direction == RIGHT:
            return [(1, 0), (0, 1), (0, -1)]
        elif self.direction == DOWN:
            return [(0, 1), (-1, 0), (1, 0)]
        else:  # LEFT
            return [(-1, 0), (0, -1), (0, 1)]
    
    def draw(self, surface):
        """Dessine le serpent sur la surface de jeu."""
        # Dessine le corps
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)  # Bordure
        
        # Dessine la tête
        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 1)  # Bordure

class Apple:
    def __init__(self, snake_body):
        """Initialise la pomme avec une position aléatoire valide."""
        self.position = self._get_random_position(snake_body)
    
    def _get_random_position(self, snake_body):
        """Trouve une position aléatoire non occupée par le serpent."""
        all_positions = [
            (x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
            if [x, y] not in snake_body
        ]
        if all_positions:
            return random.choice(all_positions)
        return None
    
    def relocate(self, snake_body):
        """Déplace la pomme à une nouvelle position aléatoire."""
        new_position = self._get_random_position(snake_body)
        if new_position:
            self.position = new_position
            return True
        return False
    
    def draw(self, surface):
        """Dessine la pomme sur la surface donnée."""
        if self.position:
            rect = pygame.Rect(
                self.position[0] * CELL_SIZE,
                self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(surface, ROUGE, rect)

class SnakeGame:
    def __init__(self):
        """Initialise le jeu."""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake - Deep Q-Learning - Groupe 21")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        self.snake = SnakeAI()
        self.apple = Apple(self.snake.body)
        self.reset()
    
    def reset(self):
        """Réinitialise le jeu."""
        self.snake.reset()
        self.apple = Apple(self.snake.body)
        self.game_over = False
        return self.snake.get_state(self.apple.position)
    
    def step(self, action):
        """
        Effectue une action et retourne (next_state, reward, done, score).
        Système de récompense amélioré avec plus de feedback.
        """
        old_distance = math.sqrt(
            (self.snake.head_pos[0] - self.apple.position[0])**2 +
            (self.snake.head_pos[1] - self.apple.position[1])**2
        )
        
        # Récompense liée à la distance à la queue
        old_tail_dist = math.sqrt((self.snake.head_pos[0] - self.snake.body[-1][0])**2 + (self.snake.head_pos[1] - self.snake.body[-1][1])**2)
        
        # Applique l'action
        if not self.snake.move(action):
            self.game_over = True
            # Pénalité plus sévère pour les collisions précoces
            reward = -10 - (15 * (1 - len(self.snake.body)/(GRID_SIZE*GRID_SIZE)))
            return self.snake.get_state(self.apple.position), reward, True, self.snake.score
        
        # Calcule la nouvelle distance à la pomme
        new_distance = math.sqrt(
            (self.snake.head_pos[0] - self.apple.position[0])**2 +
            (self.snake.head_pos[1] - self.apple.position[1])**2
        )
        
        # Calcule la nouvelle distance à la queue
        new_tail_dist = math.sqrt((self.snake.head_pos[0] - self.snake.body[-1][0])**2 + (self.snake.head_pos[1] - self.snake.body[-1][1])**2)
        
        # Récompense de progression vers la pomme
        reward = old_distance - new_distance
        
        # Vérifie si le serpent mange la pomme
        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            # Récompense progressive selon la taille du serpent
            reward = 10 + (len(self.snake.body) * 0.5)
            if not self.apple.relocate(self.snake.body):
                # Victoire !
                reward = 100 + (len(self.snake.body) * 2)
                self.game_over = True
                return self.snake.get_state(self.apple.position), reward, True, self.snake.score
        
        # Pénalité adaptative pour le temps sans manger
        max_steps = GRID_SIZE * GRID_SIZE * 2
        if self.snake.steps_without_apple > max_steps:
            self.game_over = True
            reward = -30  # Punition plus forte
            return self.snake.get_state(self.apple.position), reward, True, self.snake.score

        # Ajoute une pénalité par pas si on tourne trop longtemps
        if self.snake.steps_without_apple > GRID_SIZE:
            reward -= 0.2  # Pénalité supplémentaire par pas après un certain temps sans manger

        # Bonus si rapprochement de la queue
        if new_tail_dist < old_tail_dist:
            reward += 0.2
        else:
            reward -= 0.2  # Pénalité si éloignement

        # Pénalité pour demi-tour
        if (action == 0 and self.snake.direction == 2) or (action == 2 and self.snake.direction == 0) or (action == 1 and self.snake.direction == 3) or (action == 3 and self.snake.direction == 1):
            reward -= 0.5
        
        # Petit bonus pour survivre
        reward += 0.1
        
        return self.snake.get_state(self.apple.position), reward, False, self.snake.score
    
    def draw(self):
        """Dessine l'état actuel du jeu."""
        self.screen.fill(GRIS_FOND)
        
        # Zone de jeu
        game_area = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.screen, NOIR, game_area)
        
        # Grille
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(
                    x * CELL_SIZE,
                    y * CELL_SIZE + SCORE_PANEL_HEIGHT,
                    CELL_SIZE,
                    CELL_SIZE
                )
                pygame.draw.rect(self.screen, GRIS_GRILLE, rect, 1)
        
        # Serpent et pomme
        self.snake.draw(self.screen)
        self.apple.draw(self.screen)
        
        # Score et infos
        score_text = self.font.render(f'Score: {self.snake.score}', True, BLANC)
        self.screen.blit(score_text, (10, 20))
        
        if self.game_over:
            game_over_text = self.font.render('GAME OVER', True, ROUGE)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()
    
    def close(self):
        """Ferme le jeu."""
        pygame.quit()

def load_model(model_path, state_size, action_size):
    """Charge un modèle préentraîné."""
    model = DQN(state_size, action_size)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Modèle chargé depuis {model_path}")
    return model

def save_model(model, model_path):
    """Sauvegarde le modèle."""
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé dans {model_path}")

def train(load_existing=False):
    # Configuration de l'environnement et du DQN
    env = SnakeGame()
    state_size = 28  # 8 dangers + 4 direction + 1 distance + 1 angle + 4 murs + 4 quadrants + 4 tail_dir + 1 tail_dist + 1 snake length
    action_size = 4  # Nombre d'actions possibles
    
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "snake_dqn_best.pth")
    
    # Initialisation ou chargement des réseaux
    if load_existing and os.path.exists(model_path):
        policy_net = load_model(model_path, state_size, action_size)
        print("Chargement du modèle existant...")
    else:
        policy_net = DQN(state_size, action_size)
        print("Création d'un nouveau modèle...")
    
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    
    memory = ReplayMemory(MEMORY_SIZE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    epsilon = EPSILON
    
    # Pour le suivi des performances
    best_score = 0
    scores_window = deque(maxlen=100)
    
    # Statistiques d'entraînement
    episode_rewards = []
    episode_lengths = []
    best_score = 0
    
    # Boucle d'entraînement principale
    for episode in range(1000):  # Nombre d'épisodes
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            # Sélection de l'action (epsilon-greedy)
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = policy_net(state_tensor).argmax().item()
            
            # Exécution de l'action
            next_state, reward, done, score = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Stockage dans la mémoire de replay
            memory.push(state, action, reward, next_state, done)
            state = next_state
            
            # Apprentissage si assez d'exemples
            if len(memory) > BATCH_SIZE:
                # Échantillonnage du batch
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                # Calcul des valeurs Q actuelles
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))
                
                # Calcul des valeurs Q cibles
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards + (1 - dones) * GAMMA * max_next_q_values
                
                # Calcul de la perte et optimisation
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Affichage
            env.draw()
            env.clock.tick(GAME_SPEED)
            
            if done:
                break
            
            # Gestion des événements pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        # Mise à jour du réseau cible
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Décroissance d'epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        # Enregistrement des statistiques
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Affichage des statistiques
        avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
        if score > best_score:
            best_score = score
            # Sauvegarde du meilleur modèle
            save_model(policy_net, model_path)
            print(f"Nouveau meilleur score ! {score} - Modèle sauvegardé")
            
        scores_window.append(score)
        avg_score = sum(scores_window) / len(scores_window)
        
        print(f"Episode {episode + 1}, Score: {score}, Moy(100): {avg_score:.2f}, Best: {best_score}, Epsilon: {epsilon:.3f}")
    
    env.close()

def play_model(model_path=None):
    """Joue au snake avec un modèle entraîné."""
    env = SnakeGame()
    state_size = 28
    action_size = 4
    
    # Charge le modèle s'il existe
    if model_path and os.path.exists(model_path):
        model = load_model(model_path, state_size, action_size)
        model.eval()  # Mode évaluation
    else:
        print("Aucun modèle trouvé. Utilisation d'un nouveau modèle.")
        model = DQN(state_size, action_size)
    
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        # Sélection de l'action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = model(state_tensor).argmax().item()
        
        # Exécution de l'action
        state, reward, done, score = env.step(action)
        
        # Affichage
        env.draw()
        env.clock.tick(15)  # Ralentit le jeu pour mieux voir
        
        # Gestion des événements pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Appuyez sur 'Q' pour quitter
                    done = True
    
    print(f"Jeu terminé! Score final: {score}")
    env.close()

if __name__ == "__main__":
    # Par défaut, on entraîne
    mode = input("Entrez 't' pour entraîner, 'p' pour jouer avec le modèle entraîné: ").lower()
    
    if mode == 'p':
        model_path = os.path.join("models", "snake_dqn_best.pth")
        play_model(model_path)
    else:
        # Demande si on veut charger un modèle existant
        load_existing = input("Charger le dernier modèle? (o/n): ").lower() == 'o'
        train(load_existing)