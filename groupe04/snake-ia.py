import pygame
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Tuple

# --- CONSTANTES DE JEU ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 50  # Très rapide pour l'entraînement

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)
VIOLET = (150, 0, 255)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Hyperparamètres du DQN
LEARNING_RATE = 0.001
GAMMA = 0.95  # Facteur de discount
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 10

# --- CLASSES DU JEU ---

class Snake:
    """Représente le serpent."""
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos, 
                     [self.head_pos[0] - 1, self.head_pos[1]], 
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        """Change la direction, empêchant le mouvement inverse."""
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        """Déplace le serpent."""
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
        """Prépare le serpent à grandir."""
        self.grow_pending = True
        self.score += 1

    def check_self_collision(self):
        """Vérifie si la tête touche le corps."""
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si le jeu est terminé."""
        return self.check_self_collision()

    def draw(self, surface):
        """Dessine le serpent."""
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)

        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)


class Apple:
    """Représente la pomme."""
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aléatoire non occupée."""
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(pos) not in occupied_positions]
        
        if not available_positions:
            return None
            
        return random.choice(available_positions)

    def relocate(self, snake_body):
        """Déplace la pomme."""
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False

    def draw(self, surface):
        """Dessine la pomme."""
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)


# --- DEEP Q-NETWORK ---

class Linear_QNet(nn.Module):
    """
    Réseau de neurones pour le Deep Q-Learning.
    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass du réseau."""
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ReplayMemory:
    """Mémoire de replay pour stocker les expériences."""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Ajoute une expérience à la mémoire."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Échantillonne un batch aléatoire."""
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """Agent utilisant Deep Q-Learning pour jouer au Snake."""
    
    def __init__(self, state_size=11, action_size=4, hidden_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START
        
        # Réseaux de neurones
        self.model = Linear_QNet(state_size, hidden_size, action_size)
        self.target_model = Linear_QNet(state_size, hidden_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimiseur et fonction de perte
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        # Mémoire de replay
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Compteurs
        self.n_games = 0
        self.steps = 0
    
    def get_state(self, snake: Snake, apple: Apple) -> np.ndarray:
        """
        Crée l'état actuel du jeu pour l'agent.
        État: [danger_straight, danger_right, danger_left,
               direction_up, direction_down, direction_left, direction_right,
               apple_up, apple_down, apple_left, apple_right]
        """
        head = snake.head_pos
        
        # Points autour de la tête
        point_l = [(head[0] - 1) % GRID_SIZE, head[1]]
        point_r = [(head[0] + 1) % GRID_SIZE, head[1]]
        point_u = [head[0], (head[1] - 1) % GRID_SIZE]
        point_d = [head[0], (head[1] + 1) % GRID_SIZE]
        
        # Direction actuelle
        dir_l = snake.direction == LEFT
        dir_r = snake.direction == RIGHT
        dir_u = snake.direction == UP
        dir_d = snake.direction == DOWN
        
        # Détection des dangers
        state = [
            # Danger tout droit
            (dir_r and point_r in snake.body[1:]) or
            (dir_l and point_l in snake.body[1:]) or
            (dir_u and point_u in snake.body[1:]) or
            (dir_d and point_d in snake.body[1:]),
            
            # Danger à droite
            (dir_u and point_r in snake.body[1:]) or
            (dir_d and point_l in snake.body[1:]) or
            (dir_l and point_u in snake.body[1:]) or
            (dir_r and point_d in snake.body[1:]),
            
            # Danger à gauche
            (dir_d and point_r in snake.body[1:]) or
            (dir_u and point_l in snake.body[1:]) or
            (dir_r and point_u in snake.body[1:]) or
            (dir_l and point_d in snake.body[1:]),
            
            # Direction actuelle
            dir_l, dir_r, dir_u, dir_d,
            
            # Position de la pomme
            apple.position[0] < head[0],  # Pomme à gauche
            apple.position[0] > head[0],  # Pomme à droite
            apple.position[1] < head[1],  # Pomme en haut
            apple.position[1] > head[1]   # Pomme en bas
        ]
        
        return np.array(state, dtype=int)
    
    def get_action(self, state: np.ndarray) -> int:
        """
        Choisit une action en utilisant epsilon-greedy.
        0: Tout droit, 1: Droite, 2: Gauche
        """
        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation: utilise le modèle
        state_tensor = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            prediction = self.model(state_tensor)
        return torch.argmax(prediction).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience dans la mémoire."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Entraîne le modèle avec un batch de la mémoire."""
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Échantillonne un batch
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertit en tenseurs
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        # Q-values actuelles
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Q-values cibles
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            next_q_values[dones] = 0.0
            target_q_values = rewards + GAMMA * next_q_values
        
        # Calcul de la perte et backpropagation
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Décroissance d'epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
    
    def update_target_network(self):
        """Met à jour le réseau cible."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def action_to_direction(self, action: int, current_direction: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convertit une action en direction.
        0: Tout droit, 1: Droite, 2: Gauche, 3: Demi-tour (non utilisé)
        """
        directions = [UP, RIGHT, DOWN, LEFT]
        current_idx = directions.index(current_direction)
        
        if action == 0:  # Tout droit
            return current_direction
        elif action == 1:  # Droite
            return directions[(current_idx + 1) % 4]
        elif action == 2:  # Gauche
            return directions[(current_idx - 1) % 4]
        else:  # Demi-tour (rare)
            return directions[(current_idx + 2) % 4]


# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface):
    """Dessine la grille."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def display_info(surface, font, snake, agent, episode, best_score):
    """Affiche les informations d'entraînement."""
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    # Score actuel
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 10))
    
    # Meilleur score
    best_text = font.render(f"Best: {best_score}", True, VERT)
    surface.blit(best_text, (10, 35))

    # Épisode
    episode_text = font.render(f"Episode: {episode}", True, BLANC)
    surface.blit(episode_text, (SCREEN_WIDTH - episode_text.get_width() - 10, 10))
    
    # Epsilon
    epsilon_text = font.render(f"Epsilon: {agent.epsilon:.3f}", True, VIOLET)
    surface.blit(epsilon_text, (SCREEN_WIDTH - epsilon_text.get_width() - 10, 35))
    
    # Mémoire
    memory_text = font.render(f"Memory: {len(agent.memory)}/{MEMORY_SIZE}", True, BLANC)
    surface.blit(memory_text, (SCREEN_WIDTH - memory_text.get_width() - 10, 60))


def display_message(surface, font, message, color=BLANC, y_offset=0):
    """Affiche un message central."""
    text_surface = font.render(message, True, color)
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))
    
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)


# --- BOUCLE PRINCIPALE ---

def train_dqn(num_episodes=1000, render=True):
    """Entraîne l'agent DQN sur plusieurs épisodes."""
    if render:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake IA - Deep Q-Network Training")
        clock = pygame.time.Clock()
        font_main = pygame.font.Font(None, 25)
    
    agent = DQNAgent()
    best_score = 0
    scores = []
    
    for episode in range(num_episodes):
        # Initialisation
        snake = Snake()
        apple = Apple(snake.body)
        
        state = agent.get_state(snake, apple)
        total_reward = 0
        steps_without_food = 0
        max_steps = GRID_SIZE * GRID_SIZE * 2
        
        running = True
        game_over = False
        
        while running and not game_over:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
            
            # L'agent choisit une action
            action = agent.get_action(state)
            new_direction = agent.action_to_direction(action, snake.direction)
            snake.set_direction(new_direction)
            
            # Exécute l'action
            old_distance = abs(snake.head_pos[0] - apple.position[0]) + abs(snake.head_pos[1] - apple.position[1])
            snake.move()
            new_distance = abs(snake.head_pos[0] - apple.position[0]) + abs(snake.head_pos[1] - apple.position[1])
            
            # Calcul de la récompense
            reward = 0
            done = False
            
            # Collision avec soi-même
            if snake.is_game_over():
                reward = -10
                done = True
                game_over = True
            # Pomme attrapée
            elif snake.head_pos == list(apple.position):
                reward = 10
                snake.grow()
                steps_without_food = 0
                if not apple.relocate(snake.body):
                    reward = 100  # Victoire
                    done = True
                    game_over = True
            # Se rapproche de la pomme
            elif new_distance < old_distance:
                reward = 0.1
            # S'éloigne de la pomme
            else:
                reward = -0.1
            
            # Timeout
            steps_without_food += 1
            if steps_without_food > max_steps:
                reward = -10
                done = True
                game_over = True
            
            # Nouvel état
            next_state = agent.get_state(snake, apple)
            
            # Stocke l'expérience
            agent.remember(state, action, reward, next_state, done)
            
            # Entraîne le modèle
            agent.train()
            
            state = next_state
            total_reward += reward
            agent.steps += 1
            
            # Rendu visuel
            if render:
                screen.fill(GRIS_FOND)
                game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
                pygame.draw.rect(screen, NOIR, game_area_rect)
                
                draw_grid(screen)
                apple.draw(screen)
                snake.draw(screen)
                
                display_info(screen, font_main, snake, agent, episode + 1, best_score)
                
                pygame.display.flip()
                clock.tick(GAME_SPEED)
        
        # Fin de l'épisode
        agent.n_games += 1
        scores.append(snake.score)
        
        if snake.score > best_score:
            best_score = snake.score
        
        # Met à jour le réseau cible
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        # Affiche les statistiques
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            print(f"Episode {episode + 1}/{num_episodes} | Score: {snake.score} | Best: {best_score} | Avg: {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    if render:
        pygame.quit()
    
    print(f"\n=== Entraînement terminé ===")
    print(f"Meilleur score: {best_score}")
    print(f"Score moyen: {np.mean(scores):.2f}")


def main():
    """Fonction principale - Lance l'entraînement DQN en boucle infinie avec affichage."""
    print("=" * 60)
    print("🧠 Snake IA - Deep Q-Network (DQN)")
    print("=" * 60)
    print("🎮 Mode: Entraînement continu avec affichage visuel")
    print("🔄 L'entraînement se poursuit indéfiniment")
    print("⏸️  Fermez la fenêtre pour arrêter")
    print("=" * 60)
    print()
    
    # Lance l'entraînement en boucle infinie avec affichage
    train_dqn(num_episodes=999999, render=True)


if __name__ == '__main__':
    main()

