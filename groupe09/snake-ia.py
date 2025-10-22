import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygame
import random
import numpy as np
from collections import deque
import os
import time
import argparse

# ============================================================================
# CONFIGURATION (identique à serpent.py)
# ============================================================================

# Hyperparamètres IA
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9

# Configuration visuelle (EXACTEMENT comme serpent.py)
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 5  # Vitesse par défaut (comme serpent.py)

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (EXACTEMENT comme serpent.py)
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions (EXACTEMENT comme serpent.py)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

MODEL_PATH = 'snake_model.pth'
STATS_PATH = 'training_stats.txt'

# ============================================================================
# CLASSES DU JEU (EXACTEMENT comme serpent.py)
# ============================================================================

pygame.init()

class Snake:
    """Classe Snake EXACTEMENT comme dans serpent.py"""
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos, 
                     [self.head_pos[0] - 1, self.head_pos[1]], 
                     [self.head_pos[0] - 2, self.head_pos[1]]]
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
        x, y = self.head_pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

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
    """Classe Apple EXACTEMENT comme dans serpent.py"""
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
            pygame.draw.circle(surface, BLANC, (rect.x + int(CELL_SIZE * 0.7), rect.y + int(CELL_SIZE * 0.3)), CELL_SIZE // 8)

# Fonctions d'affichage EXACTEMENT comme dans serpent.py
def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

def display_info(surface, font, snake, start_time):
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 20))
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_text = font.render(f"Temps: {minutes:02d}:{seconds:02d}", True, BLANC)
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 20))
    max_cells = GRID_SIZE * GRID_SIZE
    fill_rate = (len(snake.body) / max_cells) * 100
    fill_text = font.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
    surface.blit(fill_text, (SCREEN_WIDTH // 2 - fill_text.get_width() // 2, 20))

# ============================================================================
# MODÈLE (MODEL)
# ============================================================================

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name=MODEL_PATH):
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name=MODEL_PATH):
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            return True
        return False

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
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
            done = (done, )
            
        pred = self.model(state)
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# ============================================================================
# AGENT
# ============================================================================

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Charger le modèle existant s'il existe
        if self.model.load():
            print("✅ Modèle chargé depuis", MODEL_PATH)
            self.load_stats()
        else:
            print("🆕 Nouveau modèle créé")
        
    def get_state(self, snake, apple):
        head = snake.head_pos
        
        # Points autour de la tête
        point_front = [head[0] + snake.direction[0], head[1] + snake.direction[1]]
        
        # Déterminer droite et gauche selon la direction actuelle
        if snake.direction == RIGHT:
            point_right = [head[0], head[1] + 1]  # Bas
            point_left = [head[0], head[1] - 1]   # Haut
        elif snake.direction == LEFT:
            point_right = [head[0], head[1] - 1]  # Haut
            point_left = [head[0], head[1] + 1]   # Bas
        elif snake.direction == UP:
            point_right = [head[0] + 1, head[1]]  # Droite
            point_left = [head[0] - 1, head[1]]   # Gauche
        else:  # DOWN
            point_right = [head[0] - 1, head[1]]  # Gauche
            point_left = [head[0] + 1, head[1]]   # Droite
        
        def is_danger(point):
            x, y = point
            # Appliquer le wrapping (comme dans le jeu original)
            x = x % GRID_SIZE
            y = y % GRID_SIZE
            wrapped_point = [x, y]
            # Danger seulement si collision avec le corps
            return wrapped_point in snake.body
        
        state = [
            # Danger (3 variables)
            is_danger(point_front),   # Danger en face
            is_danger(point_right),   # Danger à droite
            is_danger(point_left),    # Danger à gauche
            
            # Direction (4 variables)
            snake.direction == LEFT,   # Direction Gauche
            snake.direction == RIGHT,  # Direction Droite
            snake.direction == UP,     # Direction Haut
            snake.direction == DOWN,   # Direction Bas
            
            # Pomme (4 variables)
            apple.position[0] < head[0],  # Pomme à Gauche
            apple.position[0] > head[0],  # Pomme à Droite
            apple.position[1] < head[1],  # Pomme en Haut
            apple.position[1] > head[1]   # Pomme en Bas
        ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
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
    
    def action_to_direction(self, action, current_direction):
        """Convertit une action [straight, right, left] en direction"""
        clock_wise = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wise.index(current_direction)
        
        if np.array_equal(action, [1, 0, 0]):  # Straight
            return clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Right turn
            return clock_wise[(idx + 1) % 4]
        else:  # Left turn
            return clock_wise[(idx - 1) % 4]
    
    def save_stats(self, record):
        with open(STATS_PATH, 'w') as f:
            f.write(f"{self.n_games}\n")
            f.write(f"{record}\n")
    
    def load_stats(self):
        if os.path.exists(STATS_PATH):
            with open(STATS_PATH, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 1:
                    self.n_games = int(lines[0].strip())
                    print(f"📊 Parties précédentes: {self.n_games}")

# ============================================================================
# ENTRAÎNEMENT (utilise l'affichage de serpent.py)
# ============================================================================

def train(fps=GAME_SPEED):
    """Fonction d'entraînement qui utilise exactement le même affichage que serpent.py"""
    pygame.init()
    
    # Configuration de l'écran (EXACTEMENT comme serpent.py)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake AI - Deep Q-Learning")
    clock = pygame.time.Clock()
    
    # Configuration des polices (EXACTEMENT comme serpent.py)
    font_main = pygame.font.Font(None, 40)
    
    # Initialisation
    agent = Agent()
    record = 0
    
    print("\n" + "="*60)
    print("🐍 SNAKE AI - DEEP Q-LEARNING")
    print("="*60)
    print(f"Vitesse: {fps} FPS")
    print("Appuyez sur Ctrl+C pour arrêter l'entraînement\n")
    
    try:
        while True:
            # Initialisation des objets du jeu (comme serpent.py)
            snake = Snake()
            apple = Apple(snake.body)
            start_time = time.time()
            frame_iteration = 0
            
            # Boucle de jeu
            running = True
            while running:
                # Gestion des événements
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                # Obtenir l'état et l'action de l'IA
                state_old = agent.get_state(snake, apple)
                action = agent.get_action(state_old)
                
                # Convertir l'action en direction
                new_direction = agent.action_to_direction(action, snake.direction)
                snake.set_direction(new_direction)
                
                # Déplacer le serpent
                snake.move()
                frame_iteration += 1
                
                # Calculer la récompense
                reward = -0.1  # Pénalité de mouvement
                game_over = False
                
                # Vérifier collision (seulement avec soi-même, pas les murs)
                if snake.check_self_collision() or frame_iteration > 100 * len(snake.body):
                    game_over = True
                    reward = -10
                    running = False
                
                # Vérifier si pomme mangée
                elif snake.head_pos == list(apple.position):
                    snake.grow()
                    reward = 10
                    if not apple.relocate(snake.body):
                        game_over = True
                        running = False
                
                # Obtenir le nouvel état
                state_new = agent.get_state(snake, apple)
                
                # Entraîner
                agent.train_short_memory(state_old, action, reward, state_new, game_over)
                agent.remember(state_old, action, reward, state_new, game_over)
                
                # Dessin (EXACTEMENT comme serpent.py)
                screen.fill(GRIS_FOND)
                game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
                pygame.draw.rect(screen, NOIR, game_area_rect)
                draw_grid(screen)
                apple.draw(screen)
                snake.draw(screen)
                display_info(screen, font_main, snake, start_time)
                
                pygame.display.flip()
                clock.tick(fps)
            
            # Fin de partie
            agent.n_games += 1
            agent.train_long_memory()
            
            if snake.score > record:
                record = snake.score
                agent.model.save()
                print(f"🏆 NOUVEAU RECORD: {record}")
            
            print(f'Partie {agent.n_games:4d} | Score: {snake.score:3d} | Record: {record:3d}')
            
            # Sauvegarder les stats tous les 10 parties
            if agent.n_games % 10 == 0:
                agent.save_stats(record)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Entraînement interrompu")
        agent.model.save()
        agent.save_stats(record)
        print(f"💾 Modèle sauvegardé ({agent.n_games} parties)")
        print(f"📊 Record: {record}")
    
    pygame.quit()

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snake AI - Deep Q-Learning')
    parser.add_argument('-fps', '--fps', type=int, default=GAME_SPEED,
                        help=f'Vitesse du jeu en FPS (défaut: {GAME_SPEED}, rapide: 1000)')
    args = parser.parse_args()
    
    train(fps=args.fps)
