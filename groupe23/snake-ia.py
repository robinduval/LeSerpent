import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os
import time

# --- CONSTANTES DE JEU ---
GRID_SIZE = 15
CELL_SIZE = 30

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)
BLEU = (0, 150, 255)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# --- RÉSEAU DE NEURONES (Deep Q-Network) ---
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# --- ENVIRONNEMENT ---
class SnakeEnvironment:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.steps_without_food = 0
        self.max_steps = GRID_SIZE * GRID_SIZE * 2
        self.previous_distance = self._distance_to_food()
        return self.get_state()
    
    def get_state(self):
        head = self.snake.head_pos
        direction = self.snake.direction
        
        point_l = [head[0] - 1, head[1]]
        point_r = [head[0] + 1, head[1]]
        point_u = [head[0], head[1] - 1]
        point_d = [head[0], head[1] + 1]
        
        dir_l = direction == LEFT
        dir_r = direction == RIGHT
        dir_u = direction == UP
        dir_d = direction == DOWN
        
        if dir_u:
            danger_straight = self._is_collision(point_u)
            danger_left = self._is_collision(point_l)
            danger_right = self._is_collision(point_r)
        elif dir_d:
            danger_straight = self._is_collision(point_d)
            danger_left = self._is_collision(point_r)
            danger_right = self._is_collision(point_l)
        elif dir_l:
            danger_straight = self._is_collision(point_l)
            danger_left = self._is_collision(point_d)
            danger_right = self._is_collision(point_u)
        else:
            danger_straight = self._is_collision(point_r)
            danger_left = self._is_collision(point_u)
            danger_right = self._is_collision(point_d)
        
        food_left = self.apple.position[0] < head[0]
        food_right = self.apple.position[0] > head[0]
        food_up = self.apple.position[1] < head[1]
        food_down = self.apple.position[1] > head[1]
        
        distance = self._distance_to_food() / (GRID_SIZE * 2)
        snake_length = len(self.snake.body) / (GRID_SIZE * GRID_SIZE)
        
        state = [
            int(danger_straight), int(danger_left), int(danger_right),
            int(dir_l), int(dir_r), int(dir_u), int(dir_d),
            int(food_left), int(food_right), int(food_up), int(food_down),
            distance, snake_length
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _is_collision(self, point):
        if point[0] < 0 or point[0] >= GRID_SIZE or point[1] < 0 or point[1] >= GRID_SIZE:
            return True
        if point in self.snake.body[1:]:
            return True
        return False
    
    def step(self, action):
        clock_wise = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wise.index(self.snake.direction)
        
        if action == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2:
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = self.snake.direction
            
        self.snake.set_direction(new_dir)
        
        old_distance = self.previous_distance
        self.snake.move()
        self.steps_without_food += 1
        
        new_distance = self._distance_to_food()
        self.previous_distance = new_distance
        
        reward = 0
        game_over = False
        
        if self.snake.is_game_over():
            reward = -100
            game_over = True
            return self.get_state(), reward, game_over
        
        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            reward = 100
            self.steps_without_food = 0
            if not self.apple.relocate(self.snake.body):
                reward = 500
                game_over = True
        else:
            if new_distance < old_distance:
                reward = 10
            else:
                reward = -15
        
        if self.steps_without_food > self.max_steps:
            reward = -100
            game_over = True
        elif self.steps_without_food > self.max_steps * 0.75:
            reward -= 5
        
        reward += 1
            
        return self.get_state(), reward, game_over
    
    def _distance_to_food(self):
        return abs(self.snake.head_pos[0] - self.apple.position[0]) + \
               abs(self.snake.head_pos[1] - self.apple.position[1])

# --- AGENT D'APPRENTISSAGE ---
class DQNAgent:
    def __init__(self):
        self.n_states = 13
        self.n_actions = 3
        self.memory = deque(maxlen=100000)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_frequency = 10
        
        self.model = DQN(self.n_states, 256, self.n_actions)
        self.target_model = DQN(self.n_states, 256, self.n_actions)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.training_step = 0
        
        # Charger le modèle s'il existe
        if os.path.exists('snake_learning.pth'):
            self.load('snake_learning.pth')
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        minibatch = random.sample(self.memory, self.batch_size)
        total_loss = 0
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            if done:
                target = reward
            else:
                next_action = torch.argmax(self.model(next_state_tensor)).item()
                target = reward + self.gamma * self.target_model(next_state_tensor)[0][next_action].item()
            
            current_q = self.model(state_tensor)
            target_q = current_q.clone()
            target_q[0][action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(current_q, target_q)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.training_step += 1
        if self.training_step % self.update_target_frequency == 0:
            self.update_target_model()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return total_loss / self.batch_size
    
    def save(self, filename='snake_learning.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filename)
    
    def load(self, filename='snake_learning.pth'):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']

# --- CLASSES DU JEU ---
class Snake:
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
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)

# --- AFFICHAGE ---
def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

def display_info(surface, font, snake, episode, epsilon, avg_score, max_score, elapsed_time):
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 8))

    record_text = font.render(f"Record: {max_score}", True, ORANGE)
    surface.blit(record_text, (10, 32))
    
    episode_text = font.render(f"Partie: {episode}", True, BLANC)
    surface.blit(episode_text, (10, 56))
    
    avg_text = font.render(f"Moy: {avg_score:.1f}", True, BLEU)
    surface.blit(avg_text, (SCREEN_WIDTH // 2 - avg_text.get_width() // 2, 20))
    
    # NOUVEAU: Affichage du timer
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    time_text = font.render(f"⏱️ {hours:02d}:{minutes:02d}:{seconds:02d}", True, (255, 255, 0))
    surface.blit(time_text, (SCREEN_WIDTH // 2 - time_text.get_width() // 2, 45))
    
    epsilon_text = font.render(f"Exploration: {epsilon:.2f}", True, VERT)
    surface.blit(epsilon_text, (SCREEN_WIDTH - epsilon_text.get_width() - 10, 15))
    
    learning_text = font.render("🧠 APPRENTISSAGE", True, ORANGE)
    surface.blit(learning_text, (SCREEN_WIDTH - learning_text.get_width() - 10, 45))

# --- PROGRAMME PRINCIPAL : JEU + APPRENTISSAGE EN CONTINU ---
def main():
    pygame.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("🐍 Snake IA - Apprentissage en Continu (Deep Q-Learning)")
    clock = pygame.time.Clock()
    
    font_main = pygame.font.Font(None, 24)
    
    env = SnakeEnvironment()
    agent = DQNAgent()
    
    # Statistiques
    episode = 0
    scores = []
    max_score = 0
    start_time = time.time()
    
    print("\n" + "="*70)
    print("🚀 SNAKE IA - APPRENTISSAGE PAR RENFORCEMENT EN CONTINU")
    print("="*70)
    print("🧠 L'IA apprend en jouant, elle s'améliore à chaque partie!")
    print("📊 Les statistiques s'affichent en temps réel")
    print("💾 Le modèle est sauvegardé automatiquement tous les 10 épisodes")
    print("\n🎮 CONTRÔLES:")
    print("   ESPACE: Pause/Reprendre")
    print("   S: Sauvegarder manuellement")
    print("   Q: Quitter")
    print("="*70 + "\n")
    
    state = env.reset()
    running = True
    pause = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause = not pause
                    print(f"{'⏸️  PAUSE' if pause else '▶️  REPRISE'}")
                elif event.key == pygame.K_s:
                    agent.save()
                    print(f"💾 Modèle sauvegardé manuellement!")
                elif event.key == pygame.K_q:
                    running = False
        
        if not pause:
            # L'IA choisit une action
            action = agent.act(state)
            
            # Exécute l'action
            next_state, reward, done = env.step(action)
            
            # Mémorise l'expérience
            agent.remember(state, action, reward, next_state, done)
            
            # APPRENTISSAGE EN TEMPS RÉEL
            agent.replay()
            
            state = next_state
            
            # Affichage
            screen.fill(GRIS_FOND)
            game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
            pygame.draw.rect(screen, NOIR, game_area_rect)
            
            draw_grid(screen)
            env.apple.draw(screen)
            env.snake.draw(screen)
            
            avg_score = np.mean(scores[-100:]) if scores else 0
            elapsed_time = time.time() - start_time
            display_info(screen, font_main, env.snake, episode, agent.epsilon, avg_score, max_score, elapsed_time)
            
            pygame.display.flip()
            clock.tick(50)  # Vitesse rapide pour apprentissage efficace
            
            if done:
                score = env.snake.score
                scores.append(score)
                episode += 1
                
                if score > max_score:
                    max_score = score
                    print(f"🏆 NOUVEAU RECORD! Score: {score} (Partie {episode})")
                
                # Affichage console
                avg = np.mean(scores[-100:]) if len(scores) >= 10 else np.mean(scores)
                print(f"Partie {episode:4d} | Score: {score:3d} | "
                      f"Record: {max_score:3d} | Moy: {avg:5.1f} | "
                      f"ε: {agent.epsilon:.3f}")
                
                # Sauvegarde automatique tous les 10 épisodes
                if episode % 10 == 0:
                    agent.save()
                    print(f"💾 Auto-sauvegarde (partie {episode})")
                
                state = env.reset()
        else:
            # Mode pause
            screen.fill(GRIS_FOND)
            game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
            pygame.draw.rect(screen, NOIR, game_area_rect)
            
            draw_grid(screen)
            env.apple.draw(screen)
            env.snake.draw(screen)
            
            avg_score = np.mean(scores[-100:]) if scores else 0
            elapsed_time = time.time() - start_time
            display_info(screen, font_main, env.snake, episode, agent.epsilon, avg_score, max_score, elapsed_time)
            
            # Message pause
            pause_font = pygame.font.Font(None, 60)
            pause_text = pause_font.render("⏸️ PAUSE", True, BLANC)
            pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            pygame.draw.rect(screen, GRIS_FOND, pause_rect.inflate(40, 40))
            screen.blit(pause_text, pause_rect)
            
            pygame.display.flip()
            clock.tick(10)
    
    # Sauvegarde finale
    agent.save()
    pygame.quit()
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("📊 STATISTIQUES FINALES")
    print("="*70)
    print(f"🏆 Score maximum: {max_score}")
    print(f"📈 Score moyen: {np.mean(scores):.2f}")
    print(f"🎮 Nombre de parties: {episode}")
    print(f"⏱️  Durée totale: {int(total_time//3600):02d}h{int((total_time%3600)//60):02d}m{int(total_time%60):02d}s")
    print(f"💾 Modèle sauvegardé dans: snake_learning.pth")
    print("="*70 + "\n")
    print("👋 À bientôt! Relancez pour continuer l'apprentissage.\n")

if __name__ == '__main__':
    main()