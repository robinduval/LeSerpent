import pygame
import random
import numpy as np
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add parent directory to path to import serpent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CONSTANTES DE JEU (similaires à serpent.py) ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 800  # Vitesse plus rapide pour l'entraînement

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
BLEU = (0, 100, 255)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# --- HYPERPARAMÈTRES DE L'IA ---
LEARNING_RATE = 0.001  # Learning rate optimal
GAMMA = 0.95  # Facteur de discount pour équilibrer court/long terme
EPSILON_START = 1.0  # Exploration initiale
EPSILON_MIN = 0.01  # Exploration minimale
EPSILON_DECAY = 0.997  # Décroissance modérée
MEMORY_SIZE = 100000  # Grande mémoire pour diversité
BATCH_SIZE = 64  # Batch size équilibré
TARGET_UPDATE = 20  # Update plus espacé pour stabilité


# --- RÉSEAU DE NEURONES (Deep Q-Network) ---
class DQN(nn.Module):
    """
    Réseau de neurones profond pour approximer la fonction Q.
    Input: État du jeu (11 features)
    Output: Q-values pour chaque action (4 actions possibles)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# --- MÉMOIRE DE REPLAY ---
class ReplayMemory:
    """
    Stocke les expériences (state, action, reward, next_state, done)
    pour l'apprentissage par batch.
    """
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# --- AGENT D'APPRENTISSAGE PAR RENFORCEMENT ---
class SnakeAgent:
    """
    Agent RL utilisant Deep Q-Learning pour apprendre à jouer au Snake.
    
    Architecture selon le sujet:
    - Agent:
        * game: environnement PyGame
        * model: réseau de neurones (Torch)
        * Training: boucle d'apprentissage
            - state = get_state(game)
            - action = get_move(state): model.predict()
            - reward, game_over, score = game.play_step(action)
            - new_state = get_state(game)
            - remember (mémoire de replay)
            - model.train()
    
    - Game (PyGame):
        * play_step(action) -> reward, game_over, score
    
    - Model (Torch):
        * Linear_QNet (DQN)
        * model.predict(state) -> action
    
    États (11 features):
    - Danger: en face, à droite, à gauche
    - Direction: Gauche, Droite, Haut, Bas
    - Pomme: Gauche, Droite, Haut, Bas
    
    Actions (4 actions absolues):
    - HAUT: (0, -1)
    - BAS: (0, 1)
    - GAUCHE: (-1, 0)
    - DROITE: (1, 0)
    
    Récompenses:
    - Attraper la pomme: +10
    - Perdre: -10
    - Autres actions: 0
    - Se déplacer: -0.1
    - Finir le jeu (victoire): +100
    """
    def __init__(self):
        # État: 11 features
        # [danger_straight, danger_right, danger_left, dir_up, dir_down, dir_left, dir_right, food_up, food_down, food_left, food_right]
        self.state_size = 11
        self.action_size = 4  # [HAUT, BAS, GAUCHE, DROITE] - Actions absolues
        
        # Réseaux de neurones
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, 256, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, 256, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimiseur et fonction de perte
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        # Mémoire de replay
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Paramètres d'exploration
        self.epsilon = EPSILON_START
        
        # Statistiques
        self.n_games = 0
        self.total_score = 0
        self.record_score = 0
        self.scores_history = deque(maxlen=100)  # Garde les 100 derniers scores
        
    def get_state(self, game):
        """
        Extrait les features de l'état actuel du jeu.
        Returns: np.array de 11 features
        """
        head = game.snake.head_pos
        
        # Points autour de la tête (normalisés avec le modulo pour wrap)
        point_l = [(head[0] - 1) % GRID_SIZE, head[1]]
        point_r = [(head[0] + 1) % GRID_SIZE, head[1]]
        point_u = [head[0], (head[1] - 1) % GRID_SIZE]
        point_d = [head[0], (head[1] + 1) % GRID_SIZE]
        
        # Direction actuelle
        dir_l = game.snake.direction == LEFT
        dir_r = game.snake.direction == RIGHT
        dir_u = game.snake.direction == UP
        dir_d = game.snake.direction == DOWN
        
        # Position de la nourriture (normalisée)
        food_left = game.apple.position[0] < head[0]
        food_right = game.apple.position[0] > head[0]
        food_up = game.apple.position[1] < head[1]
        food_down = game.apple.position[1] > head[1]
        
        # État (11 features)
        state = [
            # Danger tout droit (collision avec le corps)
            (dir_r and point_r in game.snake.body[1:]) or
            (dir_l and point_l in game.snake.body[1:]) or
            (dir_u and point_u in game.snake.body[1:]) or
            (dir_d and point_d in game.snake.body[1:]),
            
            # Danger à droite (collision avec le corps)
            (dir_u and point_r in game.snake.body[1:]) or
            (dir_d and point_l in game.snake.body[1:]) or
            (dir_l and point_u in game.snake.body[1:]) or
            (dir_r and point_d in game.snake.body[1:]),
            
            # Danger à gauche (collision avec le corps)
            (dir_d and point_r in game.snake.body[1:]) or
            (dir_u and point_l in game.snake.body[1:]) or
            (dir_r and point_u in game.snake.body[1:]) or
            (dir_l and point_d in game.snake.body[1:]),
            
            # Direction actuelle
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Position de la nourriture
            food_left,
            food_right,
            food_up,
            food_down,
        ]
        
        return np.array(state, dtype=int)
    
    def get_action(self, state):
        """
        Choisit une action selon epsilon-greedy.
        Returns: action (0=HAUT, 1=BAS, 2=GAUCHE, 3=DROITE)
        """
        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation: utilise le réseau
        self.policy_net.eval()  # Mode eval pour BatchNorm
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        self.policy_net.train()  # Retour en mode train
        return q_values.argmax().item()
    
    def train_step(self):
        """
        Entraîne le réseau sur un batch d'expériences.
        """
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Échantillonne un batch
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertit en tenseurs
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q-values actuelles
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: utilise policy_net pour sélectionner l'action, target_net pour évaluer
        with torch.no_grad():
            # Sélectionne la meilleure action avec policy_net
            next_actions = self.policy_net(next_states).argmax(1)
            # Évalue cette action avec target_net
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        # Calcul de la perte et optimisation (Huber Loss pour plus de robustesse)
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Décroissance de epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
    
    def update_target_network(self):
        """Met à jour le réseau cible."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filename="snake_dqn_model.pth"):
        """Sauvegarde le modèle."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'n_games': self.n_games,
        }, filename)
        print(f"Modèle sauvegardé: {filename}")
    
    def load_model(self, filename="snake_dqn_model.pth"):
        """Charge le modèle."""
        if os.path.exists(filename):
            try:
                checkpoint = torch.load(filename)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                self.n_games = checkpoint['n_games']
                print(f"Modèle chargé: {filename}")
                return True
            except (RuntimeError, KeyError) as e:
                print(f"⚠️  Impossible de charger le modèle (architecture incompatible)")
                print(f"    Démarrage avec un nouveau modèle...")
                return False
        return False


# --- CLASSES DU JEU (adaptées pour l'IA) ---
class Snake:
    """Serpent pour le jeu IA."""
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos[:], 
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

    def check_self_collision(self):
        return self.head_pos in self.body[1:]

    def draw(self, surface):
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)

        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)


class Apple:
    """Pomme pour le jeu IA."""
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


# --- ENVIRONNEMENT DE JEU ---
class SnakeGame:
    """Environnement de jeu pour l'entraînement de l'IA."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Réinitialise le jeu."""
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.frame_iteration = 0
        return self.snake
    
    def is_collision(self, point=None):
        """Vérifie si un point est en collision."""
        if point is None:
            point = self.snake.head_pos
        
        # Collision avec soi-même
        if point in self.snake.body[1:]:
            return True
        
        # Pas de collision avec les murs car on utilise le modulo (wrap around)
        return False
    
    def play_step(self, action):
        """
        Exécute une action et retourne (reward, game_over, score).
        action: 0=HAUT (0, -1), 1=BAS (0, 1), 2=GAUCHE (-1, 0), 3=DROITE (1, 0)
        """
        self.frame_iteration += 1
        
        # Convertit l'action en direction absolue
        action_map = [UP, DOWN, LEFT, RIGHT]
        new_dir = action_map[action]
        
        self.snake.set_direction(new_dir)
        
        # Déplace le serpent
        self.snake.move()
        
        # Initialise la récompense
        reward = -0.1  # Se déplacer: -0.1
        game_over = False
        
        # Vérifie les collisions
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake.body):
            game_over = True
            reward = -10  # Perdre: -10
            return reward, game_over, self.snake.score
        
        # Vérifie si la pomme est mangée
        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            reward = 10  # Attraper la pomme: +10
            self.frame_iteration = 0  # Reset le compteur pour éviter timeout
            if not self.apple.relocate(self.snake.body):
                # Victoire totale! Finir le jeu: 100
                game_over = True
                reward = 100
        
        return reward, game_over, self.snake.score


# --- FONCTIONS D'AFFICHAGE ---
def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def display_info(surface, font, game, agent, best_score, avg_score):
    """Affiche les informations d'entraînement."""
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    # Score actuel
    score_text = font.render(f"Score: {game.snake.score}", True, BLANC)
    surface.blit(score_text, (10, 10))

    # Nombre de parties
    games_text = font.render(f"Parties: {agent.n_games}", True, BLANC)
    surface.blit(games_text, (10, 35))
    
    # Score moyen
    avg_text = font.render(f"Moy: {avg_score:.1f}", True, BLANC)
    surface.blit(avg_text, (10, 55))

    # Meilleur score
    best_text = font.render(f"Record: {best_score}", True, VERT)
    surface.blit(best_text, (SCREEN_WIDTH // 2 - 80, 10))

    # Epsilon (exploration)
    epsilon_text = font.render(f"Epsilon: {agent.epsilon:.3f}", True, BLEU)
    surface.blit(epsilon_text, (SCREEN_WIDTH // 2 - 80, 35))

    # Mémoire
    memory_text = font.render(f"Mémoire: {len(agent.memory)}", True, BLANC)
    surface.blit(memory_text, (SCREEN_WIDTH - 180, 10))


# --- BOUCLE PRINCIPALE D'ENTRAÎNEMENT ---
def train():
    """Fonction d'entraînement de l'agent RL."""
    pygame.init()
    
    # Configuration de l'écran
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake IA - Deep Q-Learning")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)
    
    # Initialisation
    agent = SnakeAgent()
    game = SnakeGame()
    
    # Tente de charger un modèle existant automatiquement
    model_path = os.path.join(os.path.dirname(__file__), "snake_dqn_model.pth")
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print("✅ Modèle existant chargé - Reprise de l'entraînement")
    else:
        print("🆕 Nouveau modèle créé - Début de l'entraînement")
    
    best_score = 0
    avg_score = 0
    running = True
    
    print("\n🐍 ENTRAÎNEMENT DE L'AGENT SNAKE IA 🐍")
    print("=" * 50)
    print("Contrôles:")
    print("  - ESC: Quitter")
    print("  - S: Sauvegarder le modèle")
    print("=" * 50)
    
    while running:
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    agent.save_model(model_path)
        
        # Obtient l'état actuel
        state_old = agent.get_state(game)
        
        # Choisit une action
        action = agent.get_action(state_old)
        
        # Exécute l'action
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)
        
        # Enregistre l'expérience
        agent.memory.push(state_old, action, reward, state_new, done)
        
        # Entraîne le réseau
        agent.train_step()
        
        # Affichage
        screen.fill(NOIR)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        draw_grid(screen)
        game.apple.draw(screen)
        game.snake.draw(screen)
        display_info(screen, font, game, agent, best_score, avg_score)
        pygame.display.flip()
        clock.tick(GAME_SPEED)
        
        # Si la partie est terminée
        if done:
            agent.n_games += 1
            agent.scores_history.append(score)
            game.reset()
            
            # Calcul du score moyen
            if len(agent.scores_history) > 0:
                avg_score = sum(agent.scores_history) / len(agent.scores_history)
            
            # Met à jour le meilleur score
            if score > best_score:
                best_score = score
                agent.save_model(model_path)
            
            # Met à jour le réseau cible
            if agent.n_games % TARGET_UPDATE == 0:
                agent.update_target_network()
            
            # Affiche les statistiques
            print(f"Partie {agent.n_games} | Score: {score:2d} | Record: {best_score:2d} | Moyenne: {avg_score:5.2f} | Epsilon: {agent.epsilon:.3f}")
    
    # Sauvegarde finale
    agent.save_model(model_path)
    pygame.quit()
    print("\n✅ Entraînement terminé!")
    print(f"📊 Total de parties: {agent.n_games}")
    print(f"🏆 Meilleur score: {best_score}")


# --- MODE DÉMONSTRATION ---
def demo():
    """Mode démonstration avec un agent entraîné."""
    pygame.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake IA - Démonstration")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)
    
    agent = SnakeAgent()
    model_path = os.path.join(os.path.dirname(__file__), "snake_dqn_model.pth")
    
    if not agent.load_model(model_path):
        print("❌ Aucun modèle trouvé. Entraînez d'abord l'agent avec mode 'train'.")
        return
    
    agent.epsilon = 0  # Pas d'exploration en mode démo
    game = SnakeGame()
    running = True
    
    print("\n🎮 MODE DÉMONSTRATION")
    print("L'agent utilise son modèle entraîné.")
    print("Appuyez sur ESC pour quitter.")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)
        
        screen.fill(NOIR)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        draw_grid(screen)
        game.apple.draw(screen)
        game.snake.draw(screen)
        display_info(screen, font, game, agent, score)
        pygame.display.flip()
        clock.tick(15)  # Vitesse de jeu normale pour la démo
        
        if done:
            print(f"Partie terminée! Score: {score}")
            time.sleep(2)
            game.reset()
    
    pygame.quit()


# --- POINT D'ENTRÉE ---
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🐍 SNAKE IA - DEEP Q-LEARNING 🧠")
    print("="*50)
    train()
