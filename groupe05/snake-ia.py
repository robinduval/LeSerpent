"""
Snake IA - Apprentissage par Renforcement (Deep Q-Learning)
============================================================
Ce fichier implémente un agent IA qui apprend à jouer au Snake
en utilisant le Deep Q-Learning (DQN) avec PyTorch.

Il utilise le jeu serpent.py de la racine du projet comme environnement.

Structure:
- SnakeGameAI: Wrapper adapté du jeu serpent.py pour l'IA
- Model: Réseau neuronal DQN avec PyTorch  
- Agent: Orchestrateur qui gère l'apprentissage
"""

import sys
import os
# Ajouter le répertoire parent au path pour importer serpent.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import random
import numpy as np
from collections import deque
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import display

# Importer les classes et constantes du jeu original
from serpent import Snake, Apple, GRID_SIZE, CELL_SIZE, SCORE_PANEL_HEIGHT
from serpent import SCREEN_WIDTH, SCREEN_HEIGHT, UP, DOWN, LEFT, RIGHT
from serpent import BLANC, NOIR, ORANGE, VERT, ROUGE, GRIS_FOND, GRIS_GRILLE
from serpent import draw_grid

# Récompenses pour l'IA
REWARD_FOOD = 10
REWARD_DEATH = -10
REWARD_MOVE = -0.1
REWARD_WIN = 100

# Vitesse d'entraînement (plus rapide que le jeu normal)
GAME_SPEED_AI = 40

# ============================================================================
# CLASSE GAME - WRAPPER POUR L'IA
# ============================================================================

class SnakeGameAI:
    """
    Wrapper du jeu Snake original pour l'apprentissage par renforcement.
    Adapte le jeu serpent.py pour retourner les récompenses nécessaires à l'IA.
    """
    
    def __init__(self, headless=False):
        """
        Args:
            headless: Si True, pas d'interface graphique (entraînement plus rapide)
        """
        self.headless = headless
        
        # Initialisation de PyGame
        if not headless:
            pygame.init()
            self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Snake IA - Deep Q-Learning')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 30)
        else:
            # Mode headless : initialiser quand même pygame pour les events
            pygame.init()
            self.clock = pygame.time.Clock()
        
        # Timer global (depuis le lancement du jeu)
        self.global_start_time = time.time()
        
        # Réinitialisation du jeu
        self.reset()
    
    def reset(self):
        """Réinitialise le jeu pour une nouvelle partie."""
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.frame_iteration = 0
        self.score = 0
        self.start_time = time.time()  # Démarrer le chronomètre
        self.record = 0  # Meilleur score
    
    def set_record(self, record):
        """Met à jour le record à afficher."""
        self.record = record
    
    def play_step(self, action):
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: Liste de 3 éléments [tout_droit, tourner_droite, tourner_gauche]
        
        Returns:
            reward: Récompense obtenue
            game_over: Boolean indiquant si la partie est terminée
            score: Score actuel
        """
        self.frame_iteration += 1
        
        # 1. Gérer les événements PyGame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Convertir l'action en direction et l'appliquer
        self._set_direction_from_action(action)
        
        # 3. Déplacer le serpent
        self.snake.move()
        
        # 4. Vérifier les collisions (Game Over)
        reward = REWARD_MOVE  # Pénalité légère pour encourager l'efficacité
        game_over = False
        
        if self.snake.is_game_over():
            game_over = True
            reward = REWARD_DEATH
            return reward, game_over, self.score
        
        # Timeout (si le serpent tourne en rond trop longtemps)
        if self.frame_iteration > 100 * len(self.snake.body):
            game_over = True
            reward = REWARD_DEATH
            return reward, game_over, self.score
        
        # 5. Vérifier si le serpent a mangé la pomme
        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            self.score = self.snake.score
            reward = REWARD_FOOD
            
            # Tenter de replacer la pomme
            if not self.apple.relocate(self.snake.body):
                # Victoire : toute la grille est remplie
                reward = REWARD_WIN
                game_over = True
                return reward, game_over, self.score
        
        # 6. Mettre à jour l'interface (seulement si pas en mode headless)
        if not self.headless:
            self._update_ui()
            self.clock.tick(GAME_SPEED_AI)
        
        return reward, game_over, self.score
    
    def _set_direction_from_action(self, action):
        """
        Convertit une action [tout_droit, droite, gauche] en direction.
        """
        # Ordre horaire des directions
        clock_wise = [RIGHT, DOWN, LEFT, UP]
        current_idx = clock_wise.index(self.snake.direction)
        
        if np.array_equal(action, [1, 0, 0]):  # Tout droit
            new_direction = clock_wise[current_idx]
        elif np.array_equal(action, [0, 1, 0]):  # Tourner à droite
            new_direction = clock_wise[(current_idx + 1) % 4]
        else:  # [0, 0, 1] - Tourner à gauche
            new_direction = clock_wise[(current_idx - 1) % 4]
        
        self.snake.set_direction(new_direction)
    
    def _update_ui(self):
        """Affiche le jeu avec PyGame (utilise les fonctions du jeu original)."""
        # Fond
        self.display.fill(GRIS_FOND)
        
        # Zone de jeu
        game_area = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.display, NOIR, game_area)
        
        # Grille
        draw_grid(self.display)
        
        # Dessiner la pomme et le serpent (méthodes originales)
        self.apple.draw(self.display)
        self.snake.draw(self.display)
        
        # Panneau supérieur
        pygame.draw.line(self.display, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)
        
        # Colonne de gauche
        # Afficher le score
        score_text = self.font.render(f"Score: {self.score}", True, BLANC)
        self.display.blit(score_text, (10, 10))
        
        # Afficher le record (meilleur score)
        record_text = self.font.render(f"Record: {self.record}", True, ORANGE)
        self.display.blit(record_text, (10, 35))
        
        # Colonne du centre
        # Afficher le timer de la partie en cours
        elapsed_game = time.time() - self.start_time
        game_minutes = int(elapsed_game // 60)
        game_seconds = int(elapsed_game % 60)
        game_time_text = self.font.render(f"Partie: {game_minutes:02d}:{game_seconds:02d}", True, BLANC)
        center_x = SCREEN_WIDTH // 2 - game_time_text.get_width() // 2
        self.display.blit(game_time_text, (center_x, 10))
        
        # Colonne de droite
        # Afficher le timer global
        elapsed_global = time.time() - self.global_start_time
        global_minutes = int(elapsed_global // 60)
        global_seconds = int(elapsed_global % 60)
        global_time_text = self.font.render(f"Total: {global_minutes:02d}:{global_seconds:02d}", True, BLANC)
        self.display.blit(global_time_text, (SCREEN_WIDTH - global_time_text.get_width() - 10, 10))
        
        pygame.display.flip()

# ============================================================================
# CLASSE MODEL - RÉSEAU NEURONAL DQN
# ============================================================================

class Linear_QNet(nn.Module):
    """
    Réseau neuronal pour le Deep Q-Learning.
    Architecture: input_size -> hidden_size -> output_size
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """Propagation avant dans le réseau."""
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        """Sauvegarde le modèle."""
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name='model.pth'):
        """Charge le modèle."""
        self.load_state_dict(torch.load(file_name))


class QTrainer:
    """
    Entraîneur pour le réseau Q.
    Gère l'optimisation et la fonction de perte.
    """
    
    def __init__(self, model, lr, gamma):
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        """
        Entraîne le modèle sur une seule transition ou un batch.
        
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: Nouvel état
            done: Boolean indiquant si l'épisode est terminé
        """
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        
        # Si ce n'est pas un batch, ajouter une dimension
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        # 1. Prédire les Q-values avec l'état actuel
        pred = self.model(state)
        
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Formule de Bellman: Q(s,a) = r + gamma * max(Q(s',a'))
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        # 2. Calculer la perte et backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# ============================================================================
# CLASSE AGENT - ORCHESTRATEUR
# ============================================================================

class Agent:
    """
    Agent d'apprentissage par renforcement.
    Gère la mémoire, l'exploration/exploitation et l'entraînement.
    """
    
    def __init__(self, load_model=True):
        self.n_games = 0
        self.epsilon = 0  # Randomness (exploration)
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=100_000)  # Replay buffer
        self.model_loaded = False  # Flag pour savoir si un modèle a été chargé
        
        # Modèle (11 inputs -> 256 hidden -> 3 outputs)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
        
        # Charger le modèle existant si disponible
        if load_model and os.path.exists('best_model.pth'):
            print("📚 Chargement du modèle existant 'best_model.pth'...")
            self.model.load('best_model.pth')
            self.model_loaded = True
            print("✅ Modèle chargé avec succès !")
        elif load_model:
            print("ℹ️  Aucun modèle existant trouvé. Démarrage d'un nouvel entraînement.")
    
    def get_state(self, game):
        """
        Extrait l'état du jeu sous forme de vecteur de features.
        
        État (11 valeurs):
        - Danger devant (1 bit)
        - Danger à droite (1 bit)
        - Danger à gauche (1 bit)
        - Direction actuelle (4 bits: gauche, droite, haut, bas)
        - Position de la nourriture (4 bits: gauche, droite, haut, bas)
        """
        head = game.snake.head_pos
        
        # Points autour de la tête
        point_l = [head[0] - 1, head[1]]
        point_r = [head[0] + 1, head[1]]
        point_u = [head[0], head[1] - 1]
        point_d = [head[0], head[1] + 1]
        
        # Direction actuelle (utilise les constantes de serpent.py)
        dir_l = game.snake.direction == LEFT
        dir_r = game.snake.direction == RIGHT
        dir_u = game.snake.direction == UP
        dir_d = game.snake.direction == DOWN
        
        # Fonction pour vérifier la collision d'un point
        def is_collision(pt):
            # Collision avec les murs
            if pt[0] < 0 or pt[0] >= GRID_SIZE or pt[1] < 0 or pt[1] >= GRID_SIZE:
                return True
            # Collision avec le corps du serpent
            if pt in game.snake.body:
                return True
            return False
        
        state = [
            # Danger tout droit
            (dir_r and is_collision(point_r)) or
            (dir_l and is_collision(point_l)) or
            (dir_u and is_collision(point_u)) or
            (dir_d and is_collision(point_d)),
            
            # Danger à droite
            (dir_u and is_collision(point_r)) or
            (dir_d and is_collision(point_l)) or
            (dir_l and is_collision(point_u)) or
            (dir_r and is_collision(point_d)),
            
            # Danger à gauche
            (dir_d and is_collision(point_r)) or
            (dir_u and is_collision(point_l)) or
            (dir_r and is_collision(point_u)) or
            (dir_l and is_collision(point_d)),
            
            # Direction actuelle
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Position de la nourriture
            game.apple.position[0] < head[0],  # Nourriture à gauche
            game.apple.position[0] > head[0],  # Nourriture à droite
            game.apple.position[1] < head[1],  # Nourriture en haut
            game.apple.position[1] > head[1]   # Nourriture en bas
        ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        """Stocke une transition dans la mémoire (replay buffer)."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        """Entraîne sur un batch aléatoire de la mémoire (experience replay)."""
        if len(self.memory) > 1000:
            # Échantillon aléatoire de 1000 transitions
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        """Entraîne sur une seule transition (apprentissage immédiat)."""
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        """
        Détermine l'action à prendre (politique epsilon-greedy).
        
        Exploration vs Exploitation:
        - Au début (epsilon élevé): actions aléatoires pour explorer
        - Plus tard (epsilon faible): actions basées sur le modèle
        - Si modèle chargé: epsilon plus faible pour exploiter l'apprentissage
        """
        # Epsilon-greedy: décroissance de l'exploration
        if self.model_loaded:
            # Modèle pré-entraîné : exploration réduite (commence à 20 au lieu de 80)
            self.epsilon = max(0, 20 - self.n_games)
        else:
            # Nouveau modèle : exploration normale
            self.epsilon = 80 - self.n_games
        
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            # Exploration: mouvement aléatoire
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation: utiliser le modèle
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
    
    def get_action_greedy(self, state):
        """
        Détermine l'action en mode purement greedy (sans exploration).
        Utilisé pour les tests de performance où on veut la meilleure action possible.
        """
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        
        return final_move

# ============================================================================
# FONCTION D'AFFICHAGE DES STATISTIQUES
# ============================================================================

def plot(scores, mean_scores):
    """Affiche les graphiques de progression."""
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Entraînement du Snake IA')
    plt.xlabel('Nombre de parties')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Score moyen')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)

# ============================================================================
# BOUCLE D'ENTRAÎNEMENT PRINCIPALE
# ============================================================================

def train(headless=False):
    """
    Boucle d'entraînement principale pour l'agent.
    
    Args:
        headless: Si True, pas d'interface graphique (entraînement plus rapide)
    
    Processus:
    1. Obtenir l'état actuel
    2. Prédire l'action avec le modèle
    3. Exécuter l'action et obtenir la récompense
    4. Mémoriser la transition
    5. Entraîner le modèle
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(load_model=True)  # Charge automatiquement le modèle si disponible
    game = SnakeGameAI(headless=headless)
    
    print("\n=== ENTRAÎNEMENT DU SNAKE IA ===")
    print("Deep Q-Learning avec PyTorch")
    if headless:
        print("🚀 Mode HEADLESS activé (entraînement ultra rapide, pas d'affichage)")
    else:
        print("👁️  Mode VISUEL activé (vous pouvez voir l'IA jouer)")
    print("Le meilleur modèle sera sauvegardé dans 'best_model.pth'")
    print("Appuyez sur Ctrl+C pour arrêter\n")
    
    try:
        while True:
            # 1. Obtenir l'état actuel
            state_old = agent.get_state(game)
            
            # 2. Prédire l'action
            final_move = agent.get_action(state_old)
            
            # 3. Exécuter l'action et obtenir le feedback
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            
            # 4. Entraîner sur la transition immédiate (short memory)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            
            # 5. Mémoriser la transition
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                # Fin de partie: entraîner sur l'expérience accumulée (long memory)
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                
                # Sauvegarder le meilleur modèle
                if score > record:
                    record = score
                    agent.model.save('best_model.pth')
                    print(f'🏆 Partie {agent.n_games} | Score: {score} | NOUVEAU RECORD: {record} ✨')
                else:
                    print(f'Partie {agent.n_games} | Score: {score} | Record: {record}')
                
                # Mettre à jour le record dans le jeu pour l'affichage
                game.set_record(record)
                
                # Mise à jour des graphiques
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                # plot(plot_scores, plot_mean_scores)  # Décommenter pour afficher les graphiques
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Entraînement arrêté par l'utilisateur")
        print(f"📊 Parties jouées: {agent.n_games}")
        print(f"🏆 Meilleur score: {record}")
        
        # Sauvegarder le modèle final
        if agent.n_games > 0:
            agent.model.save('final_model.pth')
            print("💾 Modèle final sauvegardé dans 'final_model.pth'")
        
        if record > 0:
            print(f"💾 Meilleur modèle disponible dans 'best_model.pth' (record: {record})")
        
        print("\n✅ Relancez le programme pour continuer l'entraînement depuis le meilleur modèle !")
        pygame.quit()

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    # Parser pour les arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Entraînement Snake IA avec Deep Q-Learning')
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Mode headless (sans interface graphique) pour un entraînement ultra rapide'
    )
    parser.add_argument(
        '--visual',
        action='store_true',
        help='Mode visuel (avec interface graphique) pour voir l\'IA jouer'
    )
    
    args = parser.parse_args()
    
    # Par défaut : mode VISUEL (pour voir l'IA jouer)
    if args.headless:
        headless = True
    else:
        headless = False  # Mode visuel par défaut
    
    train(headless=headless)
