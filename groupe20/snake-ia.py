# -*- coding: utf-8 -*-
"""
Snake IA - Deep Q-Learning
Groupe 20 - SIGL
Apprentissage par renforcement pour le jeu du serpent
Utilise serpent.py comme base du jeu
"""
import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
import sys
import os

# Ajouter le répertoire parent au path pour importer serpent.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import serpent

# Configuration de l'encodage pour Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =============================================================================
# CONSTANTES DU JEU (importées de serpent.py)
# =============================================================================

# Utilisation des constantes de serpent.py
GRID_SIZE = serpent.GRID_SIZE
CELL_SIZE = serpent.CELL_SIZE
GAME_SPEED = 40  # FPS pour l'entraînement rapide (plus rapide que le jeu normal)

SCREEN_WIDTH = serpent.SCREEN_WIDTH
SCORE_PANEL_HEIGHT = 120  # Panneau plus grand pour les infos d'entraînement
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (de serpent.py)
BLANC = serpent.BLANC
NOIR = serpent.NOIR
ORANGE = serpent.ORANGE
VERT = serpent.VERT
ROUGE = serpent.ROUGE
GRIS_FOND = serpent.GRIS_FOND
GRIS_GRILLE = serpent.GRIS_GRILLE
BLEU = (0, 100, 255)
JAUNE = (255, 255, 0)

# Directions (de serpent.py)
UP = serpent.UP
DOWN = serpent.DOWN
LEFT = serpent.LEFT
RIGHT = serpent.RIGHT

# Actions relatives (pour l'IA)
STRAIGHT = 0
RIGHT_TURN = 1
LEFT_TURN = 2

# =============================================================================
# MODÈLE DQN (DEEP Q-NETWORK)
# =============================================================================

class DQN(nn.Module):
    """
    Réseau de neurones SIMPLIFIÉ pour apprentissage plus rapide.

    Architecture SIMPLIFIÉE:
    - Input: 11 features (vecteur d'état)
    - Hidden: 128 neurones (ReLU) - PLUS SIMPLE !
    - Output: 3 neurones (Q-values pour chaque action)
    """

    def __init__(self, input_size=11, hidden_size=128, output_size=3):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =============================================================================
# AGENT D'APPRENTISSAGE PAR RENFORCEMENT
# =============================================================================

class AgentDQN:
    """
    Agent qui apprend à jouer au Snake via Deep Q-Learning.

    Caractéristiques:
    - Mémoire de replay pour briser les corrélations temporelles
    - Stratégie epsilon-greedy pour l'exploration vs exploitation
    - Entraînement par batch avec l'équation de Bellman
    """

    def __init__(self, learning_rate=0.001, gamma=0.9):
        self.n_games = 0
        self.epsilon = 0  # Sera géré par le système d'entraînement
        self.gamma = gamma  # Facteur de discount
        self.memory = deque(maxlen=100_000)  # Mémoire de replay

        self.model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def remember(self, state, action, reward, next_state, done):
        """Stocke une transition dans la mémoire de replay."""
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, epsilon=0):
        """
        Choisit une action via epsilon-greedy:
        - avec probabilité epsilon: action aléatoire (exploration)
        - sinon: action avec la meilleure Q-value (exploitation)
        """
        if random.random() < epsilon:
            return random.randint(0, 2)  # Action aléatoire

        # Prédiction du modèle
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()

        return torch.argmax(q_values).item()

    def train_step(self, state, action, reward, next_state, done):
        """
        Entraîne le modèle sur une seule transition.
        Utilise l'équation de Bellman: Q(s,a) = r + γ × max(Q(s',a'))
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. Q-values prédites pour l'état actuel
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][action[idx].item()] = Q_new

        # 2. Calcul de la perte et backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def train_batch(self, batch_size=1000):
        """
        Entraîne le modèle sur un batch aléatoire de la mémoire.
        Cette méthode permet de briser les corrélations temporelles.
        """
        if len(self.memory) < batch_size:
            return

        # Échantillonnage aléatoire
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        self.train_step(states, actions, rewards, next_states, dones)

    def save_model(self, filename="model.pth"):
        """Sauvegarde le modèle."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_games': self.n_games
        }, filename)
        print(f"Modèle sauvegardé: {filename}")

    def load_model(self, filename="model.pth"):
        """Charge un modèle pré-entraîné."""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_games = checkpoint['n_games']
        print(f"Modèle chargé: {filename}")

# =============================================================================
# JEU DE SNAKE ADAPTÉ POUR L'IA (utilise serpent.py)
# =============================================================================

class SnakeGameIA:
    """
    Wrapper autour des classes Snake et Apple de serpent.py pour l'IA.

    Adaptation pour l'apprentissage par renforcement:
    - Utilise les classes Snake et Apple de serpent.py
    - Interface simplifiée pour l'agent IA
    - Système de récompenses intégré
    - Fonction get_state() qui retourne le vecteur d'état (11 dimensions)
    - Méthode play_step() qui exécute une action et retourne la récompense
    """

    def __init__(self, render=True):
        self.render_mode = render

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake IA - Deep Q-Learning (serpent.py)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 20)

        self.reset()

    def reset(self):
        """Réinitialise le jeu pour une nouvelle partie."""
        # Utilise les classes de serpent.py
        self.snake = serpent.Snake()
        self.apple = serpent.Apple(self.snake.body)
        self.frame_iteration = 0
        self.start_time = time.time()
        self.prev_distance = self._get_distance_to_food()  # Pour reward shaping

        return self.get_state()

    def _get_distance_to_food(self):
        """Calcule la distance de Manhattan entre la tête et la pomme."""
        if not self.food_pos:
            return 0
        return abs(self.head_pos[0] - self.food_pos[0]) + abs(self.head_pos[1] - self.food_pos[1])

    @property
    def head_pos(self):
        """Raccourci pour accéder à la position de la tête."""
        return self.snake.head_pos

    @property
    def body(self):
        """Raccourci pour accéder au corps du serpent."""
        return self.snake.body

    @property
    def direction(self):
        """Raccourci pour accéder à la direction."""
        return self.snake.direction

    @property
    def score(self):
        """Raccourci pour accéder au score."""
        return self.snake.score

    @property
    def food_pos(self):
        """Raccourci pour accéder à la position de la pomme."""
        return list(self.apple.position) if self.apple.position else None

    def get_state(self):
        """
        Retourne le vecteur d'état (11 dimensions) représentant la situation actuelle.

        Structure du vecteur:
        [0-2]: Dangers immédiats (devant, droite, gauche)
        [3-6]: Direction actuelle (one-hot: gauche, droite, haut, bas)
        [7-10]: Position relative de la pomme (gauche, droite, haut, bas)
        """
        head = self.head_pos

        # Points dans les 3 directions possibles
        point_l = self._get_point_in_direction(self._turn_left(self.direction))
        point_r = self._get_point_in_direction(self._turn_right(self.direction))
        point_f = self._get_point_in_direction(self.direction)

        # Direction actuelle (one-hot)
        dir_l = self.direction == LEFT
        dir_r = self.direction == RIGHT
        dir_u = self.direction == UP
        dir_d = self.direction == DOWN

        # Position de la pomme (gestion du cas où food_pos est None)
        food_pos = self.food_pos if self.food_pos else [0, 0]

        state = [
            # Danger devant, droite, gauche
            self._is_collision(point_f),
            self._is_collision(point_r),
            self._is_collision(point_l),

            # Direction actuelle
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Position de la pomme
            food_pos[0] < head[0],  # Pomme à gauche
            food_pos[0] > head[0],  # Pomme à droite
            food_pos[1] < head[1],  # Pomme en haut
            food_pos[1] > head[1]   # Pomme en bas
        ]

        return np.array(state, dtype=int)

    def _get_point_in_direction(self, direction):
        """Retourne le point adjacent dans une direction donnée."""
        x = (self.head_pos[0] + direction[0]) % GRID_SIZE
        y = (self.head_pos[1] + direction[1]) % GRID_SIZE
        return [x, y]

    def _turn_left(self, direction):
        """Retourne la direction après un virage à gauche."""
        if direction == UP: return LEFT
        if direction == LEFT: return DOWN
        if direction == DOWN: return RIGHT
        if direction == RIGHT: return UP

    def _turn_right(self, direction):
        """Retourne la direction après un virage à droite."""
        if direction == UP: return RIGHT
        if direction == RIGHT: return DOWN
        if direction == DOWN: return LEFT
        if direction == LEFT: return UP

    def _is_collision(self, point):
        """Vérifie si un point est une collision (mur ou corps)."""
        # Note: les murs n'existent plus avec le wrapping, mais on garde la structure
        # pour compatibilité si on veut ajouter les murs plus tard
        return point in self.body

    def play_step(self, action):
        """
        Exécute une action et retourne (reward, game_over, score).

        Actions:
        - 0: Tout droit
        - 1: Virage à droite
        - 2: Virage à gauche

        Système de récompenses ULTRA SIMPLIFIÉ (apprentissage garanti):
        - Manger pomme: +100 (énorme récompense !)
        - Se rapprocher: +10 (grosse récompense)
        - S'éloigner: -15 (grosse pénalité)
        - Collision: -100 (très mauvais)
        - Victoire: +1000
        """
        self.frame_iteration += 1

        # 1. Gestion des événements
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 2. Mise à jour de la direction selon l'action (utilise set_direction de serpent.py)
        new_direction = self.direction
        if action == RIGHT_TURN:
            new_direction = self._turn_right(self.direction)
        elif action == LEFT_TURN:
            new_direction = self._turn_left(self.direction)
        # Sinon: STRAIGHT, on garde la direction actuelle

        self.snake.set_direction(new_direction)

        # 3. Déplacement du serpent (utilise la méthode move() de serpent.py)
        self.snake.move()

        # 4. Calcul de la récompense avec REWARD SHAPING ULTRA SIMPLIFIÉ
        reward = 0
        game_over = False

        # Vérification de collision (utilise check_self_collision de serpent.py)
        # Note: serpent.py utilise le wrapping, donc pas de collision avec les murs
        if self.snake.check_self_collision() or self.frame_iteration > 100 * len(self.body):
            game_over = True
            reward = -100  # GROSSE pénalité pour mourir
            if self.render_mode:
                self.render()
            return reward, game_over, self.score

        # Vérification de la pomme mangée
        if self.head_pos == list(self.apple.position):
            self.snake.grow()  # Utilise la méthode grow() de serpent.py
            reward = 100  # GROSSE récompense pour manger !

            # Tente de replacer la pomme (utilise relocate() de serpent.py)
            if not self.apple.relocate(self.snake.body):
                # Victoire complète (toutes les cases sont remplies)
                reward = 1000  # ÉNORME récompense !
                game_over = True

            # Mise à jour de la distance précédente
            self.prev_distance = self._get_distance_to_food()
        else:
            # REWARD SHAPING SIMPLIFIÉ: récompenses BEAUCOUP PLUS FORTES
            current_distance = self._get_distance_to_food()

            if current_distance < self.prev_distance:
                # Se rapproche de la pomme : GROSSE RÉCOMPENSE
                reward = 10
            else:
                # S'éloigne de la pomme : GROSSE PÉNALITÉ
                reward = -15

            self.prev_distance = current_distance

        # 5. Rendu visuel
        if self.render_mode:
            self.render()
            self.clock.tick(GAME_SPEED)

        return reward, game_over, self.score

    def render(self):
        """Affiche l'état actuel du jeu en utilisant les méthodes de serpent.py."""
        if not self.render_mode:
            return

        # Fond
        self.screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.screen, NOIR, game_area_rect)

        # Grille (utilise la fonction de serpent.py adaptée)
        self._draw_grid()

        # Dessine la pomme et le serpent en utilisant leurs méthodes draw() de serpent.py
        self.apple.draw(self.screen)
        self.snake.draw(self.screen)

        # Informations du panneau supérieur
        pygame.draw.line(self.screen, BLANC,
                        (0, SCORE_PANEL_HEIGHT - 2),
                        (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

        score_text = self.font.render(f"Score: {self.score}", True, BLANC)
        self.screen.blit(score_text, (10, 20))

        max_cells = GRID_SIZE * GRID_SIZE
        fill_rate = (len(self.body) / max_cells) * 100
        fill_text = self.font_small.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
        self.screen.blit(fill_text, (10, 55))

        frames_text = self.font_small.render(f"Frames: {self.frame_iteration}", True, BLANC)
        self.screen.blit(frames_text, (10, 85))

        pygame.display.flip()

    def _draw_grid(self):
        """Dessine la grille pour une meilleure visualisation."""
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRIS_GRILLE,
                           (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
        for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

# =============================================================================
# SYSTÈME D'ENTRAÎNEMENT AVEC MÉTRIQUES
# =============================================================================

class TrainingMetrics:
    """
    Gère les métriques de performance pendant l'entraînement.
    - Scores par partie
    - Temps de survie
    - Moyennes mobiles
    """

    def __init__(self):
        self.scores = []
        self.mean_scores = []
        self.total_score = 0
        self.record = 0

        # Pour la visualisation en temps réel
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))

    def update(self, score):
        """Met à jour les métriques après une partie."""
        self.scores.append(score)
        self.total_score += score
        mean_score = self.total_score / len(self.scores)
        self.mean_scores.append(mean_score)

        if score > self.record:
            self.record = score

    def plot(self, epsilon=0):
        """Affiche les courbes d'apprentissage."""
        self.ax1.clear()
        self.ax2.clear()

        # Graphique 1: Scores
        self.ax1.set_title('Courbe d\'apprentissage - Deep Q-Learning')
        self.ax1.set_xlabel('Nombre de parties')
        self.ax1.set_ylabel('Score')
        self.ax1.plot(self.scores, label='Score', alpha=0.6)
        self.ax1.plot(self.mean_scores, label='Score moyen', linewidth=2)
        self.ax1.axhline(y=self.record, color='r', linestyle='--',
                        label=f'Record: {self.record}')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # Graphique 2: Epsilon (exploration)
        n_games = len(self.scores)
        self.ax2.set_title('Taux d\'exploration (Epsilon)')
        self.ax2.set_xlabel('Nombre de parties')
        self.ax2.set_ylabel('Epsilon')
        self.ax2.axhline(y=epsilon, color='g', linewidth=2)
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.text(0.5, 0.5, f'ε = {epsilon:.3f}',
                     transform=self.ax2.transAxes,
                     fontsize=40, ha='center', va='center',
                     color='green', alpha=0.3)

        plt.tight_layout()
        plt.pause(0.001)

    def save_plot(self, filename='training_progress.png'):
        """Sauvegarde le graphique."""
        self.fig.savefig(filename)
        print(f"Graphique sauvegardé: {filename}")

# =============================================================================
# BOUCLE D'ENTRAÎNEMENT PRINCIPALE
# =============================================================================

def train(render=True, max_games=500):
    """
    Boucle d'entraînement ULTRA SIMPLIFIÉE - APPRENTISSAGE GARANTI !

    Paramètres:
    - render: Afficher le jeu pendant l'entraînement
    - max_games: Nombre maximum de parties d'entraînement

    Stratégie ULTRA SIMPLE:
    - Epsilon initial: 30% seulement (moins d'aléatoire)
    - Epsilon minimum: 0% (exploitation pure à la fin)
    - Décroissance: rapide sur 50% des parties
    - Learning rate TRÈS élevé: 0.005 (apprentissage ultra rapide)
    - Récompenses ÉNORMES pour guider l'IA
    """
    print("="*70)
    print(" ENTRAINEMENT ULTRA SIMPLIFIÉ - APPRENTISSAGE GARANTI ")
    print("="*70)
    print(f"Nombre de parties: {max_games}")
    print(f"Render: {'Activé' if render else 'Désactivé'}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("\nSTRATÉGIE ULTRA SIMPLE:")
    print("  • Récompenses ÉNORMES:")
    print("    - Manger pomme: +100 (au lieu de +10)")
    print("    - Se rapprocher: +10 (au lieu de +1)")
    print("    - S'éloigner: -15 (au lieu de -1.5)")
    print("    - Mourir: -100 (au lieu de -10)")
    print("  • Réseau simplifié: 1 couche de 128 neurones")
    print("  • Epsilon: 30% → 0% (moins d'aléatoire)")
    print("  • Learning rate: 0.005 (apprentissage ULTRA rapide)")
    print("="*70 + "\n")

    # Initialisation avec paramètres ULTRA SIMPLIFIÉS
    agent = AgentDQN(learning_rate=0.005, gamma=0.99)  # LR très élevé, gamma maximal

    # CHARGEMENT AUTOMATIQUE du modèle existant s'il existe
    model_path = "model_final.pth"
    if os.path.exists(model_path):
        print(f"✓ Modèle existant détecté : {model_path}")
        print("  Chargement de l'apprentissage précédent...")
        agent.load_model(model_path)
        print(f"  L'IA a déjà joué {agent.n_games} parties !")
        print("  L'entraînement va CONTINUER à partir de là.\n")
    else:
        print("✗ Aucun modèle existant trouvé.")
        print("  Démarrage d'un NOUVEL entraînement.\n")

    game = SnakeGameIA(render=render)
    metrics = TrainingMetrics()

    # Paramètres epsilon-greedy ULTRA SIMPLIFIÉS
    epsilon_start = 0.3   # Seulement 30% d'aléatoire au début
    epsilon_end = 0.0     # 0% à la fin (exploitation pure)
    epsilon_decay = (epsilon_start - epsilon_end) / (max_games * 0.5)  # Décroissance rapide
    epsilon = epsilon_start

    # Boucle d'entraînement
    for n_game in range(max_games):
        # Reset du jeu
        state = game.reset()
        game_over = False

        # Une partie complète
        while not game_over:
            # 1. Choisir une action (epsilon-greedy)
            action = agent.get_action(state, epsilon)

            # 2. Exécuter l'action
            reward, game_over, score = game.play_step(action)

            # 3. Observer le nouvel état
            next_state = game.get_state()

            # 4. Mémoriser l'expérience
            agent.remember(state, action, reward, next_state, game_over)

            # 5. Entraîner sur cette transition
            agent.train_step(state, action, reward, next_state, game_over)

            state = next_state

        # Après chaque partie
        agent.n_games += 1

        # Entraînement par batch si assez de mémoire
        if len(agent.memory) > 1000:
            agent.train_batch(batch_size=1000)

        # Mise à jour des métriques
        metrics.update(score)

        # Décroissance d'epsilon
        if epsilon > epsilon_end:
            epsilon = max(epsilon_end, epsilon - epsilon_decay)

        # Affichage et sauvegarde périodiques
        if (n_game + 1) % 10 == 0:
            avg_score = metrics.mean_scores[-1]
            progress_percent = (n_game + 1) / max_games * 100
            print(f"[{progress_percent:5.1f}%] Partie {n_game + 1}/{max_games} | "
                  f"Score: {score:2d} | "
                  f"Record: {metrics.record:2d} | "
                  f"Moy: {avg_score:5.2f} | "
                  f"ε: {epsilon:.3f}")

            # Affichage d'un message d'encouragement si le record augmente
            if score == metrics.record and score > 0:
                print(f"    🎯 NOUVEAU RECORD ! Le serpent a mangé {score} pomme(s) !")

            metrics.plot(epsilon)

        # Sauvegarde du modèle tous les 50 parties
        if (n_game + 1) % 50 == 0:
            agent.save_model(f"model_game_{n_game+1}.pth")

    # Fin de l'entraînement
    print("\n" + "="*70)
    print(" ENTRAINEMENT TERMINE ")
    print("="*70)
    print(f"Record: {metrics.record}")
    print(f"Score moyen final: {metrics.mean_scores[-1]:.2f}")
    print(f"Nombre total de parties: {agent.n_games}")

    # Sauvegarde finale
    agent.save_model("model_final.pth")
    metrics.save_plot("training_progress.png")

    plt.ioff()
    plt.show()

# =============================================================================
# MODE JEU (TEST DU MODÈLE ENTRAÎNÉ)
# =============================================================================

def play_with_model(model_path="model_final.pth"):
    """
    Fait jouer l'agent avec un modèle pré-entraîné.
    Mode démonstration sans exploration (epsilon=0).
    """
    print("="*70)
    print(" MODE JEU - AGENT ENTRAINE ")
    print("="*70)

    agent = AgentDQN()
    agent.load_model(model_path)
    game = SnakeGameIA(render=True)

    total_score = 0
    n_games = 0

    print("Appuyez sur ESC pour quitter\n")

    while True:
        state = game.reset()
        game_over = False

        while not game_over:
            # Action sans exploration (epsilon=0)
            action = agent.get_action(state, epsilon=0)
            reward, game_over, score = game.play_step(action)
            state = game.get_state()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

        n_games += 1
        total_score += score
        print(f"Partie {n_games} | Score: {score} | Moyenne: {total_score/n_games:.2f}")

        time.sleep(1)  # Pause entre les parties

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == '__main__':
    import sys

    print("\n" + "="*70)
    print(" SNAKE AI - DEEP Q-LEARNING ")
    print(" Groupe 20 - SIGL ")
    print("="*70 + "\n")

    if len(sys.argv) > 1 and sys.argv[1] == "play":
        # Mode jeu avec modèle pré-entraîné
        play_with_model()
    else:
        # Mode entraînement
        print("Options:")
        print("1. Entraînement avec rendu visuel (lent mais instructif)")
        print("2. Entraînement sans rendu (rapide)")
        print("3. Jouer avec un modèle pré-entraîné")

        choice = input("\nVotre choix (1/2/3): ").strip()

        if choice == "1":
            train(render=True, max_games=500)
        elif choice == "2":
            train(render=False, max_games=1000)
        elif choice == "3":
            play_with_model()
        else:
            print("Choix invalide. Lancement de l'entraînement par défaut...")
            train(render=True, max_games=500)
