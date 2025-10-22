"""
Snake IA - Deep Q-Learning
Implémentation d'un agent d'apprentissage par renforcement pour jouer à Snake
Utilise PyTorch pour le réseau neuronal et le GPU si disponible
"""

import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import sys
import os
import matplotlib.pyplot as plt
from IPython import display
import torch.cuda.amp as amp  # Mixed precision training pour GPU

# Ajouter le répertoire parent au path pour importer serpent.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serpent import Snake, Apple, GRID_SIZE, CELL_SIZE, SCREEN_WIDTH, SCORE_PANEL_HEIGHT, SCREEN_HEIGHT
from serpent import BLANC, NOIR, ORANGE, VERT, ROUGE, GRIS_FOND, GRIS_GRILLE
from serpent import UP, DOWN, LEFT, RIGHT

# =============================================================================
# HYPERPARAMÈTRES
# =============================================================================
MAX_MEMORY = 100_000  # Taille maximale de la mémoire de replay
BATCH_SIZE = 3000     # Taille des batches pour l'entraînement
LR = 0.001            # Taux d'apprentissage
GAMMA = 0.8           # Facteur de discount
EPSILON_START = 80    # Exploration initiale
GAME_SPEED = 1000      # Vitesse du jeu pour l'IA

# =============================================================================
# BLOC 1: MODEL (TORCH) - Réseau de neurones profond
# =============================================================================

class Linear_QNet(nn.Module):
    """
    Réseau de neurones profond pour le Q-Learning.
    Architecture: Input -> Hidden1 -> Hidden2 -> Output
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Propagation avant à travers le réseau."""
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        """Sauvegarde le modèle."""
        model_folder_path = '../groupe13/model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """Gestionnaire d'entraînement du modèle avec support GPU."""
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Détection automatique du GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"🖥️  Entraînement sur: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    def train_step(self, state, action, reward, next_state, done):
        """Entraîne le modèle sur un batch d'expériences."""
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        if len(state.shape) == 1:
            # Reshape pour un seul échantillon (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. Prédiction Q avec l'état actuel
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Formule Q-Learning: Q_new = r + γ * max(Q_next)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2. Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class OptimizedQTrainer(QTrainer):
    """
    Version optimisée du QTrainer avec:
    - Mixed precision training (float16) pour GPU - gain ~30%
    - Batch processing vectorisé
    - Calculs parallèles sur GPU
    """
    def __init__(self, model, lr, gamma):
        super().__init__(model, lr, gamma)

        # Mixed precision scaler pour GPU (entraînement en float16)
        self.scaler = amp.GradScaler() if torch.cuda.is_available() else None

        if self.scaler:
            print(f"⚡ Mixed Precision (float16): ACTIVÉ - Gain ~30%")

    def train_step_optimized(self, states, actions, rewards, next_states, dones):
        """
        Version optimisée de train_step avec mixed precision et vectorisation.
        Traite tout le batch en parallèle sur le GPU.

        Args:
            states: Liste ou array d'états
            actions: Liste ou array d'actions
            rewards: Liste ou array de récompenses
            next_states: Liste ou array de nouveaux états
            dones: Liste ou array de booléens (jeu terminé)
        """
        # Conversion en tensors sur GPU en une seule fois (très rapide)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(self.device)

        if self.scaler:
            # === MODE GPU OPTIMISÉ AVEC MIXED PRECISION ===
            with torch.amp.autocast('cuda'):  # Utilise float16 automatiquement
                # Prédiction Q avec l'état actuel (parallèle sur tout le batch)
                pred = self.model(states)

                # Calcul de Q_target de manière vectorisée (TOUT EN PARALLÈLE)
                with torch.no_grad():  # Pas de gradient pour next_states (économie mémoire)
                    next_q_values = self.model(next_states)
                    max_next_q = torch.max(next_q_values, dim=1)[0]

                    # Q_new = reward + gamma * max(Q_next) si pas done
                    # Opération vectorisée : tout le batch en une seule ligne !
                    target_q = rewards + self.gamma * max_next_q * (~dones).float()

                # Mise à jour des targets (vectorisé)
                target = pred.clone()
                batch_indices = torch.arange(len(dones)).to(self.device)
                action_indices = torch.argmax(actions, dim=1)
                # Convertir target_q au même dtype que target (float16 en autocast)
                target[batch_indices, action_indices] = target_q.to(target.dtype)

                # Loss sur tout le batch
                loss = self.criterion(pred, target)

            # Backpropagation avec mixed precision (gère automatiquement float16/32)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            # === MODE CPU (SANS MIXED PRECISION) ===
            pred = self.model(states)

            with torch.no_grad():
                next_q_values = self.model(next_states)
                max_next_q = torch.max(next_q_values, dim=1)[0]
                target_q = rewards + self.gamma * max_next_q * (~dones).float()

            target = pred.clone()
            batch_indices = torch.arange(len(dones))
            action_indices = torch.argmax(actions, dim=1)
            target[batch_indices, action_indices] = target_q

            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

# =============================================================================
# BLOC 2: GAME (PyGame) - Adaptation du jeu pour l'IA
# =============================================================================

class SnakeGameAI:
    """Adaptation du jeu Snake pour l'entraînement IA."""

    def __init__(self, speed=GAME_SPEED, display_enabled=False):
        pygame.init()
        self.speed = speed
        self.display_enabled = display_enabled

        if self.display_enabled:
            # Mode avec affichage visuel
            self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Snake IA - Deep Q-Learning')
            self.font = pygame.font.Font(None, 25)
        else:
            # Mode sans affichage (plus rapide)
            # Crée une surface invisible pour que les méthodes draw() fonctionnent
            self.display = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.font = pygame.font.Font(None, 25)

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Réinitialise le jeu."""
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.frame_iteration = 0
        self.score = 0

    def play_step(self, action):
        """
        Effectue un pas de jeu avec l'action donnée.
        Action: [tout droit, tourner droite, tourner gauche]

        Returns:
            reward: récompense pour cette action
            game_over: booléen indiquant si le jeu est terminé
            score: score actuel
        """
        self.frame_iteration += 1

        # 1. Gérer les événements pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Déplacer le serpent selon l'action
        self._move(action)
        self.snake.move()

        # 3. Calculer la récompense
        reward = 0
        game_over = False

        # Timeout si le serpent tourne en rond (évite les boucles infinies)
        if self.snake.is_game_over() or self.frame_iteration > 100 * len(self.snake.body):
            game_over = True
            reward = -10  # Récompense: Perdre -10
            return reward, game_over, self.score

        # 4. Vérifier si la pomme est mangée
        if self.snake.head_pos == list(self.apple.position):
            self.score += 1
            reward = 10  # Récompense: Attraper la pomme +10
            self.snake.grow()

            # Vérifier la victoire (toute la grille remplie)
            if not self.apple.relocate(self.snake.body):
                reward = 100  # Récompense: Finir le jeu +100
                game_over = True

        # 5. Mettre à jour l'interface (seulement si affichage activé)
        if self.display_enabled:
            self._update_ui()
            self.clock.tick(self.speed)

        return reward, game_over, self.score

    def _move(self, action):
        """
        Convertit l'action [tout droit, droite, gauche] en direction.
        Action[0] = tout droit
        Action[1] = tourner à droite
        Action[2] = tourner à gauche
        """
        # Directions dans le sens horaire: RIGHT, DOWN, LEFT, UP
        clock_wise = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wise.index(self.snake.direction)

        if np.array_equal(action, [1, 0, 0]):
            # Tout droit - pas de changement
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Tourner à droite (sens horaire)
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            # Tourner à gauche (sens anti-horaire)
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.snake.set_direction(new_dir)

    def is_collision(self, pt=None):
        """Vérifie s'il y a collision à un point donné."""
        if pt is None:
            pt = self.snake.head_pos

        # Collision avec les bords
        if pt[0] < 0 or pt[0] >= GRID_SIZE or pt[1] < 0 or pt[1] >= GRID_SIZE:
            return True

        # Collision avec soi-même
        if pt in self.snake.body[1:]:
            return True

        return False

    def _update_ui(self):
        """
        Met à jour l'affichage du jeu pygame - appelée à chaque frame.

        Cette fonction est responsable de dessiner tous les éléments visuels:
        1. Le fond et la zone de jeu
        2. La grille de repérage
        3. La pomme (nourriture)
        4. Le serpent (tête et corps)
        5. Les informations (score, frames)
        6. Rafraîchir l'écran

        L'ordre de dessin est important: les éléments dessinés en dernier
        apparaissent au-dessus des précédents (système de layers).
        """
        # === ÉTAPE 1: FOND ===
        # Remplit tout l'écran avec la couleur de fond grise
        # Cela efface le contenu de la frame précédente
        self.display.fill(GRIS_FOND)

        # === ÉTAPE 2: ZONE DE JEU ===
        # Crée un rectangle noir pour la zone de jeu principale
        # Position: x=0, y=SCORE_PANEL_HEIGHT (décalé vers le bas pour laisser place au panneau de score)
        # Taille: SCREEN_WIDTH x SCREEN_WIDTH (zone carrée)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        # Dessine ce rectangle en noir sur l'écran
        pygame.draw.rect(self.display, NOIR, game_area_rect)

        # === ÉTAPE 3: GRILLE ===
        # Dessine les lignes de la grille pour visualiser les cellules
        # Facilite la compréhension des mouvements du serpent
        self._draw_grid()

        # === ÉTAPE 4: POMME ===
        # Dessine la pomme (nourriture) à sa position actuelle
        # Utilise la méthode draw() de l'objet Apple (définie dans serpent.py)
        self.apple.draw(self.display)

        # === ÉTAPE 5: SERPENT ===
        # Dessine le serpent complet (tête en orange, corps en vert)
        # Utilise la méthode draw() de l'objet Snake (définie dans serpent.py)
        # Le serpent est dessiné après la pomme pour apparaître au-dessus
        self.snake.draw(self.display)

        # === ÉTAPE 6: INFORMATIONS (HUD - Head-Up Display) ===
        # Affiche le score actuel en haut à gauche
        # font.render() crée une surface de texte avec:
        #   - Le texte à afficher
        #   - True pour l'antialiasing (lissage)
        #   - BLANC pour la couleur
        score_text = self.font.render(f"Score: {self.score}", True, BLANC)
        # blit() copie la surface de texte sur l'écran à la position (10, 20)
        self.display.blit(score_text, (10, 20))

        # Affiche le nombre de frames (itérations) en haut à droite
        # Utile pour le débogage: permet de voir si l'IA tourne en rond
        # Le timeout est déclenché à 100 * len(snake.body) frames
        frame_text = self.font.render(f"Frames: {self.frame_iteration}", True, BLANC)
        # Positionné à droite (SCREEN_WIDTH - 150 pixels du bord droit)
        self.display.blit(frame_text, (SCREEN_WIDTH - 150, 20))

        # === ÉTAPE 7: RAFRAÎCHISSEMENT ===
        # pygame.display.flip() actualise l'affichage complet
        # Utilise la technique du "double buffering":
        #   - On dessine sur un buffer invisible
        #   - flip() échange le buffer visible avec l'invisible
        # Cela évite le scintillement et assure un affichage fluide
        pygame.display.flip()

    def _draw_grid(self):
        """
        Dessine la grille de repérage sur la zone de jeu.

        Cette fonction trace des lignes verticales et horizontales pour visualiser
        les cellules de la grille. Cela aide à:
        - Comprendre les mouvements du serpent (cellule par cellule)
        - Déboguer le positionnement des éléments
        - Améliorer l'aspect visuel du jeu

        La grille utilise la couleur GRIS_GRILLE pour rester discrète.
        """
        # === LIGNES VERTICALES ===
        # Parcourt l'écran horizontalement par pas de CELL_SIZE (30 pixels)
        # range(0, SCREEN_WIDTH, CELL_SIZE) génère: 0, 30, 60, 90, ...
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            # pygame.draw.line(surface, couleur, point_départ, point_arrivée)
            # Ligne du haut (décalée par SCORE_PANEL_HEIGHT) au bas de l'écran
            pygame.draw.line(self.display, GRIS_GRILLE,
                           (x, SCORE_PANEL_HEIGHT),      # Point de départ (x, top)
                           (x, SCREEN_HEIGHT))           # Point d'arrivée (x, bottom)

        # === LIGNES HORIZONTALES ===
        # Parcourt l'écran verticalement par pas de CELL_SIZE
        # Commence à SCORE_PANEL_HEIGHT pour ne pas dessiner sur le panneau de score
        for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
            # Ligne de gauche (x=0) à droite (x=SCREEN_WIDTH)
            pygame.draw.line(self.display, GRIS_GRILLE,
                           (0, y),                       # Point de départ (left, y)
                           (SCREEN_WIDTH, y))            # Point d'arrivée (right, y)

# =============================================================================
# BLOC 3: AGENT - Logique d'apprentissage par renforcement
# =============================================================================

class Agent:
    """Agent d'apprentissage par renforcement utilisant Deep Q-Learning."""

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Contrôle exploration vs exploitation
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)  # Mémoire de replay

        # Modèle: 11 états en entrée, 3 actions en sortie
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # Device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_state(self, game):
        """
        Obtient l'état actuel du jeu sous forme de vecteur de 11 éléments.

        États (11 bits):
        - Danger [tout droit, droite, gauche] (3 bits)
        - Direction actuelle [gauche, droite, haut, bas] (4 bits)
        - Position nourriture [gauche, droite, haut, bas] (4 bits)
        """
        head = game.snake.head_pos

        # Points autour de la tête (pour détecter les dangers)
        point_l = [head[0] - 1, head[1]]
        point_r = [head[0] + 1, head[1]]
        point_u = [head[0], head[1] - 1]
        point_d = [head[0], head[1] + 1]

        # Direction actuelle
        dir_l = game.snake.direction == LEFT
        dir_r = game.snake.direction == RIGHT
        dir_u = game.snake.direction == UP
        dir_d = game.snake.direction == DOWN

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

            # Direction actuelle (4 bits one-hot)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Position de la nourriture par rapport à la tête (4 bits)
            game.apple.position[0] < head[0],  # Pomme à gauche
            game.apple.position[0] > head[0],  # Pomme à droite
            game.apple.position[1] < head[1],  # Pomme en haut
            game.apple.position[1] > head[1]   # Pomme en bas
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience dans la mémoire de replay."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Entraîne le modèle sur un batch d'expériences passées (replay d'expérience)."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Entraîne le modèle sur une seule expérience (apprentissage immédiat)."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Choisit une action basée sur l'état actuel.
        Utilise epsilon-greedy: exploration aléatoire au début, puis exploitation.

        Returns:
            action: [tout droit, droite, gauche] - un seul élément à 1
        """
        # Exploration vs Exploitation (epsilon diminue avec le temps)
        self.epsilon = EPSILON_START - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # Exploration: mouvement aléatoire
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation: utilise le modèle pour prédire la meilleure action
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


class OptimizedAgent(Agent):
    """
    Version optimisée de l'Agent avec:
    - Batch processing amélioré et doublé sur GPU
    - Utilisation de OptimizedQTrainer
    - Entraînement long memory vectorisé
    """
    def __init__(self):
        super().__init__()
        # Remplacer le trainer par la version optimisée
        self.trainer = OptimizedQTrainer(self.model, lr=LR, gamma=self.gamma)

        # Batch size plus grand pour mieux utiliser le GPU
        if torch.cuda.is_available():
            self.batch_size = BATCH_SIZE * 2
            print(f"🔥 Batch Size GPU: {self.batch_size} (x2 pour optimisation)")
        else:
            self.batch_size = BATCH_SIZE
            print(f"📊 Batch Size CPU: {self.batch_size}")

    def train_long_memory_optimized(self):
        """
        Version optimisée avec batch processing parallèle sur GPU.
        Traite tout le batch en une seule fois de manière vectorisée.
        """
        if len(self.memory) < self.batch_size:
            # Pas assez de mémoire, utiliser tout
            mini_sample = list(self.memory)
        else:
            # Échantillonnage aléatoire
            mini_sample = random.sample(self.memory, self.batch_size)

        # Décompression des tuples (plus efficace que zip(*))
        states, actions, rewards, next_states, dones = map(list, zip(*mini_sample))

        # Entraînement avec la version optimisée (vectorisée sur GPU)
        self.trainer.train_step_optimized(states, actions, rewards, next_states, dones)

    def get_action_batch(self, states):
        """
        Prédit les actions pour plusieurs états en parallèle sur GPU.
        Utile pour traiter plusieurs expériences simultanément.

        Args:
            states: Liste d'états numpy arrays

        Returns:
            Liste d'actions [tout droit, droite, gauche]
        """
        # Convertir en tensor batch
        states_tensor = torch.tensor(np.array(states), dtype=torch.float).to(self.device)

        # Prédiction parallèle sur GPU (très rapide)
        with torch.no_grad():
            predictions = self.model(states_tensor)
            moves = torch.argmax(predictions, dim=1).cpu().numpy()

        # Convertir en format one-hot
        actions = []
        for move in moves:
            action = [0, 0, 0]
            action[move] = 1
            actions.append(action)

        return actions

# =============================================================================
# VISUALISATION
# =============================================================================

# Active le mode interactif de matplotlib
# Permet de mettre à jour les graphiques en temps réel sans bloquer le programme
plt.ion()

def plot(scores, mean_scores):
    """
    Affiche les graphiques d'entraînement en temps réel avec matplotlib.

    Cette fonction crée un graphique dynamique montrant:
    - Les scores individuels de chaque partie (courbe bleue)
    - La moyenne mobile des scores (courbe rouge)

    Le graphique se met à jour automatiquement après chaque partie terminée.

    Args:
        scores: Liste des scores de chaque partie [score1, score2, ...]
        mean_scores: Liste des scores moyens cumulés [moy1, moy2, ...]

    Technique utilisée:
        - IPython.display pour l'affichage dynamique
        - matplotlib.pyplot pour le tracé des courbes
        - Mode interactif (ion) pour éviter de bloquer le programme
    """
    # === ÉTAPE 1: EFFACEMENT DE L'AFFICHAGE PRÉCÉDENT ===
    # display.clear_output() efface le graphique précédent dans la console/notebook
    # wait=True évite le scintillement en attendant que le nouveau graphique soit prêt
    # Cela permet une transition fluide entre les frames
    display.clear_output(wait=True)

    # === ÉTAPE 2: AFFICHAGE DE LA FIGURE ACTUELLE ===
    # plt.gcf() = "get current figure" récupère la figure matplotlib active
    # display.display() affiche cette figure dans la console IPython/Jupyter
    display.display(plt.gcf())

    # === ÉTAPE 3: EFFACEMENT DU CONTENU DE LA FIGURE ===
    # plt.clf() = "clear figure" efface tout le contenu de la figure
    # Sans cela, les courbes se superposeraient à chaque appel
    plt.clf()

    # === ÉTAPE 4: CONFIGURATION DU GRAPHIQUE ===
    # Définit le titre principal du graphique (en haut)
    plt.title('Entraînement Deep Q-Learning - Snake IA')
    # Label de l'axe X (horizontal) - représente le numéro de partie
    plt.xlabel('Nombre de parties')
    # Label de l'axe Y (vertical) - représente le score obtenu
    plt.ylabel('Score')

    # === ÉTAPE 5: TRACÉ DES COURBES ===
    # Courbe des scores individuels (bleu transparent)
    # plt.plot(y_values, options...) trace une ligne reliant tous les points
    # alpha=0.6 rend la ligne semi-transparente (60% opaque)
    # Cela permet de voir la tendance sans surcharger visuellement
    plt.plot(scores, label='Score', color='blue', alpha=0.6)

    # Courbe du score moyen (rouge épaisse)
    # linewidth=2 rend cette courbe plus visible (2 pixels d'épaisseur)
    # C'est l'indicateur clé: montre si l'IA s'améliore dans le temps
    # Un score moyen croissant = apprentissage réussi
    plt.plot(mean_scores, label='Score moyen', color='red', linewidth=2)

    # === ÉTAPE 6: CONFIGURATION DES AXES ===
    # ylim(ymin=0) fixe le minimum de l'axe Y à 0
    # Les scores ne peuvent pas être négatifs, donc on part de 0
    # ymax est automatique (s'adapte au score maximum)
    plt.ylim(ymin=0)

    # === ÉTAPE 7: AJOUT DE LA LÉGENDE ===
    # Affiche une boîte explicative identifiant chaque courbe
    # Utilise les 'label' définis dans plt.plot()
    plt.legend()

    # === ÉTAPE 8: AFFICHAGE DES VALEURS FINALES ===
    # plt.text(x, y, texte) place du texte à une position donnée
    # len(scores)-1 = index de la dernière partie (position X)
    # scores[-1] = dernier score obtenu (position Y)
    # Affiche la valeur exacte du dernier score
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))

    # Affiche la valeur du dernier score moyen (formaté avec 2 décimales)
    # .2f = format float avec 2 chiffres après la virgule
    plt.text(len(mean_scores)-1, mean_scores[-1], f'{mean_scores[-1]:.2f}')

    # === ÉTAPE 9: AJOUT DE LA GRILLE ===
    # plt.grid(True) active la grille de fond
    # alpha=0.3 rend la grille discrète (30% opaque)
    # Facilite la lecture des valeurs sans surcharger visuellement
    plt.grid(True, alpha=0.3)

    # === ÉTAPE 10: AFFICHAGE ET RAFRAÎCHISSEMENT ===
    # plt.show(block=False) affiche le graphique sans bloquer le programme
    # block=False permet au code de continuer à s'exécuter
    # Si block=True, le programme attendrait que l'utilisateur ferme la fenêtre
    plt.show(block=False)

    # === ÉTAPE 11: PAUSE POUR RAFRAÎCHISSEMENT ===
    # plt.pause(0.1) fait une pause de 0.1 seconde (100 millisecondes)
    # Cette pause est NÉCESSAIRE pour que matplotlib actualise la fenêtre
    # Sans cela, la fenêtre pourrait ne pas se rafraîchir correctement
    plt.pause(.1)

# =============================================================================
# FONCTION D'ENTRAÎNEMENT ULTRA-RAPIDE (OPTIMISÉE GPU)
# =============================================================================

def train_ultra_fast(max_games=10000):
    """
    Version ULTRA-RAPIDE avec toutes les optimisations GPU activées:
    - Mixed precision training (float16) - gain ~30%
    - Batch processing doublé
    - Vectorisation maximale sur GPU
    - Entraînement long memory tous les 5 jeux
    - Pas d'affichage pendant l'entraînement

    Args:
        max_games: Nombre de parties (défaut: 10 000)

    Returns:
        record: Meilleur score
        scores: Liste des scores
    """
    print("=" * 60)
    print(f"🚀 ENTRAÎNEMENT ULTRA-RAPIDE - {max_games} PARTIES")
    print("=" * 60)
    print(f"GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 Mémoire GPU: {gpu_mem:.2f} GB")
        print(f"⚡ Mixed Precision (float16): ACTIVÉ")
        print(f"🔥 Batch Size: {BATCH_SIZE * 2} (doublé)")
        print(f"⚡ Gain de vitesse attendu: 10-15x")
    else:
        print(f"⚠️  Mode CPU - Optimisations limitées")
    print("=" * 60)
    print("\n⚡ Mode ultra-rapide - Affichage minimal")
    print("📊 Statistiques tous les 100 parties\n")

    scores = []
    total_score = 0
    record = 0

    # Utiliser l'agent optimisé avec mixed precision
    agent = OptimizedAgent()

    # Jeu avec affichage (vitesse rapide mais visible)
    game = SnakeGameAI(speed=GAME_SPEED, display_enabled=True)

    # Désactiver matplotlib pendant l'entraînement
    plt.ioff()

    while agent.n_games < max_games:
        # 1. État actuel
        state_old = agent.get_state(game)

        # 2. Action
        final_move = agent.get_action(state_old)

        # 3. Jouer
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. Mémoriser (pas d'entraînement court pour aller plus vite)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1

            # Entraînement long optimisé tous les 5 jeux (batch processing GPU)
            if agent.n_games % 5 == 0:
                agent.train_long_memory_optimized()

            scores.append(score)
            total_score += score

            if score > record:
                record = score
                agent.model.save()
                print(f"🏆 Partie {agent.n_games}/{max_games} | "
                      f"🎯 NOUVEAU RECORD: {record}")

            # Statistiques tous les 100 parties
            if agent.n_games % 100 == 0:
                mean_100 = sum(scores[-100:]) / min(100, len(scores))
                mean_total = total_score / agent.n_games
                print(f"📊 Partie {agent.n_games}/{max_games} | "
                      f"Score: {score} | "
                      f"Moy(100): {mean_100:.2f} | "
                      f"Moy(total): {mean_total:.2f} | "
                      f"Record: {record}")

    # Statistiques finales
    print("\n" + "=" * 60)
    print("📈 STATISTIQUES FINALES - ULTRA-RAPIDE")
    print("=" * 60)
    print(f"🏆 Meilleur score: {record}")
    print(f"📊 Score moyen global: {total_score / len(scores):.2f}")
    print(f"📊 Score moyen (1000 dernières): {sum(scores[-1000:]) / 1000:.2f}")
    print(f"📊 Score moyen (100 dernières): {sum(scores[-100:]) / 100:.2f}")
    print(f"💾 Modèle sauvegardé: ./model/model.pth")
    print("=" * 60)

    return record, scores

# =============================================================================
# FONCTION D'ENTRAÎNEMENT RAPIDE (10 000 PARTIES)
# =============================================================================

def train_fast(max_games=10000):
    """
    Version rapide pour entraîner sur un grand nombre de parties.
    Désactive l'affichage graphique pour accélérer l'entraînement.

    Args:
        max_games: Nombre maximum de parties à jouer (défaut: 10 000)

    Returns:
        record: Meilleur score atteint
        scores: Liste de tous les scores
    """
    print("=" * 60)
    print(f"🐍 ENTRAÎNEMENT RAPIDE - {max_games} PARTIES 🧠")
    print("=" * 60)
    print(f"GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU utilisé: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 60)
    print("\n⚡ Mode rapide activé - affichage minimal\n")

    scores = []
    total_score = 0
    record = 0
    agent = Agent()

    # Créer le jeu avec affichage
    game = SnakeGameAI(speed=GAME_SPEED, display_enabled=True)

    # Masquer la fenêtre pygame pour accélérer (optionnel)
    # os.environ['SDL_VIDEODRIVER'] = 'dummy'

    while agent.n_games < max_games:
        # 1. Obtenir l'état actuel
        state_old = agent.get_state(game)

        # 2. Obtenir l'action de l'agent
        final_move = agent.get_action(state_old)

        # 3. Jouer le mouvement
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. Entraîner mémoire courte
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Mémoriser l'expérience
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # 6. Entraîner mémoire longue
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            scores.append(score)
            total_score += score

            if score > record:
                record = score
                agent.model.save()
                print(f"🏆 Partie {agent.n_games}/{max_games} | 🎯 NOUVEAU RECORD: {record}")

            # Affichage tous les 100 parties
            if agent.n_games % 100 == 0:
                mean_100 = sum(scores[-100:]) / min(100, len(scores))
                mean_total = total_score / agent.n_games
                print(f"📊 Partie {agent.n_games}/{max_games} | "
                      f"Score: {score} | "
                      f"Moy(100): {mean_100:.2f} | "
                      f"Moy(total): {mean_total:.2f} | "
                      f"Record: {record}")

    # Statistiques finales
    print("\n" + "=" * 60)
    print("📈 STATISTIQUES FINALES - 10 000 PARTIES")
    print("=" * 60)
    print(f"🏆 Meilleur score: {record}")
    print(f"📊 Score moyen global: {total_score / len(scores):.2f}")
    print(f"📊 Score moyen (1000 dernières): {sum(scores[-1000:]) / 1000:.2f}")
    print(f"📊 Score moyen (100 dernières): {sum(scores[-100:]) / 100:.2f}")
    print(f"🎮 Parties jouées: {len(scores)}")
    print(f"💾 Modèle sauvegardé: ./model/model.pth")
    print("=" * 60)

    return record, scores

# =============================================================================
# FONCTION D'ENTRAÎNEMENT PRINCIPALE
# =============================================================================

def train(display_game=False):
    """
    Fonction principale d'entraînement de l'agent.

    Args:
        display_game: Si True, affiche la fenêtre du jeu (plus lent)
                     Si False, entraînement sans affichage (rapide)
    """
    print("=" * 60)
    print("🐍 SNAKE IA - DEEP Q-LEARNING 🧠")
    print("=" * 60)
    print(f"GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU utilisé: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 60)
    if display_game:
        print("\n🎮 Mode avec affichage du jeu (plus lent)")
    else:
        print("\n⚡ Mode sans affichage du jeu (rapide)")
    print("\nDémarrage de l'entraînement...\n")

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(display_enabled=display_game)

    while True:
        # 1. Obtenir l'état actuel
        state_old = agent.get_state(game)

        # 2. Obtenir le mouvement de l'agent (model.predict)
        final_move = agent.get_action(state_old)

        # 3. Effectuer le mouvement et obtenir les résultats
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. Entraîner la mémoire courte (model.train - immédiat)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Remember (stocker l'expérience)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # 6. Entraîner la mémoire longue (model.train - replay)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                print(f"🏆 Nouveau record! Score: {record} | Partie: {agent.n_games}")

            print(f'Partie {agent.n_games} | Score: {score} | Record: {record}')

            # Mettre à jour les graphiques
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Snake IA - Deep Q-Learning Optimisé')
    parser.add_argument('--ultra', action='store_true',
                       help='🚀 Mode ULTRA-RAPIDE avec toutes les optimisations GPU (recommandé)')
    parser.add_argument('--fast', action='store_true',
                       help='⚡ Mode rapide standard pour 10 000 parties')
    parser.add_argument('--games', type=int, default=10000,
                       help='Nombre de parties (défaut: 10000)')
    parser.add_argument('--display', action='store_true',
                       help='🎮 Afficher la fenêtre du jeu (mode normal uniquement)')

    args = parser.parse_args()

    if args.ultra:
        print("🚀 Mode ULTRA-RAPIDE sélectionné (Optimisations GPU)\n")
        record, scores = train_ultra_fast(max_games=args.games)
    elif args.fast:
        print("⚡ Mode rapide sélectionné\n")
        record, scores = train_fast(max_games=args.games)
    else:
        if args.display:
            print("🎮 Mode normal avec affichage du jeu\n")
            train(display_game=True)
        else:
            print("📊 Mode normal sans affichage du jeu (graphiques uniquement)\n")
            train(display_game=False)

