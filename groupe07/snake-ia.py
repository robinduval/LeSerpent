"""
SNAKE IA - Groupe 07

ALGORITHME GÉNÉTIQUE OPTIMISÉ (Neuroévolution)
🔥 200 agents par génération (DOUBLÉ !)
⚡ SÉLECTION STRICTE : Top 20% SEULEMENT ! (les 40 meilleurs)
Reproduction + Mutation ciblée (25% des poids, force 0.4)
500 générations = 100,000 parties jouées ! 🚀

OPTIMISATIONS BASÉES SUR LES MEILLEURES PRATIQUES :
- Population : 200 agents (2x plus d'entraînement par génération)
- Sélection ULTRA-STRICTE : 20% seulement (pression évolutive forte)
- Rewards OPTIMISÉS pour Algo Génétique :
  * +1000 par pomme (signal MASSIF)
  * +bonus exponentiel si >5 pommes (favorise champions)
  * +1 par mouvement (survie)
  * +bonus efficacité (score × 100 / moves)
  * -100 si mort sans manger
- Mutation ciblée : 25% des poids avec force 0.4
- Timeout : 50 moves sans pomme (strict comme sNNake)
- Élitisme strict : Top 5% gardés (10 meilleurs agents)

OBJECTIF : Manger BEAUCOUP de pommes EN PEU de mouvements !

États (11 features) :
- Danger : en face, à droite, à gauche (3)
- Direction : Gauche, Droite, Haut, Bas (4)
- Pomme : Gauche, Droite, Haut, Bas (4)

Réseau de neurones : 11 → 16 → 16 → 3
Actions : Tout droit, Droite, Gauche

TEMPS D'ENTRAÎNEMENT : ~150-200 minutes (100,000 parties !)
Progression sauvegardée tous les 10 générations
Checkpoints : gen10, gen20, ..., gen500
Sauvegarde finale : best_agent_final.npy (MEILLEUR GLOBAL)
"""

import pygame
import random
import time
import numpy as np
import copy

# --- CONSTANTES DE JEU ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 10000  # Très rapide pour l'entraînement (simulation rapide)

# --- CONSTANTES ALGORITHME GÉNÉTIQUE (Optimisées) ---
POPULATION_SIZE = 200  # DOUBLÉ à 200 pour beaucoup plus d'entraînement !
SURVIVORS = 40  # Top 20% (40 sur 200)
MUTATION_RATE = 0.25  # Plus de mutation pour compenser sélection stricte
MUTATION_STRENGTH = 0.4  # Mutations plus fortes

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (en RGB)
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# --- CLASSES DU JEU ---

class Snake:
    """Représente le serpent, sa position, sa direction et son corps."""
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos, 
                     [self.head_pos[0] - 1, self.head_pos[1]], 
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        """Change la direction, empêchant le mouvement inverse immédiat."""
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        """Déplace le serpent d'une case dans la direction actuelle."""
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
        """Prépare le serpent à grandir au prochain mouvement."""
        self.grow_pending = True
        self.score += 1

    def check_self_collision(self):
        """Vérifie si la tête touche une partie du corps."""
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si le jeu est terminé."""
        return self.check_self_collision()

    def draw(self, surface):
        """Dessine le serpent sur la surface de jeu."""
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)

        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)

class Apple:
    """Représente la pomme (nourriture) et sa position."""
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aléatoire non occupée par le serpent."""
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(pos) not in occupied_positions]
        
        if not available_positions:
            return None
            
        return random.choice(available_positions)

    def relocate(self, snake_body):
        """Déplace la pomme vers une nouvelle position aléatoire."""
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False

    def draw(self, surface):
        """Dessine la pomme sur la surface de jeu."""
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)

# --- ALGORITHME GÉNÉTIQUE : RÉSEAU DE NEURONES ---

class NeuralNetwork:
    """Réseau de neurones simple pour l'agent."""
    def __init__(self):
        # Architecture : 11 inputs → 16 hidden → 16 hidden → 3 outputs
        self.weights1 = np.random.randn(11, 16) * 0.5
        self.bias1 = np.random.randn(16) * 0.5
        self.weights2 = np.random.randn(16, 16) * 0.5
        self.bias2 = np.random.randn(16) * 0.5
        self.weights3 = np.random.randn(16, 3) * 0.5
        self.bias3 = np.random.randn(3) * 0.5
    
    def forward(self, inputs):
        """Propage les inputs à travers le réseau."""
        # Layer 1
        hidden1 = np.maximum(0, np.dot(inputs, self.weights1) + self.bias1)  # ReLU
        # Layer 2
        hidden2 = np.maximum(0, np.dot(hidden1, self.weights2) + self.bias2)  # ReLU
        # Output layer
        output = np.dot(hidden2, self.weights3) + self.bias3
        return output
    
    def get_params(self):
        """Retourne tous les paramètres du réseau."""
        return {
            'w1': self.weights1,
            'b1': self.bias1,
            'w2': self.weights2,
            'b2': self.bias2,
            'w3': self.weights3,
            'b3': self.bias3
        }
    
    def set_params(self, params):
        """Charge les paramètres dans le réseau."""
        self.weights1 = params['w1']
        self.bias1 = params['b1']
        self.weights2 = params['w2']
        self.bias2 = params['b2']
        self.weights3 = params['w3']
        self.bias3 = params['b3']

class GeneticAgent:
    """Agent avec réseau de neurones pour l'algorithme génétique."""
    def __init__(self):
        self.network = NeuralNetwork()
        self.fitness = 0
        self.moves = 0
        self.score = 0
    
    def get_state(self, snake, apple):
        """
        Extrait les 11 features de l'état selon les specs :
        - Danger (3) : en face, à droite, à gauche
        - Direction (4) : Gauche, Droite, Haut, Bas
        - Pomme (4) : Gauche, Droite, Haut, Bas
        """
        head_x, head_y = snake.head_pos
        apple_x, apple_y = apple.position
        
        # Direction actuelle
        dir_left = snake.direction == LEFT
        dir_right = snake.direction == RIGHT
        dir_up = snake.direction == UP
        dir_down = snake.direction == DOWN
        
        # Calculer position danger (devant, droite, gauche)
        if dir_up:
            danger_straight = [(head_x, (head_y - 1) % GRID_SIZE)]
            danger_right = [((head_x + 1) % GRID_SIZE, head_y)]
            danger_left = [((head_x - 1) % GRID_SIZE, head_y)]
        elif dir_down:
            danger_straight = [(head_x, (head_y + 1) % GRID_SIZE)]
            danger_right = [((head_x - 1) % GRID_SIZE, head_y)]
            danger_left = [((head_x + 1) % GRID_SIZE, head_y)]
        elif dir_left:
            danger_straight = [((head_x - 1) % GRID_SIZE, head_y)]
            danger_right = [(head_x, (head_y - 1) % GRID_SIZE)]
            danger_left = [(head_x, (head_y + 1) % GRID_SIZE)]
        else:  # dir_right
            danger_straight = [((head_x + 1) % GRID_SIZE, head_y)]
            danger_right = [(head_x, (head_y + 1) % GRID_SIZE)]
            danger_left = [(head_x, (head_y - 1) % GRID_SIZE)]
        
        # Vérifier dangers
        is_danger_straight = list(danger_straight[0]) in snake.body[1:]
        is_danger_right = list(danger_right[0]) in snake.body[1:]
        is_danger_left = list(danger_left[0]) in snake.body[1:]
        
        # Position pomme
        apple_left = apple_x < head_x
        apple_right = apple_x > head_x
        apple_up = apple_y < head_y
        apple_down = apple_y > head_y
        
        # État : 11 features
        state = [
            # Danger (3)
            int(is_danger_straight),
            int(is_danger_right),
            int(is_danger_left),
            # Direction (4)
            int(dir_left),
            int(dir_right),
            int(dir_up),
            int(dir_down),
            # Pomme (4)
            int(apple_left),
            int(apple_right),
            int(apple_up),
            int(apple_down)
        ]
        
        return np.array(state, dtype=float)
    
    def choose_action(self, state, snake):
        """
        Choisit une action basée sur le réseau de neurones.
        Actions : [Tout droit, Droite, Gauche]
        """
        output = self.network.forward(state)
        action_idx = np.argmax(output)
        
        # Convertir action en nouvelle direction
        if action_idx == 0:  # Tout droit
            return snake.direction
        elif action_idx == 1:  # Droite
            if snake.direction == UP:
                return RIGHT
            elif snake.direction == RIGHT:
                return DOWN
            elif snake.direction == DOWN:
                return LEFT
            else:  # LEFT
                return UP
        else:  # Gauche (action_idx == 2)
            if snake.direction == UP:
                return LEFT
            elif snake.direction == LEFT:
                return DOWN
            elif snake.direction == DOWN:
                return RIGHT
            else:  # RIGHT
                return UP

# --- FONCTIONS ALGORITHME GÉNÉTIQUE ---

def create_population(size):
    """Crée une population initiale d'agents."""
    return [GeneticAgent() for _ in range(size)]

def crossover(parent1, parent2):
    """Croisement (reproduction) entre deux parents."""
    child = GeneticAgent()
    
    # Croiser les poids de chaque layer
    for key in parent1.network.get_params().keys():
        if random.random() < 0.5:
            child.network.get_params()[key] = copy.deepcopy(parent1.network.get_params()[key])
        else:
            child.network.get_params()[key] = copy.deepcopy(parent2.network.get_params()[key])
    
    return child

def mutate(agent, mutation_rate=MUTATION_RATE):
    """Applique une mutation aux poids du réseau (mutation ciblée)."""
    params = agent.network.get_params()
    
    for key in params.keys():
        # Muter seulement certains poids aléatoires (plus efficace)
        num_weights = params[key].size
        num_mutations = int(num_weights * mutation_rate)  # 20% des poids
        
        if num_mutations > 0:
            # Sélectionner indices aléatoires
            flat_params = params[key].flatten()
            indices = np.random.choice(num_weights, num_mutations, replace=False)
            # Appliquer mutation
            flat_params[indices] += np.random.randn(num_mutations) * MUTATION_STRENGTH
            params[key] = flat_params.reshape(params[key].shape)
    
    agent.network.set_params(params)
    return agent

def select_best(population, n):
    """Sélectionne les n meilleurs agents basés sur leur fitness."""
    return sorted(population, key=lambda x: x.fitness, reverse=True)[:n]

def evaluate_fitness(agent):
    """Évalue un agent en le faisant jouer une partie."""
    snake = Snake()
    apple = Apple(snake.body)
    game_over = False
    moves = 0
    moves_since_food = 0
    max_moves = 200  # Limite initiale
    max_moves_without_food = 50  # Timeout strict comme dans sNNake (10-30 moves)
    
    while not game_over and moves < max_moves:
        state = agent.get_state(snake, apple)
        action = agent.choose_action(state, snake)
        snake.set_direction(action)
        snake.move()
        moves += 1
        moves_since_food += 1
        
        # Timeout si pas de pomme mangée
        if moves_since_food > max_moves_without_food:
            game_over = True
            break
        
        # Vérifier si pomme mangée
        if snake.head_pos == list(apple.position):
            snake.grow()
            moves_since_food = 0  # Reset timeout
            if not apple.relocate(snake.body):
                break  # Victoire
            max_moves = moves + 200  # Rallonger
        
        # Vérifier game over
        if snake.is_game_over():
            game_over = True
    
    # Calculer fitness OPTIMISÉ pour ALGO GÉNÉTIQUE
    # Problème détecté : 38 pommes = seulement 444 fitness (TROP FAIBLE)
    # Solution : Rewards MASSIFS pour créer différenciation claire
    fitness = 0
    
    # 1. REWARD ÉNORME pour pommes (algo génétique a besoin de gros signaux)
    fitness += snake.score * 1000  # +1000 par pomme (comme TheAILearner: +5000)
    
    # 2. BONUS EXPONENTIEL pour gros scores (favoriser les champions)
    if snake.score > 5:
        bonus = (snake.score - 5) ** 2 * 100  # Bonus quadratique après 5 pommes
        fitness += bonus
    
    # 3. PETIT BONUS pour survie (ne pas dominer les pommes)
    fitness += moves * 1  # +1 par mouvement
    
    # 4. PÉNALITÉ pour mort sans manger
    if game_over and snake.score == 0:
        fitness -= 100  # Pénalité claire
    
    # 5. BONUS efficacité (moins de moves = mieux)
    if snake.score > 0:
        efficiency_bonus = snake.score * 100 / (moves + 1)  # Bonus pour rapidité
        fitness += efficiency_bonus
    
    agent.fitness = max(0, fitness)  # Jamais négatif
    agent.score = snake.score
    agent.moves = moves
    
    return agent

# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface):
    """Dessine la grille pour une meilleure visualisation."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

def display_info(surface, font, snake, generation, best_fitness, avg_fitness):
    """Affiche les stats de l'entraînement."""
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    gen_text = font.render(f"Génération: {generation}", True, BLANC)
    surface.blit(gen_text, (10, 10))
    
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 40))
    
    best_text = font.render(f"Best Fitness: {best_fitness}", True, VERT)
    surface.blit(best_text, (SCREEN_WIDTH - 250, 10))
    
    avg_text = font.render(f"Avg Fitness: {avg_fitness:.0f}", True, ORANGE)
    surface.blit(avg_text, (SCREEN_WIDTH - 250, 40))

# --- ENTRAÎNEMENT GÉNÉTIQUE ---

def train_genetic():
    """Entraînement par algorithme génétique (optimisé)."""
    max_generations = 500  # 500 générations pour apprentissage MASSIF
    
    print("╔═════════════════════════════════════════════╗")
    print("║  ENTRAÎNEMENT GÉNÉTIQUE OPTIMISÉ           ║")
    print("║     (REWARDS MASSIFS - FIX DÉTECTÉ)        ║")
    print("╚═════════════════════════════════════════════╝")
    print(f"🔥 Population: {POPULATION_SIZE} agents/génération")
    print(f"🔄 Générations: {max_generations} (= {POPULATION_SIZE * max_generations:,} parties !)")
    print(f"⚡ Survivors: {SURVIVORS} (TOP 20% - sélection STRICTE)")
    print(f"🔀 Mutation rate: {MUTATION_RATE}")
    print(f"💪 Mutation strength: {MUTATION_STRENGTH}")
    print(f"⏱️  Temps estimé: 150-200 minutes")
    print(f"\n🎯 FITNESS OPTIMISÉE (38 pommes devrait = 40k+ fitness):")
    print(f"   ✅ +1000 par pomme (signal ÉNORME)")
    print(f"   🚀 +bonus exponentiel si >5 pommes")
    print(f"   🏃 +1 par mouvement (survie)")
    print(f"   ⚡ +bonus efficacité (score × 100 / moves)")
    print(f"   ❌ -100 si mort sans manger")
    print(f"\n🎮 OBJECTIF : Manger BEAUCOUP en peu de mouvements !\n")
    
    # Créer population initiale
    population = create_population(POPULATION_SIZE)
    generation = 0
    
    best_scores_history = []
    
    # CORRECTION : Garder le MEILLEUR agent GLOBAL de toutes les générations
    global_best_agent = None
    global_best_fitness = -float('inf')
    
    while generation < max_generations:
        generation += 1
        print(f"{'='*50}")
        print(f"🔄 GÉNÉRATION {generation}/{max_generations}")
        print(f"{'='*50}")
        
        # Évaluer tous les agents (avec barre de progression)
        for i, agent in enumerate(population):
            evaluate_fitness(agent)
            if (i + 1) % 40 == 0 or (i + 1) == POPULATION_SIZE:
                progress = (i + 1) / POPULATION_SIZE * 100
                print(f"  Évaluation: [{i+1}/{POPULATION_SIZE}] {progress:.0f}%", end='\r')
        print()  # Nouvelle ligne
        
        # Statistiques
        generation_best = max(population, key=lambda x: x.fitness)
        avg_fitness = sum(a.fitness for a in population) / len(population)
        avg_score = sum(a.score for a in population) / len(population)
        best_scores_history.append(generation_best.score)
        
        # MISE À JOUR DU MEILLEUR AGENT GLOBAL
        if generation_best.fitness > global_best_fitness:
            global_best_fitness = generation_best.fitness
            global_best_agent = copy.deepcopy(generation_best)
            print(f"\n🏆 NOUVEAU RECORD GLOBAL ! Fitness: {global_best_fitness:.0f}, Score: {global_best_agent.score}\n")
        
        # Calculer efficacité (de la génération)
        if generation_best.score > 0 and generation_best.moves > 0:
            efficiency = generation_best.score / generation_best.moves
            moves_per_apple = generation_best.moves / generation_best.score
        else:
            efficiency = 0
            moves_per_apple = 0
        
        print(f"📊 RÉSULTATS GÉNÉRATION:")
        print(f"  🥇 Meilleur score: {generation_best.score} pommes")
        print(f"  🏃 Meilleur moves: {generation_best.moves}")
        print(f"  ⚡ Efficacité: {efficiency:.4f} ({moves_per_apple:.1f} moves/pomme)")
        print(f"  ⭐ Best fitness: {generation_best.fitness:.0f}")
        print(f"  📈 Fitness moyenne: {avg_fitness:.0f}")
        print(f"  🍎 Score moyen: {avg_score:.2f}")
        print(f"  🎖️  RECORD GLOBAL: {global_best_agent.score if global_best_agent else 0} pommes")
        
        # Sélectionner les meilleurs
        survivors = select_best(population, SURVIVORS)
        
        # Créer nouvelle génération
        new_population = []
        
        # Élitisme strict : garder les meilleurs (5%)
        elite_count = max(1, POPULATION_SIZE // 20)  # Top 5% = 10 agents
        new_population.extend(survivors[:elite_count])
        
        # Créer le reste par reproduction
        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
        
        # Sauvegarder le MEILLEUR GLOBAL tous les 10 générations
        if generation % 10 == 0 and global_best_agent:
            filename = f'best_agent_gen{generation}.npy'
            np.save(filename, global_best_agent.network.get_params())
            print(f"\n💾 Sauvegarde MEILLEUR GLOBAL: {filename} (score: {global_best_agent.score})")
        
        # Afficher un résumé tous les 50 générations
        if generation % 50 == 0:
            progress_pct = (generation / max_generations) * 100
            print(f"\n{'🎯'*20}")
            print(f"PROGRESSION : {generation}/{max_generations} ({progress_pct:.0f}%)")
            print(f"Meilleur score jusqu'ici: {max(best_scores_history)} pommes")
            print(f"{'🎯'*20}\n")
    
    print(f"\n{'='*50}")
    print("✅ ENTRAÎNEMENT TERMINÉ !")
    print(f"{'='*50}")
    print(f"🏆 Meilleur score GLOBAL: {global_best_agent.score if global_best_agent else 0} pommes")
    print(f"🎯 Fitness GLOBALE: {global_best_fitness:.0f}")
    print(f"📈 Progression: {best_scores_history[0]} → {best_scores_history[-1]}")
    print(f"\n💾 Sauvegarde finale du MEILLEUR GLOBAL...")
    
    # Sauvegarder le meilleur agent GLOBAL final
    if global_best_agent:
        np.save('best_agent_final.npy', global_best_agent.network.get_params())
        print(f"✅ Sauvegarde terminée : best_agent_final.npy")
    
    return global_best_agent

# --- VISUALISATION ---

def visualize_best(agent):
    """Visualise le meilleur agent en train de jouer."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake IA - Meilleur Agent")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)
    
    running = True
    while running:
        snake = Snake()
        apple = Apple(snake.body)
        game_over = False
        
        while not game_over and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    break  # Recommencer
            
            state = agent.get_state(snake, apple)
            action = agent.choose_action(state, snake)
            snake.set_direction(action)
            snake.move()
            
            if snake.head_pos == list(apple.position):
                snake.grow()
                if not apple.relocate(snake.body):
                    break
            
            if snake.is_game_over():
                game_over = True
            
            # Dessin
            screen.fill(GRIS_FOND)
            pygame.draw.rect(screen, NOIR, (0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH))
            draw_grid(screen)
            apple.draw(screen)
            snake.draw(screen)
            display_info(screen, font, snake, "Démo", agent.fitness, 0)
            
            pygame.display.flip()
            clock.tick(12)  # Vitesse normale
        
        if game_over:
            time.sleep(1)
    
    pygame.quit()

# --- MAIN ---

def main():
    """Point d'entrée principal."""
    print("SNAKE IA - Algorithme Génétique")
    print("1. Entraîner")
    print("2. Visualiser le meilleur")
    choice = input("Choix (1/2) : ")
    
    if choice == "1":
        best_agent = train_genetic()
        print("\nVoulez-vous visualiser le meilleur agent ? (o/n)")
        if input().lower() == 'o':
            visualize_best(best_agent)
    elif choice == "2":
        # Charger le meilleur agent
        try:
            agent = GeneticAgent()
            import glob
            import os
            
            # Priorité 1 : Chercher best_agent_final.npy (MEILLEUR GLOBAL)
            if os.path.exists('best_agent_final.npy'):
                print("Chargement du MEILLEUR GLOBAL : best_agent_final.npy")
                params = np.load('best_agent_final.npy', allow_pickle=True).item()
                agent.network.set_params(params)
                visualize_best(agent)
            # Priorité 2 : Chercher le dernier checkpoint
            else:
                files = glob.glob('best_agent_gen*.npy')
                if files:
                    latest = max(files, key=lambda x: int(x.split('gen')[1].split('.')[0]))
                    print(f"Chargement de {latest}...")
                    params = np.load(latest, allow_pickle=True).item()
                    agent.network.set_params(params)
                    visualize_best(agent)
                else:
                    print("Aucun agent sauvegardé trouvé. Veuillez d'abord entraîner.")
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            print("Veuillez d'abord entraîner.")

if __name__ == '__main__':
    main()
