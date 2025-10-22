import pygame
import random
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(None, 25)

# --- CONSTANTES DE JEU ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 1000  # Très rapide pour l'entraînement

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 60
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
BLEU1 = (0, 0, 255)
BLEU2 = (0, 100, 255)

# Directions
Direction = namedtuple('Direction', 'x, y')
UP = Direction(0, -1)
DOWN = Direction(0, 1)
LEFT = Direction(-1, 0)
RIGHT = Direction(1, 0)


class SnakeGameAI:
    """Version du jeu Snake adaptée pour l'IA avec Deep Q-Learning"""
    
    def __init__(self, w=SCREEN_WIDTH, h=SCREEN_HEIGHT, display=True):
        self.w = w
        self.h = h
        self.display_enabled = display
        
        # Initialisation de l'affichage
        if self.display_enabled:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI - Deep Q-Learning')
            self.clock = pygame.time.Clock()
        
        self.reset()
    
    def reset(self):
        """Réinitialise le jeu"""
        # Direction initiale
        self.direction = RIGHT
        
        # Position initiale du serpent
        self.head = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.snake = [self.head,
                      [self.head[0] - 1, self.head[1]],
                      [self.head[0] - 2, self.head[1]]]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
    
    def _place_food(self):
        """Place la nourriture à une position aléatoire"""
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        self.food = [x, y]
        
        # S'assurer que la nourriture n'est pas sur le serpent
        if self.food in self.snake:
            self._place_food()
    
    def play_step(self, action):
        """
        Exécute une étape de jeu avec l'action donnée
        
        Args:
            action: [straight, right, left] - un seul est 1, les autres 0
        
        Returns:
            reward: récompense pour cette action
            game_over: booléen indiquant si le jeu est terminé
            score: score actuel
        """
        self.frame_iteration += 1
        
        # 1. Collecter les entrées utilisateur (pour quitter)
        if self.display_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # 2. Bouger le serpent
        self._move(action)  # Met à jour la tête
        self.snake.insert(0, self.head)
        
        # 3. Vérifier si game over
        reward = 0
        game_over = False
        
        # Game over si collision ou trop de mouvements sans manger
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 4. Placer nouvelle nourriture ou juste bouger
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = 0.1  # Petite récompense pour rester en vie
        
        # 5. Mise à jour de l'interface et l'horloge
        if self.display_enabled:
            self._update_ui()
            self.clock.tick(GAME_SPEED)
        
        # 6. Retourner game over et score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        """Vérifie les collisions avec les murs ou le corps"""
        if pt is None:
            pt = self.head
        
        # Collision avec les bords
        if pt[0] < 0 or pt[0] >= GRID_SIZE or pt[1] < 0 or pt[1] >= GRID_SIZE:
            return True
        
        # Collision avec soi-même
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def _update_ui(self):
        """Met à jour l'affichage graphique"""
        self.display.fill(GRIS_FOND)
        
        # Zone de jeu
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.display, NOIR, game_area_rect)
        
        # Dessiner la grille
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.display, (80, 80, 80), (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
        for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.display, (80, 80, 80), (0, y), (SCREEN_WIDTH, y))
        
        # Dessiner le serpent
        for i, pt in enumerate(self.snake):
            if i == 0:  # Tête
                rect = pygame.Rect(pt[0] * CELL_SIZE, pt[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, 
                                   CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.display, ORANGE, rect)
                pygame.draw.rect(self.display, NOIR, rect, 2)
            else:  # Corps
                rect = pygame.Rect(pt[0] * CELL_SIZE, pt[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, 
                                   CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.display, VERT, rect)
                pygame.draw.rect(self.display, NOIR, rect, 1)
        
        # Dessiner la nourriture
        food_rect = pygame.Rect(self.food[0] * CELL_SIZE, 
                                self.food[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, 
                                CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.display, ROUGE, food_rect, border_radius=5)
        pygame.draw.circle(self.display, BLANC, 
                          (food_rect.x + CELL_SIZE * 0.7, food_rect.y + CELL_SIZE * 0.3), 
                          CELL_SIZE // 8)
        
        # Afficher le score
        text = font.render(f"Score: {self.score}  |  Generation: {self.frame_iteration}", True, BLANC)
        self.display.blit(text, [10, 20])
        pygame.display.flip()
    
    def _move(self, action):
        """
        Met à jour la direction en fonction de l'action
        action: [straight, right, left]
        """
        clock_wise = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):  # Tout droit
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Tourner à droite
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1] Tourner à gauche
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
        
        # Calculer la nouvelle position de la tête
        x = self.head[0]
        y = self.head[1]
        
        x += self.direction.x
        y += self.direction.y
        
        self.head = [x, y]
