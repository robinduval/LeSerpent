"""
Deep Q-Learning Snake AI
A complete implementation of DQN for solving the Snake game.

This file contains:
- SnakeGameAI: Adapted game for RL training
- Linear_QNet: Deep Q-Network model
- QTrainer: Training component
- Agent: Main RL agent with experience replay
- Helper functions and main entry point

Author: GitHub Copilot + TCHEKACHEV David, SUZOR Antonin
"""

import pygame
import random
import numpy as np
from collections import deque
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os

# ==================== CONSTANTS & CONFIGURATION ====================

# Game constants
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 100  # Faster for training (increased from 40)

# Screen dimensions
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)  # Snake head
GREEN = (0, 200, 0)     # Snake body
RED = (200, 0, 0)       # Apple
GRAY = (50, 50, 50)
LIGHT_GRAY = (80, 80, 80)

# Directions
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

# RL Hyperparameters (Simplified)
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # Learning rate
GAMMA = 0.9
HIDDEN_SIZE = 256  # Single hidden layer

# ==================== SNAKE GAME FOR AI ====================

class SnakeGameAI:
    """
    Snake game adapted for reinforcement learning.
    Features:
    - play_step(action) method for RL training
    - Reward calculation
    - Optional headless mode for faster training
    - Wrap-around walls (toroidal topology)
    """
    
    def __init__(self, w=GRID_SIZE, h=GRID_SIZE, display=True):
        """
        Initialize the game.
        
        Args:
            w: Grid width
            h: Grid height
            display: Whether to show pygame window
        """
        self.w = w
        self.h = h
        self.display_enabled = display
        
        # Initialize pygame
        if self.display_enabled:
            pygame.init()
            # Try to set window as always on top (platform-specific)
            try:
                # For Linux/X11
                import os
                os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'
            except:
                pass
            
            self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Snake AI - Deep Q-Learning')
            
            # Bring window to front
            try:
                import ctypes
                import platform
                if platform.system() == 'Windows':
                    # Windows: Set window always on top
                    hwnd = pygame.display.get_wm_info()['window']
                    ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
            except:
                pass  # Not critical if it fails
            
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        self.reset()
        
    def reset(self):
        """Reset game to initial state."""
        # Initial snake position (starting from left side)
        self.direction = Direction.RIGHT
        self.head = [self.w // 4, self.h // 2]
        self.snake = [
            self.head,
            [self.head[0] - 1, self.head[1]],
            [self.head[0] - 2, self.head[1]]
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        """Place food at random unoccupied position."""
        while True:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            self.food = [x, y]
            if self.food not in self.snake:
                break
                
    def play_step(self, action):
        """
        Execute one game step with given action.
        
        Args:
            action: [straight, right, left] - one-hot encoded
            
        Returns:
            reward: float - reward for this step
            game_over: bool - whether game ended
            score: int - current score
        """
        self.frame_iteration += 1
        
        # Handle pygame events (for display)
        if self.display_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # Move snake
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Check game over conditions
        reward = 0
        game_over = False
        
        # Timeout condition (prevent infinite loops)
        if self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -20  # Increased penalty
            return reward, game_over, self.score
            
        # Check self-collision
        if self._is_collision():
            game_over = True
            reward = -20  # Increased penalty (was -10)
            return reward, game_over, self.score
            
        # Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            
            # Check victory (grid full)
            if len(self.snake) == self.w * self.h:
                game_over = True
                reward = 100
                return reward, game_over, self.score
        else:
            # Remove tail if no food eaten
            if len(self.snake) > 0:  # Safety check
                self.snake.pop()
            # Survival reward (increased from 0.1)
            reward = 0.5
            
        # Update display
        if self.display_enabled:
            self._update_ui()
            self.clock.tick(GAME_SPEED)
            
        return reward, game_over, self.score
        
    def _move(self, action):
        """
        Convert action to direction and move snake.
        action = [straight, right, left]
        """
        # Current direction
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        # Determine new direction based on action
        if np.array_equal(action, [1, 0, 0]):  # Straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Right turn
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # [0, 0, 1] Left turn
            new_dir = clock_wise[(idx - 1) % 4]
            
        self.direction = new_dir
        
        # Move head in new direction (with wrap-around)
        x = self.head[0]
        y = self.head[1]
        dx, dy = self.direction.value
        x = (x + dx) % self.w
        y = (y + dy) % self.h
        self.head = [x, y]
        
    def _is_collision(self, point=None):
        """
        Check if point collides with snake body.
        Walls wrap around, so only self-collision matters.
        """
        if point is None:
            point = self.head
        # Check if head hits body (excluding head itself)
        if point in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        """Draw game state."""
        # Background
        self.display.fill(GRAY)
        
        # Game area
        game_area = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.display, BLACK, game_area)
        
        # Draw grid
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.display, LIGHT_GRAY, (x, SCORE_PANEL_HEIGHT), 
                           (x, SCREEN_HEIGHT))
        for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.display, LIGHT_GRAY, (0, y), (SCREEN_WIDTH, y))
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            x = segment[0] * CELL_SIZE
            y = segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT
            if i == 0:  # Head
                pygame.draw.rect(self.display, ORANGE, 
                               pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.display, BLACK, 
                               pygame.Rect(x, y, CELL_SIZE, CELL_SIZE), 2)
            else:  # Body
                pygame.draw.rect(self.display, GREEN, 
                               pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.display, BLACK, 
                               pygame.Rect(x, y, CELL_SIZE, CELL_SIZE), 1)
        
        # Draw food
        food_x = self.food[0] * CELL_SIZE
        food_y = self.food[1] * CELL_SIZE + SCORE_PANEL_HEIGHT
        pygame.draw.rect(self.display, RED, 
                        pygame.Rect(food_x, food_y, CELL_SIZE, CELL_SIZE), 
                        border_radius=5)
        pygame.draw.circle(self.display, WHITE, 
                          (food_x + int(CELL_SIZE * 0.7), 
                           food_y + int(CELL_SIZE * 0.3)), 
                          CELL_SIZE // 8)
        
        # Score panel
        pygame.draw.rect(self.display, GRAY, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
        pygame.draw.line(self.display, WHITE, (0, SCORE_PANEL_HEIGHT - 2), 
                        (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)
        
        # Display score
        text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, (10, 25))
        
        # Display AI indicator
        ai_text = self.font.render("AI Mode", True, ORANGE)
        self.display.blit(ai_text, (SCREEN_WIDTH - 150, 25))
        
        pygame.display.flip()

# ==================== NEURAL NETWORK MODEL ====================

class Linear_QNet(nn.Module):
    """
    Simple Q-Network with single hidden layer.
    Architecture: Input -> Hidden (ReLU) -> Output
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the network.
        
        Args:
            input_size: Size of state vector
            hidden_size: Number of neurons in hidden layer (256)
            output_size: Number of actions (3)
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
        
    def save(self, file_name='model.pth'):
        """Save model weights to file."""
        torch.save(self.state_dict(), file_name)
        print(f"Model saved to {file_name}")
        
    def load(self, file_name='model.pth'):
        """Load model weights from file."""
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()
            print(f"Model loaded from {file_name}")
        else:
            print(f"No model found at {file_name}")

# ==================== TRAINING COMPONENT ====================

class QTrainer:
    """
    Trainer for the Q-Network.
    Implements the Q-learning update rule with experience replay.
    """
    
    def __init__(self, model, lr, gamma):
        """
        Initialize trainer.
        
        Args:
            model: Linear_QNet instance
            lr: Learning rate
            gamma: Discount factor
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        """
        Perform one training step.
        
        Args:
            state: Current state(s)
            action: Action(s) taken
            reward: Reward(s) received
            next_state: Next state(s)
            done: Done flag(s)
        """
        # Convert to tensors
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        
        # Handle both single and batch inputs
        if len(state.shape) == 1:
            # Single input - reshape to (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
        # 1. Predicted Q values with current state
        pred = self.model(state)
        
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Bellman equation: Q_new = r + γ * max(Q_next)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                
            # Update Q value for the action taken
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        # 2. Calculate loss
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        
        # 3. Backpropagation
        loss.backward()
        self.optimizer.step()

# ==================== AGENT ====================

class Agent:
    """
    Deep Q-Learning Agent.
    Manages the training process, memory, and decision making.
    """
    
    def __init__(self):
        """Initialize agent with model, memory, and hyperparameters."""
        self.n_games = 0
        self.epsilon = 0  # Exploration rate
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience replay memory
        
        # Model: Simple architecture with 1 hidden layer
        self.model = Linear_QNet(16, HIDDEN_SIZE, 3)  # 16 features -> 256 -> 3 actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    def get_state(self, game):
        """
        Extract enhanced state features from game.
        
        Returns 16 features (improved from 11):
        - Danger straight, right, left (3)
        - Current direction (4)
        - Food location relative to head (4)
        - Body proximity in 4 directions (4)
        - Tail is nearby (1)
        
        Args:
            game: SnakeGameAI instance
            
        Returns:
            numpy array of 16 features
        """
        # Safety check: ensure snake exists
        if not game.snake or len(game.snake) == 0:
            return np.zeros(16, dtype=int)
            
        head = game.snake[0]
        
        # Points around the head
        point_l = [(head[0] - 1) % game.w, head[1]]
        point_r = [(head[0] + 1) % game.w, head[1]]
        point_u = [head[0], (head[1] - 1) % game.h]
        point_d = [head[0], (head[1] + 1) % game.h]
        
        # Points 2 steps away (for better planning)
        point_ll = [(head[0] - 2) % game.w, head[1]]
        point_rr = [(head[0] + 2) % game.w, head[1]]
        point_uu = [head[0], (head[1] - 2) % game.h]
        point_dd = [head[0], (head[1] + 2) % game.h]
        
        # Current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        # Helper function to check if body is near
        def body_proximity(point):
            """Check if body segment is at or near this point"""
            if point in game.snake[1:]:
                return 1
            # Check neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = [(point[0] + dx) % game.w, (point[1] + dy) % game.h]
                if neighbor in game.snake[1:]:
                    return 1
            return 0
        
        # Danger detection (immediate collision)
        state = [
            # Danger straight (immediate)
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),
            
            # Danger right
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d)),
            
            # Danger left
            (dir_d and game._is_collision(point_r)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d)),
            
            # Current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location (considering wrap-around for shortest path)
            game.food[0] < game.head[0],  # Food left
            game.food[0] > game.head[0],  # Food right
            game.food[1] < game.head[1],  # Food up
            game.food[1] > game.head[1],  # Food down
            
            # NEW: Body proximity in each direction
            body_proximity(point_l),  # Body nearby on left
            body_proximity(point_r),  # Body nearby on right
            body_proximity(point_u),  # Body nearby above
            body_proximity(point_d),  # Body nearby below
            
            # NEW: Tail proximity (helps avoid self-trap)
            len(game.snake) > 3 and (
                game.snake[-1] in [point_l, point_r, point_u, point_d] or
                game.snake[-2] in [point_l, point_r, point_u, point_d]
            )
        ]
        
        return np.array(state, dtype=int)
        
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory.
        
        Args:
            state, action, reward, next_state, done: Experience tuple
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def train_long_memory(self):
        """
        Train on batch from memory (experience replay).
        Samples random batch for better learning.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train on single recent experience.
        
        Args:
            state, action, reward, next_state, done: Single experience
        """
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            action: [straight, right, left] one-hot encoded
        """
        # Exploration vs Exploitation tradeoff
        # Improved epsilon decay: faster convergence, stops at 10
        self.epsilon = max(10, 100 - self.n_games)
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            # Random move (exploration)
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Model prediction (exploitation)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move

# ==================== HELPER FUNCTIONS ====================

# ==================== HELPER FUNCTIONS ====================

def print_training_stats(game_num, score, record, mean_score, last_n_scores=None):
    """
    Print training statistics to console.
    
    Args:
        game_num: Current game number
        score: Score of current game
        record: Best score so far
        mean_score: Average score across all games
        last_n_scores: List of recent scores for trend analysis
    """
    # Print current game stats
    print(f'Game {game_num:4d} | Score: {score:3d} | '
          f'Record: {record:3d} | Mean: {mean_score:6.2f}', end='')
    
    # Calculate and display trend from last 10 games
    if last_n_scores and len(last_n_scores) >= 10:
        recent_mean = sum(last_n_scores[-10:]) / 10
        print(f' | Last 10: {recent_mean:5.2f}', end='')
    
    print()  # New line
    
    # Print milestone messages
    if game_num % 50 == 0:
        print(f"\n{'='*60}")
        print(f"  Milestone: {game_num} games completed!")
        print(f"  Current Record: {record}")
        print(f"  Overall Mean Score: {mean_score:.2f}")
        if last_n_scores and len(last_n_scores) >= 50:
            recent_50_mean = sum(last_n_scores[-50:]) / 50
            print(f"  Last 50 Games Mean: {recent_50_mean:.2f}")
        print(f"{'='*60}\n")

# ==================== MAIN FUNCTIONS ====================

def train():
    """
    Training mode - train a new model from scratch.
    Runs indefinitely until manually stopped (Ctrl+C).
    """
    print("=" * 60)
    print("Starting Training Mode")
    print("=" * 60)
    print(f"Hyperparameters:")
    print(f"  - Learning Rate: {LR}")
    print(f"  - Gamma: {GAMMA}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Memory Size: {MAX_MEMORY}")
    print(f"  - Hidden Layer: {HIDDEN_SIZE} neurons (single layer)")
    print("=" * 60)
    
    # Check for existing model
    model_path = 'model.pth'
    existing_model = os.path.exists(model_path)
    
    if existing_model:
        print("\n✓ Existing model found! Continuing training from saved model.")
    else:
        print("\n○ No existing model. Starting fresh training.")
    
    print("\nPress Ctrl+C to stop training\n")
    
    # Training statistics
    scores = []
    total_score = 0
    record = 0
    agent = Agent()
    
    # Load existing model if available
    if existing_model:
        agent.model.load()
        print("Model weights loaded successfully!\n")
    
    game = SnakeGameAI(display=True)  # Show display during training
    
    try:
        while True:
            # Get current state
            state_old = agent.get_state(game)
            
            # Get action based on current state
            final_move = agent.get_action(state_old)
            
            # Perform action and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            
            # Train short memory (immediate learning)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            
            # Remember the experience
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                # Train long memory (experience replay)
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                
                # Save model if new record
                if score > record:
                    record = score
                    agent.model.save()
                    
                # Update statistics
                scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                
                # Print progress with statistics
                print_training_stats(agent.n_games, score, record, mean_score, scores)
                
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Training interrupted by user")
        print("="*60)
        print(f"Total games played: {agent.n_games}")
        print(f"Best score: {record}")
        print(f"Final mean score: {mean_score:.2f}")
        if len(scores) >= 10:
            recent_mean = sum(scores[-10:]) / len(scores[-10:])
            print(f"Last 10 games mean: {recent_mean:.2f}")
        if len(scores) >= 50:
            recent_mean = sum(scores[-50:]) / len(scores[-50:])
            print(f"Last 50 games mean: {recent_mean:.2f}")
        print("="*60)


def play():
    """
    Inference mode - load trained model and play.
    Displays the AI playing the game.
    """
    print("=" * 60)
    print("Starting Inference Mode")
    print("=" * 60)
    
    agent = Agent()
    
    # Try to load trained model
    model_path = 'model.pth'
    if os.path.exists(model_path):
        agent.model.load()
    else:
        print(f"\n⚠️  No trained model found at {model_path}")
        print("Please train the model first using: python snake-ia.py --train")
        return
        
    # Disable exploration (pure exploitation)
    agent.epsilon = 0
    
    print("\nAI is now playing! Close the window or press Ctrl+C to stop.\n")
    
    game = SnakeGameAI(display=True)  # Show display
    game_count = 0
    total_score = 0
    
    try:
        while True:
            # Get state
            state = agent.get_state(game)
            
            # Get action from model (no randomness)
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = agent.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[move] = 1
            
            # Play step
            reward, done, score = game.play_step(final_move)
            
            if done:
                game_count += 1
                total_score += score
                avg_score = total_score / game_count
                print(f'Game {game_count} | Score: {score} | Average: {avg_score:.2f}')
                game.reset()
                
    except KeyboardInterrupt:
        print(f"\n\nInference stopped")
        print(f"Games played: {game_count}")
        if game_count > 0:
            print(f"Average score: {total_score / game_count:.2f}")
        pygame.quit()


def main():
    """
    Main entry point with argument parsing.
    Supports three modes:
    - --train: Training mode
    - --play: Inference mode
    - Auto-detect: Play if model exists, otherwise train
    """
    parser = argparse.ArgumentParser(
        description='Snake AI using Deep Q-Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python snake-ia.py --train    # Train a new model
  python snake-ia.py --play     # Play with trained model
  python snake-ia.py            # Auto-detect mode
        """
    )
    parser.add_argument('--train', action='store_true', 
                       help='Training mode - train a new model from scratch')
    parser.add_argument('--play', action='store_true', 
                       help='Inference mode - play with trained model')
    
    args = parser.parse_args()
    
    if args.train:
        train()
    elif args.play:
        play()
    else:
        # Auto-detect mode
        model_path = 'model.pth'
        if os.path.exists(model_path):
            print("Trained model found. Starting inference mode...")
            print("(Use --train to train a new model)\n")
            play()
        else:
            print("No trained model found. Starting training mode...")
            print("(Use --play to play with a trained model)\n")
            train()


if __name__ == '__main__':
    main()
