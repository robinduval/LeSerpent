# -*- coding: utf-8 -*-
"""Deep Q-Learning snake agent with PyGame environment and Torch model."""

from __future__ import annotations

import csv
import random
from collections import deque, namedtuple
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pygame
import torch
from torch import nn, optim

Point = namedtuple("Point", "x y")

# Directions
UP: Tuple[int, int] = (0, -1)
DOWN: Tuple[int, int] = (0, 1)
LEFT: Tuple[int, int] = (-1, 0)
RIGHT: Tuple[int, int] = (1, 0)

ACTIONS = (UP, DOWN, LEFT, RIGHT)

DEFAULT_CHECKPOINT_PATH = Path(__file__).with_name("snake_checkpoint.pth")
DEFAULT_LOG_PATH = Path(__file__).with_name("snake_training_log.csv")


def rotate_right(direction: Tuple[int, int]) -> Tuple[int, int]:
    """Rotate a grid direction clockwise."""
    return direction[1], -direction[0]


def rotate_left(direction: Tuple[int, int]) -> Tuple[int, int]:
    """Rotate a grid direction counter clockwise."""
    return -direction[1], direction[0]


@dataclass(frozen=True)
class GameConfig:
    grid_size: int = 15
    cell_size: int = 30
    game_speed: int = 1000  # frames per second when rendering
    border: int = 20
    reward_food: float = 10.0
    reward_move: float = -0.1
    reward_death: float = -100.0
    reward_win: float = 100.0
    frame_limit_factor: int = 100


class SnakeGame:
    """PyGame environment adapted for reinforcement learning."""

    def __init__(self, config: GameConfig | None = None, render: bool = False) -> None:
        self.config = config or GameConfig()
        self.render_enabled = render
        self.display: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.winning_length = self.config.grid_size * self.config.grid_size
        self.direction: Tuple[int, int] = RIGHT
        self.snake: list[Point] = []
        self.head: Point | None = None
        self.food: Point | None = None
        self.score: int = 0
        self.frame_iteration: int = 0
        if self.render_enabled:
            pygame.init()
            width = self.config.grid_size * self.config.cell_size
            height = width + self.config.border * 2
            self.display = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Snake AI - Groupe 10")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 24)
        self.reset()

    def reset(self) -> None:
        center_x = self.config.grid_size // 2
        center_y = self.config.grid_size // 2
        self.direction = RIGHT
        self.snake = [
            Point(center_x, center_y),
            Point(center_x - 1, center_y),
            Point(center_x - 2, center_y),
        ]
        self.head = self.snake[0]
        self.score = 0
        self.frame_iteration = 0
        self.food = None
        self._place_food()
        if self.render_enabled:
            self._update_ui()

    def _place_food(self) -> None:
        available = {
            (x, y)
            for x in range(self.config.grid_size)
            for y in range(self.config.grid_size)
        }
        occupied = {(segment.x, segment.y) for segment in self.snake}
        free_positions = list(available - occupied)
        if free_positions:
            x, y = random.choice(free_positions)
            self.food = Point(x, y)
        else:
            self.food = None

    def play_step(self, action: int | Sequence[int]) -> Tuple[float, bool, int]:
        """Execute one action step and return reward, game_over flag, and score."""
        if self.render_enabled:
            self._handle_events()
        self.frame_iteration += 1
        reward = self.config.reward_move
        done = False
        self._move(action)
        self.snake.insert(0, self.head)
        if self._is_collision(self.head):
            reward = self.config.reward_death
            done = True
        elif self.food and self.head == self.food:
            reward += self.config.reward_food
            self.score += 1
            self._place_food()
            if len(self.snake) == self.winning_length:
                reward += self.config.reward_win
                done = True
        else:
            self.snake.pop()
        if not done and self.frame_iteration > self.config.frame_limit_factor * len(self.snake):
            reward = self.config.reward_death
            done = True
        if done:
            if self.render_enabled:
                self._update_ui()
            return reward, True, self.score
        if self.render_enabled:
            self._update_ui()
            assert self.clock is not None
            self.clock.tick(self.config.game_speed)
        return reward, False, self.score

    def _move(self, action: int | Sequence[int]) -> None:
        if isinstance(action, Iterable) and not isinstance(action, (str, bytes)):
            action_index = int(np.argmax(np.asarray(action, dtype=np.float32)))
        else:
            action_index = int(action)
        action_index = max(0, min(action_index, len(ACTIONS) - 1))
        new_direction = ACTIONS[action_index]
        if self.snake and len(self.snake) > 1:
            opposite = (-self.direction[0], -self.direction[1])
            if new_direction == opposite:
                new_direction = self.direction
        self.direction = new_direction
        if not self.head:
            raise RuntimeError("Snake head is undefined.")
        new_head = Point(self.head.x + self.direction[0], self.head.y + self.direction[1])
        self.head = new_head

    def _is_collision(self, point: Point | None = None) -> bool:
        if point is None:
            point = self.head
        if point is None:
            return True
        if (
            point.x < 0
            or point.x >= self.config.grid_size
            or point.y < 0
            or point.y >= self.config.grid_size
        ):
            return True
        if point in self.snake[1:]:
            return True
        return False

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def _update_ui(self) -> None:
        if not self.render_enabled or not self.display:
            return
        width = self.config.grid_size * self.config.cell_size
        border = self.config.border
        self.display.fill((30, 30, 30))
        pygame.draw.rect(
            self.display,
            (0, 0, 0),
            pygame.Rect(0, border, width, width),
        )
        for x in range(self.config.grid_size):
            for y in range(self.config.grid_size):
                rect = pygame.Rect(
                    x * self.config.cell_size,
                    border + y * self.config.cell_size,
                    self.config.cell_size,
                    self.config.cell_size,
                )
                pygame.draw.rect(self.display, (45, 45, 45), rect, 1)
        if self.food:
            rect = pygame.Rect(
                self.food.x * self.config.cell_size,
                border + self.food.y * self.config.cell_size,
                self.config.cell_size,
                self.config.cell_size,
            )
            pygame.draw.rect(self.display, (200, 0, 0), rect)
        for index, segment in enumerate(self.snake):
            rect = pygame.Rect(
                segment.x * self.config.cell_size,
                border + segment.y * self.config.cell_size,
                self.config.cell_size,
                self.config.cell_size,
            )
            color = (255, 165, 0) if index == 0 else (0, 200, 0)
            pygame.draw.rect(self.display, color, rect)
            pygame.draw.rect(self.display, (0, 0, 0), rect, 1)
        if self.font:
            score_surface = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.display.blit(score_surface, (10, 5))
        pygame.display.flip()


class LinearQNet(nn.Module):
    """Simple fully connected network for Deep Q-Learning."""

    def __init__(self, input_size: int, hidden_size: int, second_hidden: int, output_size: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, output_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state)

    def predict(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            prediction = self.forward(state)
        return prediction


class QTrainer:
    """Utility class to run optimization steps on the Q-network."""

    def __init__(self, model: LinearQNet, lr: float, gamma: float) -> None:
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device = next(model.parameters()).device

    def train_step(
        self,
        state: Sequence[float] | Sequence[Sequence[float]],
        action: Sequence[int] | int,
        reward: Sequence[float] | float,
        next_state: Sequence[float] | Sequence[Sequence[float]],
        done: Sequence[bool] | bool,
        target_model: LinearQNet | None = None,
    ) -> float:
        state_tensor = torch.tensor(np.asarray(state), dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(np.asarray(next_state), dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(np.asarray(action), dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(np.asarray(reward), dtype=torch.float32, device=self.device)

        single_sample = state_tensor.ndim == 1
        if single_sample:
            state_tensor = state_tensor.unsqueeze(0)
            next_state_tensor = next_state_tensor.unsqueeze(0)
            action_tensor = action_tensor.unsqueeze(0)
            reward_tensor = reward_tensor.unsqueeze(0)
            done = (bool(done),)

        if isinstance(done, bool):
            done = (done,)

        done_list = list(done)

        pred = self.model(state_tensor)
        with torch.no_grad():
            target = pred.clone()
            if target_model is not None:
                next_eval = self.model(next_state_tensor)
                next_actions = torch.argmax(next_eval, dim=1)
                next_pred = target_model(next_state_tensor)
            else:
                next_pred = self.model(next_state_tensor)
                next_actions = None
        for idx, terminal in enumerate(done_list):
            q_new = reward_tensor[idx]
            if not terminal:
                if target_model is not None and next_actions is not None:
                    q_next = next_pred[idx][next_actions[idx]]
                else:
                    q_next = torch.max(next_pred[idx])
                q_new = reward_tensor[idx] + self.gamma * q_next
            target[idx][action_tensor[idx]] = q_new
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


class Agent:
    """Deep Q-Learning agent that interacts with the SnakeGame environment."""

    def __init__(
        self,
        state_size: int = 11,
        action_size: int = 4,
        gamma: float = 0.9,
        lr: float = 1e-3,
        max_memory: int = 50_000,
        batch_size: int = 512,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory: deque[
            Tuple[np.ndarray, int, float, np.ndarray, bool]
        ] = deque(maxlen=max_memory)
        self.n_games = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LinearQNet(state_size, 256, 128, action_size).to(self.device)
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma)
        self.target_model = LinearQNet(state_size, 256, 128, action_size).to(self.device)
        self.target_update_interval = 500
        self.learn_step_counter = 0
        self._sync_target_model(force=True)
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.target_model.eval()
        self.record_score = 0

    def get_state(self, game: SnakeGame) -> np.ndarray:
        if not game.head:
            raise RuntimeError("Game state is not initialised.")
        head = game.head
        dir_vector = game.direction
        right_vector = rotate_right(dir_vector)
        left_vector = rotate_left(dir_vector)
        point_straight = Point(head.x + dir_vector[0], head.y + dir_vector[1])
        point_right = Point(head.x + right_vector[0], head.y + right_vector[1])
        point_left = Point(head.x + left_vector[0], head.y + left_vector[1])
        food = game.food if game.food else head
        state = np.array(
            [
                int(game._is_collision(point_straight)),
                int(game._is_collision(point_right)),
                int(game._is_collision(point_left)),
                int(dir_vector == LEFT),
                int(dir_vector == RIGHT),
                int(dir_vector == UP),
                int(dir_vector == DOWN),
                int(food.x < head.x),
                int(food.x > head.x),
                int(food.y < head.y),
                int(food.y > head.y),
            ],
            dtype=np.float32,
        )
        return state

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> float | None:
        if not self.memory:
            return None
        batch_size = min(self.batch_size, len(self.memory))
        mini_sample = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(
            states,
            actions,
            rewards,
            next_states,
            dones,
            target_model=self.target_model,
        )
        self._post_train_update()
        return loss

    def train_short_memory(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        loss = self.trainer.train_step(
            state,
            action,
            reward,
            next_state,
            done,
            target_model=self.target_model,
        )
        self._post_train_update()
        return loss

    def get_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return self.greedy_action(state)

    def greedy_action(self, state: np.ndarray) -> int:
        prediction = self.model.predict(state)
        return int(torch.argmax(prediction).item())

    def update_exploration(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def _sync_target_model(self, force: bool = False) -> None:
        if force or self.learn_step_counter == 0 or self.learn_step_counter % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def _post_train_update(self) -> None:
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_interval == 0:
            self._sync_target_model(force=True)

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self.model.state_dict(),
            "target_model_state": self.target_model.state_dict(),
            "optimizer_state": self.trainer.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "n_games": self.n_games,
            "record_score": self.record_score,
            "memory": list(self.memory),
            "learn_step_counter": self.learn_step_counter,
            "target_update_interval": self.target_update_interval,
        }
        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "Agent":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")
        agent = cls(**kwargs)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=agent.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        agent.model.load_state_dict(checkpoint["model_state"])
        agent.trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        target_state = checkpoint.get("target_model_state")
        if target_state:
            agent.target_model.load_state_dict(target_state)
        else:
            agent._sync_target_model(force=True)
        for param in agent.target_model.parameters():
            param.requires_grad = False
        agent.target_model.eval()
        agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
        agent.n_games = checkpoint.get("n_games", agent.n_games)
        agent.record_score = checkpoint.get("record_score", agent.record_score)
        agent.learn_step_counter = checkpoint.get("learn_step_counter", agent.learn_step_counter)
        agent.target_update_interval = checkpoint.get("target_update_interval", agent.target_update_interval)
        memory_data = checkpoint.get("memory")
        if memory_data:
            agent.memory = deque(memory_data, maxlen=agent.memory.maxlen)
        return agent


def train(
    episodes: int = 500,
    render: bool = False,
    config: GameConfig | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    save_every: int = 1,
    log_path: str | Path | None = None,
    render_speed: int | None = None,
) -> Agent:
    """High-level training loop for the Snake agent."""
    if episodes <= 0:
        raise ValueError("Le nombre d'episodes doit etre strictement positif.")
    config = config or GameConfig()
    if render_speed is not None and render_speed <= 0:
        raise ValueError("render_speed doit etre strictement positif.")
    if render:
        target_speed = render_speed if render_speed else max(300, config.game_speed)
        config = replace(config, game_speed=target_speed)
        print(f"Vitesse d affichage (FPS): {target_speed}")
    log_file = Path(log_path) if log_path else DEFAULT_LOG_PATH
    checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    if resume and checkpoint.exists():
        agent = Agent.load(checkpoint)
        print(
            f"Reprise de l'entrainement depuis {checkpoint} "
            f"(parties jouees: {agent.n_games}, record: {agent.record_score})"
        )
    else:
        agent = Agent()
        if checkpoint.exists() and not resume:
            print(f"Reinitialisation de l'entrainement (checkpoint ignore: {checkpoint}).")
    game = SnakeGame(config=config, render=render)
    scores: list[int] = []
    mean_scores: list[float] = []
    initial_games = agent.n_games
    target_games = initial_games + episodes
    print(f"Journal d entrainement: {log_file}")
    try:
        while agent.n_games < target_games:
            state_old = agent.get_state(game)
            action = agent.get_action(state_old)
            reward, done, score = game.play_step(action)
            state_new = agent.get_state(game)
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)
            if done:
                game.reset()
                agent.n_games += 1
                session_index = agent.n_games - initial_games
                avg_loss = agent.train_long_memory()
                agent.update_exploration()
                scores.append(score)
                mean_score = float(np.mean(scores[-20:]))
                mean_scores.append(mean_score)
                if score > agent.record_score:
                    agent.record_score = score
                message = (
                    f"Session {session_index}/{episodes} | "
                    f"Total: {agent.n_games} | "
                    f"Score: {score} | "
                    f"Record: {agent.record_score} | "
                    f"Moyenne (20): {mean_score:.2f} | "
                    f"Epsilon: {agent.epsilon:.3f}"
                )
                if avg_loss is not None:
                    message += f" | Loss: {avg_loss:.4f}"
                print(message)
                if save_every > 0 and session_index % save_every == 0:
                    agent.save(checkpoint)
                _log_training_progress(
                    log_file,
                    agent.n_games,
                    session_index,
                    episodes,
                    score,
                    agent.record_score,
                    mean_score,
                    agent.epsilon,
                    avg_loss,
                    len(agent.memory),
                )
        return agent
    finally:
        agent.save(checkpoint)
        pygame.quit()


def play(
    episodes: int = 1,
    render: bool = True,
    config: GameConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Run trained agent for a given number of episodes with optional rendering."""
    if episodes <= 0:
        raise ValueError("Le nombre d'episodes doit etre strictement positif.")
    config = config or GameConfig()
    checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    if checkpoint.exists():
        agent = Agent.load(checkpoint)
        print(
            f"Chargement du checkpoint {checkpoint} "
            f"(parties jouees: {agent.n_games}, record: {agent.record_score})"
        )
    else:
        agent = Agent()
        print("Aucun checkpoint trouve, lancement avec un agent non entraine.")
    agent.epsilon = 0.0
    game = SnakeGame(config=config, render=render)
    completed = 0
    last_score = 0
    normal_completion = False
    try:
        while completed < episodes:
            state = agent.get_state(game)
            action = agent.greedy_action(state)
            _, done, score = game.play_step(action)
            if done:
                completed += 1
                last_score = score
                print(f"Partie {completed}/{episodes} terminee | Score: {score}")
                if completed < episodes:
                    game.reset()
        normal_completion = True
    finally:
        if render and game.render_enabled and normal_completion:
            _display_end_screen(game, completed, last_score, episodes)
            _wait_for_exit(game)
        if pygame.get_init():
            pygame.quit()


def _log_training_progress(
    log_file: Path,
    total_games: int,
    session_index: int,
    episodes: int,
    score: int,
    record_score: int,
    mean_score: float,
    epsilon: float,
    avg_loss: float | None,
    memory_size: int,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_file.exists()
    with log_file.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(
                [
                    "total_games",
                    "session_index",
                    "session_target",
                    "score",
                    "record_score",
                    "mean_score",
                    "epsilon",
                    "avg_loss",
                    "memory_size",
                ]
            )
        writer.writerow(
            [
                total_games,
                session_index,
                episodes,
                score,
                record_score,
                f"{mean_score:.4f}",
                f"{epsilon:.6f}",
                "" if avg_loss is None else f"{avg_loss:.6f}",
                memory_size,
            ]
        )


def _display_end_screen(game: SnakeGame, completed: int, final_score: int, episodes: int) -> None:
    if not game.render_enabled or not pygame.display.get_init() or not game.display:
        return
    game._update_ui()
    width = game.display.get_width()
    height = game.display.get_height()
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    game.display.blit(overlay, (0, 0))
    header_font = pygame.font.SysFont("arial", 32)
    info_font = pygame.font.SysFont("arial", 22)
    title_text = f"Simulation terminee ({completed}/{episodes})"
    title_surface = header_font.render(title_text, True, (255, 255, 255))
    title_rect = title_surface.get_rect(center=(width // 2, height // 2 - 40))
    game.display.blit(title_surface, title_rect)
    score_text = f"Dernier score: {final_score}"
    score_surface = info_font.render(score_text, True, (220, 220, 220))
    score_rect = score_surface.get_rect(center=(width // 2, height // 2))
    game.display.blit(score_surface, score_rect)
    hint_surface = info_font.render("Fermez la fenetre ou appuyez sur ESC/ENTREE/ESPACE pour quitter.", True, (200, 200, 200))
    hint_rect = hint_surface.get_rect(center=(width // 2, height // 2 + 40))
    game.display.blit(hint_surface, hint_rect)
    pygame.display.flip()
    print("Simulation terminee. Fermez la fenetre ou appuyez sur ESC/ENTREE/ESPACE pour quitter.")


def _wait_for_exit(game: SnakeGame) -> None:
    if not game.render_enabled or not pygame.display.get_init():
        return
    clock = pygame.time.Clock()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE,
                pygame.K_RETURN,
                pygame.K_SPACE,
                pygame.K_q,
            ):
                waiting = False
        clock.tick(15)


if __name__ == "__main__":
    play(episodes=1, render=True)
