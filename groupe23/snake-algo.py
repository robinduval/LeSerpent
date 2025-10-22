import pygame
import random
import time
import heapq
import matplotlib.pyplot as plt
import os

GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 8
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BG_GRAY = (50, 50, 50)
GRID_GRAY = (80, 80, 80)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

current_algorithm = 'A*'
game_results = []

class Snake:
    def __init__(self):
        start_x = GRID_SIZE // 4
        start_y = GRID_SIZE // 2
        self.head_pos = [start_x, start_y]
        self.body = [
            [start_x, start_y],
            [start_x - 1, start_y],
            [start_x - 2, start_y]
        ]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_direction):
        opposite_direction = (new_direction[0] * -1, new_direction[1] * -1)
        if opposite_direction != tuple(self.direction):
            self.direction = new_direction

    def move(self):
        new_x = (self.head_pos[0] + self.direction[0]) % GRID_SIZE
        new_y = (self.head_pos[1] + self.direction[1]) % GRID_SIZE
        new_head = [new_x, new_y]
        
        self.body.insert(0, new_head)
        self.head_pos = new_head
        
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
            x = segment[0] * CELL_SIZE
            y = segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT
            body_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, GREEN, body_rect)
            pygame.draw.rect(surface, BLACK, body_rect, 1)
        
        head_x = self.head_pos[0] * CELL_SIZE
        head_y = self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT
        head_rect = pygame.Rect(head_x, head_y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, BLACK, head_rect, 2)

class Apple:
    def __init__(self, snake_body):
        self.position = self.find_free_position(snake_body)

    def find_free_position(self, occupied_cells):
        all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        free_cells = [cell for cell in all_cells if list(cell) not in occupied_cells]
        
        if not free_cells:
            return None
        return random.choice(free_cells)

    def relocate(self, snake_body):
        new_position = self.find_free_position(snake_body)
        if new_position:
            self.position = new_position
            return True
        return False

    def draw(self, surface):
        if self.position:
            apple_x = self.position[0] * CELL_SIZE
            apple_y = self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT
            apple_rect = pygame.Rect(apple_x, apple_y, CELL_SIZE, CELL_SIZE)
            
            pygame.draw.rect(surface, RED, apple_rect, border_radius=5)
            
            highlight_x = apple_rect.x + int(CELL_SIZE * 0.7)
            highlight_y = apple_rect.y + int(CELL_SIZE * 0.3)
            pygame.draw.circle(surface, WHITE, (highlight_x, highlight_y), CELL_SIZE // 8)

def manhattan_distance(position_a, position_b):
    dx = abs(position_a[0] - position_b[0])
    dy = abs(position_a[1] - position_b[1])
    
    wrapped_dx = min(dx, GRID_SIZE - dx)
    wrapped_dy = min(dy, GRID_SIZE - dy)
    
    return wrapped_dx + wrapped_dy

def calculate_space_accessibility(position, blocked_cells):
    visited = {tuple(position)}
    queue = [tuple(position)]
    accessible_count = 0
    
    while queue:
        current = queue.pop(0)
        accessible_count += 1
        
        for neighbor in get_neighbors(current):
            if neighbor in visited or neighbor in blocked_cells:
                continue
            
            visited.add(neighbor)
            queue.append(neighbor)
    
    return accessible_count

def evaluate_move_safety(snake, next_position, apple_position):
    simulated_body = [list(next_position)] + [list(pos) for pos in snake.body[:-1]]
    
    if list(next_position) == list(apple_position):
        simulated_body.append(list(snake.body[-1]))
    
    blocked = set(tuple(pos) for pos in simulated_body[:-1])
    accessible = calculate_space_accessibility(next_position, blocked)
    
    body_length = len(simulated_body)
    safety_ratio = accessible / (GRID_SIZE * GRID_SIZE - body_length + 1)
    
    return safety_ratio

def get_neighbors(position):
    for direction in DIRECTIONS:
        neighbor_x = (position[0] + direction[0]) % GRID_SIZE
        neighbor_y = (position[1] + direction[1]) % GRID_SIZE
        yield (neighbor_x, neighbor_y)

def find_path_astar(start, goal, blocked_cells):
    open_set = []
    start_tuple = tuple(start)
    goal_tuple = tuple(goal)
    
    initial_cost = manhattan_distance(start_tuple, goal_tuple)
    heapq.heappush(open_set, (initial_cost, 0, start_tuple, None))
    
    visited = {}
    distance_from_start = {start_tuple: 0}

    while open_set:
        estimated_total, current_distance, current_pos, parent = heapq.heappop(open_set)
        
        if current_pos in visited:
            continue
            
        visited[current_pos] = parent
        
        if current_pos == goal_tuple:
            path = [current_pos]
            while visited[path[-1]] is not None:
                path.append(visited[path[-1]])
            path.reverse()
            return path
        
        for neighbor in get_neighbors(current_pos):
            if neighbor in blocked_cells and neighbor != goal_tuple:
                continue
            
            new_distance = current_distance + 1
            if neighbor not in distance_from_start or new_distance < distance_from_start[neighbor]:
                distance_from_start[neighbor] = new_distance
                heuristic = manhattan_distance(neighbor, goal_tuple)
                estimated_cost = new_distance + heuristic * 1.1
                heapq.heappush(open_set, (estimated_cost, new_distance, neighbor, current_pos))
    
    return None

def find_path_bfs(start, goal, blocked_cells):
    start_tuple = tuple(start)
    goal_tuple = tuple(goal)
    
    if start_tuple == goal_tuple:
        return [start_tuple]
    
    queue = [(start_tuple, [start_tuple])]
    visited = {start_tuple}
    
    while queue:
        current_pos, path = queue.pop(0)
        
        for neighbor in get_neighbors(current_pos):
            if neighbor == goal_tuple:
                return path + [neighbor]
            
            if neighbor in visited:
                continue
            
            if neighbor in blocked_cells and neighbor != goal_tuple:
                continue
            
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    
    return None

def is_reachable(start, goal, blocked_cells):
    start_tuple = tuple(start)
    goal_tuple = tuple(goal)
    
    if start_tuple == goal_tuple:
        return True
    
    queue = [start_tuple]
    visited = {start_tuple}
    
    while queue:
        current = queue.pop(0)
        
        for neighbor in get_neighbors(current):
            if neighbor == goal_tuple:
                return True
            
            if neighbor in visited or neighbor in blocked_cells:
                continue
            
            visited.add(neighbor)
            queue.append(neighbor)
    
    return False

def simulate_path_safety(snake, path, goal):
    simulated_body = [list(pos) for pos in snake.body]
    
    for step in path[1:]:
        simulated_body.insert(0, [step[0], step[1]])
        
        if tuple(step) == goal:
            pass
        else:
            simulated_body.pop()
    
    blocked = set(tuple(pos) for pos in simulated_body[:-1])
    head = tuple(simulated_body[0])
    tail = tuple(simulated_body[-1])
    
    return is_reachable(head, tail, blocked)

def find_safe_fallback_move(snake):
    for direction in DIRECTIONS:
        next_x = (snake.head_pos[0] + direction[0]) % GRID_SIZE
        next_y = (snake.head_pos[1] + direction[1]) % GRID_SIZE
        
        if [next_x, next_y] not in snake.body:
            return direction
    
    return snake.direction

def convert_cell_to_direction(current_pos, next_cell):
    delta_x = next_cell[0] - current_pos[0]
    delta_y = next_cell[1] - current_pos[1]
    
    if delta_x > 1:
        delta_x -= GRID_SIZE
    if delta_x < -1:
        delta_x += GRID_SIZE
    if delta_y > 1:
        delta_y -= GRID_SIZE
    if delta_y < -1:
        delta_y += GRID_SIZE
    
    if delta_x > 0:
        return RIGHT
    if delta_x < 0:
        return LEFT
    if delta_y > 0:
        return DOWN
    if delta_y < 0:
        return UP
    
    return None

def compute_next_direction(snake, apple):
    global current_algorithm
    
    current_pos = tuple(snake.head_pos)
    target_pos = tuple(apple.position)
    blocked_cells = set(tuple(pos) for pos in snake.body[:-1])
    
    if current_algorithm == 'A*':
        path = find_path_astar(current_pos, target_pos, blocked_cells)
    else:
        path = find_path_bfs(current_pos, target_pos, blocked_cells)
    
    if path and len(path) >= 2:
        next_cell = path[1]
        
        if not simulate_path_safety(snake, path, target_pos):
            tail_pos = tuple(snake.body[-1])
            
            if current_algorithm == 'A*':
                path_to_tail = find_path_astar(current_pos, tail_pos, blocked_cells)
            else:
                path_to_tail = find_path_bfs(current_pos, tail_pos, blocked_cells)
            
            if path_to_tail and len(path_to_tail) >= 2:
                next_cell = path_to_tail[1]
            else:
                return find_safe_fallback_move(snake)
        
        direction = convert_cell_to_direction(snake.head_pos, next_cell)
        if direction:
            return direction
    
    return find_safe_fallback_move(snake)

def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        line_start = (x, SCORE_PANEL_HEIGHT)
        line_end = (x, SCREEN_HEIGHT)
        pygame.draw.line(surface, GRID_GRAY, line_start, line_end)
    
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        line_start = (0, y)
        line_end = (SCREEN_WIDTH, y)
        pygame.draw.line(surface, GRID_GRAY, line_start, line_end)

def main():
    global current_algorithm, game_results
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f'Snake - Algo ({current_algorithm})')
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_big = pygame.font.Font(None, 80)

    snake = Snake()
    apple = Apple(snake.body)

    is_running = True
    is_game_over = False
    is_victory = False
    start_time = time.time()
    end_time = None
    move_counter = 0
    total_moves = 0

    while is_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

        if not is_game_over and not is_victory:
            move_counter += 1
            
            if move_counter >= GAME_SPEED // 6:
                next_direction = compute_next_direction(snake, apple)
                snake.set_direction(next_direction)
                snake.move()
                total_moves += 1
                move_counter = 0

                if snake.check_self_collision():
                    is_game_over = True
                    end_time = time.time()

                if snake.head_pos == list(apple.position):
                    snake.grow()
                    
                    if not apple.relocate(snake.body):
                        is_victory = True
                        is_game_over = True
                        end_time = time.time()

        screen.fill(BG_GRAY)
        
        game_area = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, BLACK, game_area)
        
        draw_grid(screen)
        apple.draw(screen)
        snake.draw(screen)

        score_label = font_main.render(f"Score: {snake.score}", True, WHITE)
        screen.blit(score_label, (10, 10))
        
        if end_time is not None:
            elapsed_seconds = int(end_time - start_time)
        else:
            elapsed_seconds = int(time.time() - start_time)
        
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        time_label = font_main.render(f"Temps: {minutes:02d}:{seconds:02d}", True, WHITE)
        time_x = SCREEN_WIDTH - time_label.get_width() - 10
        screen.blit(time_label, (time_x, 10))
        
        algo_label = font_main.render(f"Algo: {current_algorithm}", True, WHITE)
        algo_x = (SCREEN_WIDTH - algo_label.get_width()) // 2
        screen.blit(algo_label, (algo_x, 10))
        
        moves_label = font_main.render(f"Mouvements: {total_moves}", True, WHITE)
        screen.blit(moves_label, (10, 45))

        if is_game_over:
            if is_victory:
                end_message = font_big.render("VICTORY!", True, GREEN)
            else:
                end_message = font_big.render("GAME OVER", True, RED)
            
            message_rect = end_message.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(end_message, message_rect)
            
            pygame.display.flip()
            pygame.time.wait(2000)
            is_running = False

        pygame.display.flip()
        clock.tick(GAME_SPEED)

    pygame.quit()
    
    if end_time:
        game_results.append({
            'algorithm': current_algorithm,
            'score': snake.score,
            'time': end_time - start_time,
            'moves': total_moves
        })
    
    if current_algorithm == 'A*':
        current_algorithm = 'BFS'
        pygame.time.wait(500)
        main()
    else:
        display_results_chart()

def display_results_chart():
    if len(game_results) < 2:
        print("Pas assez de résultats pour générer le graphique.")
        return
    
    algorithms = [result['algorithm'] for result in game_results]
    scores = [result['score'] for result in game_results]
    times = [result['time'] for result in game_results]
    moves = [result.get('moves', 0) for result in game_results]
    
    colors = ['#4f81bd', '#9bbb59']
    figure, (axis_score, axis_time, axis_moves) = plt.subplots(1, 3, figsize=(15, 5))

    axis_score.bar(algorithms, scores, color=colors)
    axis_score.set_ylabel('Score')
    axis_score.set_title('Comparaison des Scores')
    max_score = max(scores + [1])
    axis_score.set_ylim(0, max_score * 1.2)
    
    for index, value in enumerate(scores):
        label_y = value + max_score * 0.02
        axis_score.text(index, label_y, str(value), ha='center')

    axis_time.bar(algorithms, times, color=colors)
    axis_time.set_ylabel('Temps (s)')
    axis_time.set_title('Comparaison du Temps')
    max_time = max(times + [1])
    axis_time.set_ylim(0, max_time * 1.2)
    
    for index, value in enumerate(times):
        label_y = value + max_time * 0.02
        axis_time.text(index, label_y, f"{value:.1f}s", ha='center')

    axis_moves.bar(algorithms, moves, color=colors)
    axis_moves.set_ylabel('Mouvements')
    axis_moves.set_title('Comparaison des Mouvements')
    max_moves = max(moves + [1])
    axis_moves.set_ylim(0, max_moves * 1.2)
    
    for index, value in enumerate(moves):
        label_y = value + max_moves * 0.02
        axis_moves.text(index, label_y, str(value), ha='center')

    plt.tight_layout()
    
    save_path = os.path.join(os.getcwd(), 'resultats_algo.png')
    plt.savefig(save_path)
    print(f"Graphique sauvegardé: {save_path}")
    plt.show()
    times = [r['time'] for r in game_results]
    
    moves = [r.get('moves', 0) for r in game_results]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Score comparison
    ax1.bar(algorithms, scores, color=['#4f81bd', '#9bbb59'])
    ax1.set_ylabel('Score')
    ax1.set_title('Comparaison des Scores')
    ax1.set_ylim(0, max(scores + [1]) * 1.2)
    for i, v in enumerate(scores):
        ax1.text(i, v + max(scores)*0.02, str(v), ha='center')

    # Time comparison
    ax2.bar(algorithms, times, color=['#4f81bd', '#9bbb59'])
    ax2.set_ylabel('Temps (s)')
    ax2.set_title('Comparaison du Temps')
    ax2.set_ylim(0, max(times + [1]) * 1.2)
    for i, v in enumerate(times):
        ax2.text(i, v + max(times)*0.02, f"{v:.1f}s", ha='center')

    # Moves comparison
    ax3.bar(algorithms, moves, color=['#4f81bd', '#9bbb59'])
    ax3.set_ylabel('Mouvements')
    ax3.set_title('Comparaison des Mouvements')
    ax3.set_ylim(0, max(moves + [1]) * 1.2)
    for i, v in enumerate(moves):
        ax3.text(i, v + max(moves)*0.02, str(v), ha='center')

    plt.tight_layout()
    # Ensure save path exists (save into current working directory)
    save_path = os.path.join(os.getcwd(), 'resultats_algo.png')
    plt.savefig(save_path)
    print(f"Graphique sauvegardé: {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
