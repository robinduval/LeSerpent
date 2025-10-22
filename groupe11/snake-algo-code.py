"""snake_algo.py

Optimized Snake control utilities with anti-loop and high-fill strategies.

API:
  get_next_direction(snake_body, apple_pos, grid_size, prefer_long_term=True)
    - snake_body: list of [x,y] positions, head first
    - apple_pos: (x,y) tuple
    - grid_size: int (square grid)
    - prefer_long_term: if True, prefer safe paths that avoid trapping
    Returns: direction tuple (dx,dy) or None if no move

Improvements:
  - Anti-loop detection: tracks recent positions to avoid cycling
  - Hamiltonian cycle strategy for high-fill scenarios (>60%)
  - Progressive safety thresholds based on grid fill percentage
  - Better tail-following with space-filling patterns
"""

from collections import deque
import heapq
from typing import List, Tuple, Optional, Set

Pos = Tuple[int, int]

# Global state for loop detection (reset between games)
_recent_positions = deque(maxlen=50)
_position_visit_count = {}
_last_apple_pos = None
_moves_since_apple = 0


def reset_loop_detection():
    """Call this when starting a new game."""
    global _recent_positions, _position_visit_count, _last_apple_pos, _moves_since_apple
    _recent_positions.clear()
    _position_visit_count.clear()
    _last_apple_pos = None
    _moves_since_apple = 0


def detect_loop(head: Pos, apple_pos: Pos) -> bool:
    """Detect if snake is in a loop (visiting same positions repeatedly without progress)."""
    global _last_apple_pos, _moves_since_apple, _position_visit_count
    
    # Reset counter if we got a new apple
    if apple_pos != _last_apple_pos:
        _last_apple_pos = apple_pos
        _moves_since_apple = 0
        _position_visit_count.clear()
    else:
        _moves_since_apple += 1
    
    # Track position visits
    _position_visit_count[head] = _position_visit_count.get(head, 0) + 1
    
    # Loop detected if we've visited same spot 3+ times recently without getting apple
    if _position_visit_count[head] >= 3 and _moves_since_apple > 20:
        return True
    
    # Also check if we're cycling through same small set of positions
    if len(_recent_positions) >= 40:
        recent_set = set(_recent_positions)
        if len(recent_set) < 15:  # Visiting less than 15 unique positions in last 40 moves
            return True
    
    _recent_positions.append(head)
    return False


def neighbors(pos: Pos, grid_size: int):
    """Toroidal neighbors (wrap-around on edges)."""
    x, y = pos
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = (x + dx) % grid_size, (y + dy) % grid_size
        yield (nx, ny)


def toroidal_distance(a: Pos, b: Pos, grid_size: int) -> int:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dx = min(dx, grid_size - dx)
    dy = min(dy, grid_size - dy)
    return dx + dy


def a_star(start: Pos, goal: Pos, obstacles: Set[Pos], grid_size: int) -> Optional[List[Pos]]:
    """A* shortest path from start to goal avoiding obstacles."""
    open_heap = []
    heapq.heappush(open_heap, (toroidal_distance(start, goal, grid_size), 0, start))
    came_from = {start: None}
    gscore = {start: 0}

    while open_heap:
        _, cost, current = heapq.heappop(open_heap)
        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        for nbr in neighbors(current, grid_size):
            if nbr in obstacles and nbr != goal:
                continue
            tentative_g = gscore[current] + 1
            if tentative_g < gscore.get(nbr, 1e9):
                came_from[nbr] = current
                gscore[nbr] = tentative_g
                f = tentative_g + toroidal_distance(nbr, goal, grid_size)
                heapq.heappush(open_heap, (f, tentative_g, nbr))

    return None


def reachable(start: Pos, target: Pos, obstacles: Set[Pos], grid_size: int) -> bool:
    """Simple BFS to check whether target is reachable from start avoiding obstacles."""
    if start == target:
        return True
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for nbr in neighbors(cur, grid_size):
            if nbr in seen or (nbr in obstacles and nbr != target):
                continue
            if nbr == target:
                return True
            seen.add(nbr)
            q.append(nbr)
    return False


def count_reachable_area(start: Pos, obstacles: Set[Pos], grid_size: int, max_depth: int = None) -> int:
    """Count number of cells reachable from start."""
    q = deque([start])
    seen = {start}
    depth = {start: 0}
    count = 0
    
    while q:
        cur = q.popleft()
        count += 1
        
        if max_depth and depth[cur] >= max_depth:
            continue
            
        for nbr in neighbors(cur, grid_size):
            if nbr in seen or nbr in obstacles:
                continue
            seen.add(nbr)
            depth[nbr] = depth[cur] + 1
            q.append(nbr)
    
    return count


def project_body_after_path(snake_body: List[List[int]], path: List[Pos]) -> List[Pos]:
    """Project snake body after following path (without growing)."""
    body = [tuple(p) for p in snake_body]
    for step in path[1:]:
        body.insert(0, step)
        body.pop()
    return body


def first_step_from_path(path: List[Pos], grid_size: int) -> Optional[Pos]:
    """Convert path to direction tuple, handling toroidal wrap."""
    if not path or len(path) < 2:
        return None
    hx, hy = path[0]
    nx, ny = path[1]
    dx = nx - hx
    dy = ny - hy
    
    # Handle wrap-around: if distance is more than half grid, we wrapped
    if abs(dx) > grid_size // 2:
        dx = -1 if dx > 0 else 1
    if abs(dy) > grid_size // 2:
        dy = -1 if dy > 0 else 1
    
    return (dx, dy)


def hamiltonian_move(head: Pos, snake_body: List[List[int]], grid_size: int) -> Optional[Pos]:
    """Follow a Hamiltonian cycle pattern to systematically fill the grid.
    Uses a serpentine (snake) pattern that guarantees full coverage.
    """
    x, y = head
    obstacles = {tuple(p) for p in snake_body[:-1]}
    tail = tuple(snake_body[-1])
    
    # Serpentine pattern: zigzag through the grid
    # Row 0: left to right (0→grid_size-1)
    # Row 1: right to left (grid_size-1→0)
    # Row 2: left to right, etc.
    
    def get_ideal_next_pos(cx, cy):
        """Calculate the next position in the Hamiltonian cycle."""
        if cy % 2 == 0:
            # Even row: move right
            if cx < grid_size - 1:
                return (cx + 1, cy)
            else:
                # End of even row: move down
                return (cx, cy + 1)
        else:
            # Odd row: move left
            if cx > 0:
                return (cx - 1, cy)
            else:
                # End of odd row: move down
                return (cx, cy + 1)
    
    # Calculate ideal next position in the cycle
    ideal_next = get_ideal_next_pos(x, y)
    
    # Wrap around at bottom
    if ideal_next[1] >= grid_size:
        ideal_next = (0, 0)
    
    # Try ideal position first
    candidates = []
    ix, iy = ideal_next
    dx_ideal = ix - x
    dy_ideal = iy - y
    
    # Handle toroidal wrap
    if abs(dx_ideal) > grid_size // 2:
        dx_ideal = -1 if dx_ideal > 0 else 1
    if abs(dy_ideal) > grid_size // 2:
        dy_ideal = -1 if dy_ideal > 0 else 1
    
    # Prioritize the ideal direction
    candidates.append((dx_ideal, dy_ideal))
    
    # Add alternative directions that keep the pattern
    if y % 2 == 0:
        # Even row: prefer right, then down, then up, then left
        for d in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
            if d not in candidates:
                candidates.append(d)
    else:
        # Odd row: prefer left, then down, then up, then right
        for d in [(-1, 0), (0, 1), (0, -1), (1, 0)]:
            if d not in candidates:
                candidates.append(d)
    
    # Try each direction and score them
    best_move = None
    best_score = -1
    
    for dx, dy in candidates:
        nx, ny = (x + dx) % grid_size, (y + dy) % grid_size
        
        if (nx, ny) in obstacles:
            continue
        
        # Simulate move
        new_body = [(nx, ny)] + [tuple(p) for p in snake_body[:-1]]
        new_obstacles = set(new_body[:-1])
        
        # Must be able to reach tail
        if not reachable((nx, ny), tail, new_obstacles, grid_size):
            continue
        
        # Score: prefer moves that follow the Hamiltonian pattern
        score = 100
        
        # Bonus if this is the ideal next position
        if (nx, ny) == ideal_next:
            score += 500
        
        # Bonus for following the row direction
        if y % 2 == 0 and dx == 1:  # Even row, moving right
            score += 200
        elif y % 2 == 1 and dx == -1:  # Odd row, moving left
            score += 200
        
        # Bonus for large reachable area (safety)
        reachable_area = count_reachable_area((nx, ny), new_obstacles, grid_size, max_depth=30)
        score += reachable_area
        
        if score > best_score:
            best_score = score
            best_move = (dx, dy)
    
    return best_move


def find_safest_move(snake_body: List[List[int]], grid_size: int, avoid_positions: Set[Pos] = None) -> Optional[Pos]:
    """Find the move that maximizes safety and reachable area.
    Optionally avoid recently visited positions to break loops.
    """
    head = tuple(snake_body[0])
    obstacles = {tuple(p) for p in snake_body[:-1]}
    avoid_positions = avoid_positions or set()

    best = None
    best_score = -1
    
    for nbr in neighbors(head, grid_size):
        if nbr in obstacles:
            continue
        
        # Penalty for recently visited positions
        visit_penalty = _position_visit_count.get(nbr, 0) * 100
        
        # Skip if visited too many times (unless no other choice)
        if nbr in avoid_positions and best is not None:
            continue
        
        # Simulate move
        new_body = [nbr] + [tuple(p) for p in snake_body[:-1]]
        new_obstacles = set(new_body[:-1])
        tail = tuple(snake_body[-1])
        
        # Must be able to reach tail
        if not reachable(nbr, tail, new_obstacles, grid_size):
            continue
        
        # Score based on reachable area
        area = count_reachable_area(nbr, new_obstacles, grid_size, max_depth=50)
        
        # Bonus for distance from walls (in non-toroidal sense, prefer center when crowded)
        center_bonus = 0
        if grid_size > 10:
            center = grid_size // 2
            dist_from_center = abs(nbr[0] - center) + abs(nbr[1] - center)
            center_bonus = max(0, grid_size - dist_from_center) * 2
        
        score = area + center_bonus - visit_penalty
        
        if score > best_score:
            best_score = score
            best = nbr

    if best is None:
        return None
    
    hx, hy = head
    bx, by = best
    dx = bx - hx
    dy = by - hy
    
    # Handle wrap
    if abs(dx) > 1:
        dx = -1 if dx > 0 else 1
    if abs(dy) > 1:
        dy = -1 if dy > 0 else 1
    
    return (dx, dy)


def get_next_direction(snake_body: List[List[int]], apple_pos: Pos, grid_size: int, prefer_long_term: bool = True) -> Optional[Pos]:
    """Compute next direction with anti-loop and high-fill strategies."""
    if not snake_body:
        return None
    
    head = tuple(snake_body[0])
    apple = tuple(apple_pos)
    length = len(snake_body)
    total_cells = grid_size * grid_size
    fill_percentage = length / total_cells
    
    # Detect loops
    in_loop = detect_loop(head, apple)
    
    # Build obstacles (all body except tail which will move)
    obstacles = {tuple(p) for p in snake_body[:-1]}
    tail = tuple(snake_body[-1])
    
    # Progressive safety strategy based on fill percentage
    if fill_percentage < 0.50:
        # Early game: be aggressive, go for apple
        safety_threshold = 0.50
    elif fill_percentage < 0.70:
        # Mid game: balance aggression and safety
        safety_threshold = 0.70
    elif fill_percentage < 0.85:
        # Late game: prioritize safety
        safety_threshold = 0.90
    else:
        # End game: survival mode, ignore apple if dangerous
        safety_threshold = 0.95
    
    # In end-game (>85% full), prioritize survival over apple
    if fill_percentage >= 0.85:
        # Try Hamiltonian pattern first
        hamiltonian = hamiltonian_move(head, snake_body, grid_size)
        if hamiltonian:
            return hamiltonian
        
        # Otherwise follow tail closely
        path_to_tail = a_star(head, tail, obstacles - {tail}, grid_size)
        if path_to_tail and len(path_to_tail) >= 2:
            return first_step_from_path(path_to_tail, grid_size)
        
        # Safest move
        safe = find_safest_move(snake_body, grid_size)
        if safe:
            return safe
    
    # If in a loop, force a different strategy
    if in_loop:
        # Try to break loop by avoiding recently visited positions
        recent_set = set(_recent_positions)
        safe = find_safest_move(snake_body, grid_size, avoid_positions=recent_set)
        if safe:
            return safe
        
        # Try hamiltonian pattern
        hamiltonian = hamiltonian_move(head, snake_body, grid_size)
        if hamiltonian:
            return hamiltonian
    
    # Try path to apple
    path = a_star(head, apple, obstacles, grid_size)
    if path and len(path) >= 2:
        # Check safety
        projected = project_body_after_path(snake_body, path)
        proj_obstacles = set(projected[:-1])
        proj_tail = projected[-1]
        
        # Calculate safety score
        reachable_after = count_reachable_area(projected[0], proj_obstacles, grid_size)
        required_space = length + 5  # Need room to maneuver
        
        if reachable_after >= required_space * safety_threshold:
            return first_step_from_path(path, grid_size)
    
    # Apple path not safe enough, try tail following
    path_to_tail = a_star(head, tail, obstacles - {tail}, grid_size)
    if path_to_tail and len(path_to_tail) >= 2:
        return first_step_from_path(path_to_tail, grid_size)
    
    # Try hamiltonian pattern
    hamiltonian = hamiltonian_move(head, snake_body, grid_size)
    if hamiltonian:
        return hamiltonian
    
    # Find safest available move
    safe = find_safest_move(snake_body, grid_size)
    if safe:
        return safe
    
    # Last resort: any legal move
    for nbr in neighbors(head, grid_size):
        if nbr not in obstacles:
            hx, hy = head
            nx, ny = nbr
            dx = nx - hx
            dy = ny - hy
            if abs(dx) > 1:
                dx = -1 if dx > 0 else 1
            if abs(dy) > 1:
                dy = -1 if dy > 0 else 1
            return (dx, dy)
    
    return None
