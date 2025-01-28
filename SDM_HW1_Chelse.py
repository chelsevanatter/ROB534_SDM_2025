#!/usr/bin/env python3

import abc
import numpy as np
import time
import heapq
import random
from sklearn.neighbors import NearestNeighbors 
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from heapq import heappop, heappush

class Maze(abc.ABC):
    """ Base Maze Class """

    def __init__(self, maze_array, start_index=None, goal_index=None):
        """
            maze_array - 2D numpy array with 1s representing free space
                        0s representing occupied space
        """
        self.maze_array = maze_array
        self.cols, self.rows = self.maze_array.shape
        self.start_index = start_index
        self.goal_index = goal_index

    def __repr__(self):
        if isinstance(self, Maze2D):
            output = "2D Maze\n"
        output += str(self.maze_array)
        return output

    @classmethod
    def from_pgm(cls, filename):
        """
            Initializes the Maze from a (8 bit) PGM file
        """
        with open(filename, 'r', encoding='latin1') as infile:
            header = infile.readline()
            width, height, _ = [int(item) for item in header.split()[1:]]
            image = np.fromfile(infile, dtype=np.uint8).reshape((height, width)) / 255

        return cls(image.T)

    def plot_maze(self):
        """ Visualizes the maze """
        self.plot_path([], "Maze")

    def plot_path(self, path, title_name=None, runtime=None, path_length=None):
        """
            Plots the provided path on the maze and optionally shows
            runtime and path length in a text box outside the plot area.
        """
        fig = plt.figure(1)
        ax1 = fig.add_subplot(1, 1, 1)

        spacing = 1.0  # Spacing between grid lines
        minor_location = MultipleLocator(spacing)

        # Set minor tick locations.
        ax1.yaxis.set_minor_locator(minor_location)
        ax1.xaxis.set_minor_locator(minor_location)

        # Set grid to use minor tick locations.
        ax1.grid(which='minor')

        colors = ['b', 'r']
        plt.imshow(self.maze_array.T, cmap=plt.get_cmap('bone'))
        
        if title_name is not None:
            fig.suptitle(title_name, fontsize=20)

        # cast path to numpy array so indexing is nicer
        path = np.array(path)
        for i in range(len(path) - 1):
            cidx = i % 2
            plt.plot([path[i, 0], path[i + 1, 0]], [path[i, 1], path[i + 1, 1]], color=colors[cidx], linewidth=4)

        # If runtime and path length are provided, add them outside the plot
        if runtime is not None and path_length is not None:
            text = f"Path Length: {path_length:.2f}\nRuntime: {runtime:.2f}s"
            
            # Adjust position and move the text further down
            fig.subplots_adjust(bottom=0.2)  # Make more room for the text box below the graph
            plt.figtext(0.5, 0.02, text, ha="center", fontsize=12)  # Move the text further down

        plt.show()

    def check_occupancy(self, state):
        """ Returns True if there is an obstacle at state """
        return self.maze_array[int(state[0]), int(state[1])] == 0
    def get_goal(self):
        """ Returns the index of the goal """
        return self.goal_index

    def get_start(self):
        """ Returns the index of the start state """
        return self.start_index

    def check_hit(self, start, deltas):
        """
            Returns True if there are any occupied states between:
            start[0] to start[0]+dx and start[1] to start[1]+dy
        """
        x, y = start
        dx, dy = deltas

        if (x < 0) or (y < 0) or (x >= self.cols) or (y >= self.rows):
            return True

        if self.maze_array[int(round(start[0])), int(round(start[1]))] == 0:
            return True

        if dx == 0.0 and dy == 0.0:  # no actual movement
            return False

        norm = max(abs(dx), abs(dy))
        dx /= norm
        dy /= norm

        for _ in range(int(norm)):
            x += dx
            y += dy
            if (x < 0) or (y < 0) or (x >= self.cols) or (y >= self.rows):
                return True
            if self.maze_array[int(x), int(y)] == 0:
                return True
        return False

    def check_occupancy(self, state):
        """ Returns True if there is an obstacle at state """
        return self.maze_array[int(state[0]), int(state[1])] == 0


class Maze2D(Maze):
    """ Maze2D Class """

    def __init__(self, maze_array, start_state=None, goal_state=None):
        super().__init__(maze_array, start_state, goal_state)

        if start_state is None:
            start_state = (0, 0)
        self.start_state = start_state
        self.start_index = self.index_from_state(self.start_state)

        if goal_state is None:
            goal_state = (self.cols-1, self.rows-1)
        self.goal_state = goal_state
        self.goal_index = self.index_from_state(self.goal_state)

    def index_from_state(self, state):
        """ Gets a unique index for the state """
        return state[0] * self.rows + state[1]

    def state_from_index(self, state_id):
        """ Returns the state at a given index """
        x = int(np.floor(state_id / self.rows))
        y = state_id % self.rows
        return (x, y)

    def get_neighbors(self, state_id):
        """ Returns a List of indices corresponding to neighbors of a given state """
        state = self.state_from_index(state_id)
        deltas = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        neighbors = []
        for delta in deltas:
            if not self.check_hit(state, delta):
                new_state = (state[0] + delta[0], state[1] + delta[1])
                neighbors.append(self.index_from_state(new_state))
        return neighbors
    
    # Step 2 part i code
    def a_star_step2_i(self):
        """ Perform A* search to find the optimal path from start to goal """
        start = self.start_index
        goal = self.goal_index

        open_set = []
        heappush(open_set, (0, start))  # (f_score, state_index)

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start)}

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def _heuristic(self, state_index):
        """ Calculate Manhattan distance heuristic to goal """
        state = self.state_from_index(state_index)
        goal = self.state_from_index(self.goal_index)
        return abs(goal[0] - state[0]) + abs(goal[1] - state[1])

    def _reconstruct_path(self, came_from, current):
        """ Reconstruct path from start to goal """
        path = [self.state_from_index(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self.state_from_index(current))
        return path[::-1]
    
    # Step 2 part ii code
    def a_star_step2_ii(self, epsilon=1):
        """ Perform A* search with a greedy heuristic adjusted by epsilon """
        start, goal = self.start_index, self.goal_index

        open_set = [(0, start)]
        came_from, g_score, f_score = {}, {start: 0}, {start: self._greedy_heuristic(start, epsilon)}

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current), len(came_from)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._greedy_heuristic(neighbor, epsilon)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return [], 0  # No path found

    def _greedy_heuristic(self, state_index, epsilon):
        """ Calculate the greedy heuristic to the goal """
        state = self.state_from_index(state_index)
        goal = self.state_from_index(self.goal_index)
        return epsilon * (abs(goal[0] - state[0]) + abs(goal[1] - state[1]))

    def run_with_epsilon_decay(self, initial_epsilon=10, time_limits=[0.05, 0.25, 1]):
        for time_limit in time_limits:
            epsilon = initial_epsilon
            start_time = time.time()
            print(f"Running with time limit {time_limit}s and initial epsilon {epsilon}")

            while time.time() - start_time < time_limit:
                path, expanded_nodes = self.a_star_step2_ii(epsilon)
                if path:  # If path found
                    break
                # Decay epsilon
                epsilon = max(1, epsilon - 0.5 * (epsilon - 1))

            print(f"Epsilon: {epsilon}, Nodes expanded: {expanded_nodes}, Path length: {len(path)}")
    
    # Step 2 part iii code
    def rrt(self, max_iter=1000, step_size=1):
            def sample_free():
                while True:
                    if random.random() < 0.2:
                        return np.array(self.goal_state)
                    else:
                        x = random.randint(0, self.cols - 1)
                        y = random.randint(0, self.rows - 1)
                        if self.check_occupancy((x, y)) == 0:
                            return np.array([x, y])

            def nearest_node(tree, sample):
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(tree)
                _, idx = nbrs.kneighbors([sample])
                return tree[idx[0][0]] 

            def steer(from_node, to_node, step_size):
                vector = to_node - from_node
                distance = np.linalg.norm(vector)
                if distance <= step_size:
                    return to_node
                return from_node + vector / distance * step_size

            # Initialize tree and time tracking
            tree = np.array([self.start_state])
            parent_map = {}
            start_time = time.time()

            for i in range(max_iter):
                sample = sample_free()
                nearest = nearest_node(tree, sample)
                new_node = steer(nearest, sample, step_size)

                if self.check_occupancy(nearest):
                    continue  # Skip if new node hits an obstacle

                tree = np.vstack([tree, new_node])
                parent_map[tuple(new_node)] = tuple(nearest)

                if np.linalg.norm(new_node - self.goal_state) <= 1:
                    path = [tuple(new_node)]
                    while tuple(nearest) != tuple(self.start_state):
                        nearest = parent_map[tuple(nearest)]
                        path.append(tuple(nearest))
                    path.reverse()

                    # Calculate path length
                    path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path) - 1))
                    runtime = time.time() - start_time  # Compute runtime
                    return path, path_length, runtime  # Return the correct number of values

                if i % 100 == 0:
                    print(f"Iteration {i}/{max_iter}, Tree size: {len(tree)}")

            # Return an empty path if no valid path is found
            runtime = time.time() - start_time  # Compute runtime after all iterations
            return [], 0, runtime  # Return the correct number of values
    
# Step 3 code
class Maze4D(Maze):
    """ Maze4D Class """

    def __init__(self, maze_array, start_state=None, goal_state=None, max_vel=2):
        super().__init__(maze_array, start_state, goal_state)

        self.max_vel = max_vel

        if start_state is None:
            start_state = np.array((0, 0, 0, 0))
        self.start_state = start_state
        self.start_index = self.index_from_state(self.start_state)

        if goal_state is None:
            goal_state = np.array((self.cols - 1, self.rows - 1, 0, 0))
        self.goal_state = goal_state
        self.goal_index = self.index_from_state(self.goal_state)

    def index_from_state(self, state):
        """ Gets a unique index for the state """
        velocities = self.max_vel + 1
        return state[3] * self.rows * self.cols * velocities + \
               state[2] * self.rows * self.cols + \
               state[0] * self.rows + \
               state[1]

    def state_from_index(self, state_id):
        """ Returns the state at a given index """
        velocities = self.max_vel + 1
        idx = state_id
        dy = int(np.floor(idx / (self.rows * self.cols * velocities)))
        idx -= dy * self.rows * self.cols * velocities
        dx = int(np.floor(idx / (self.rows * self.cols)))
        idx -= dx * self.rows * self.cols
        x = int(np.floor(idx / self.rows))
        y = idx % self.rows
        return (x, y, dx, dy)

    def get_neighbors(self, state_id):
        """ Returns a List of indices corresponding to neighbors of a given state in the 4D maze """
        state = self.state_from_index(state_id)
        neighbors = []

        deltas = [
            [-1, 0], [1, 0],  # horizontal
            [0, -1], [0, 1],  # vertical
            [-1, -1], [1, 1],  # diagonal
            [0, 0]  # no movement
        ]

        for delta in deltas:
            new_dx = state[2] + delta[0]
            new_dy = state[3] + delta[1]

            if new_dx < 0 or new_dx > self.max_vel or new_dy < 0 or new_dy > self.max_vel:
                continue

            if not self.check_hit(state[0:2], [new_dx, new_dy]):
                new_state = (state[0] + new_dx, state[1] + new_dy, new_dx, new_dy)
                neighbors.append(self.index_from_state(new_state))

        return neighbors

    @staticmethod
    def a_star_search_step3_i(maze, start_state, goal_state):
        """ A* search for solving the maze """
        def heuristic(state, goal_state):
            return abs(state[0] - goal_state[0]) + abs(state[1] - goal_state[1])

        def reconstruct_path(came_from, current_state):
            path = [current_state]
            while current_state in came_from:
                current_state = came_from[current_state]
                path.insert(0, current_state)
            return path

        open_set = []
        heapq.heappush(open_set, (0, tuple(start_state)))
        came_from = {}
        g_score = {tuple(start_state): 0}
        f_score = {tuple(start_state): heuristic(start_state, goal_state)}

        while open_set:
            _, current_state = heapq.heappop(open_set)

            if current_state == tuple(goal_state):
                return reconstruct_path(came_from, current_state)

            for neighbor in maze.get_neighbors(maze.index_from_state(current_state)):
                neighbor_state = maze.state_from_index(neighbor)
                tentative_g_score = g_score[tuple(current_state)] + 1

                if tuple(neighbor_state) not in g_score or tentative_g_score < g_score[tuple(neighbor_state)]:
                    came_from[tuple(neighbor_state)] = current_state
                    g_score[tuple(neighbor_state)] = tentative_g_score
                    f_score[tuple(neighbor_state)] = tentative_g_score + heuristic(neighbor_state, goal_state)
                    heapq.heappush(open_set, (f_score[tuple(neighbor_state)], tuple(neighbor_state)))

        return None  # No path found
    
    def a_star_step3_ii(self, epsilon=1):
        """
        Perform A* search with a greedy heuristic adjusted by epsilon.
        The cost is based on the heuristic scaled by epsilon for the greedy part.
        """
        start = self.start_index
        goal = self.goal_index
        
        # Priority queue (min-heap)
        open_set = []
        heappush(open_set, (0, start))  # (f_score, state_index)

        # Dictionaries for tracking costs and paths
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._greedy_heuristic(start, epsilon)}

        while open_set:
            _, current = heappop(open_set)

            # If we reach the goal, reconstruct the path
            if current == goal:
                return self._reconstruct_path(came_from, current), len(came_from)

            # Process neighbors
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1  # Assume uniform cost for grid steps

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update cost to reach neighbor
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._greedy_heuristic(neighbor, epsilon)
                    
                    # Push to priority queue
                    heappush(open_set, (f_score[neighbor], neighbor))

        return [], 0  # If no path is found

    def _greedy_heuristic(self, state_index, epsilon):
        """ Calculate the greedy heuristic to the goal, scaled by epsilon """
        state = self.state_from_index(state_index)
        goal = self.state_from_index(self.goal_index)
        # Manhattan distance heuristic for 4D space, can adjust it for more complex calculations
        return epsilon * (abs(goal[0] - state[0]) + abs(goal[1] - state[1]) + abs(goal[2] - state[2]) + abs(goal[3] - state[3]))

    def _reconstruct_path(self, came_from, current):
        """ Reconstruct path from start to goal """
        path = [self.state_from_index(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self.state_from_index(current))
        return path[::-1]  # Reverse the path

    def run_with_epsilon_decay(self, initial_epsilon=10, time_limits=[0.05, 0.25, 1]):
        for time_limit in time_limits:
            epsilon = initial_epsilon
            start_time = time.time()
            #print(f"Running with time limit {time_limit}s and initial epsilon {epsilon}")

            while True:
                #print(f"Running A* with epsilon={epsilon}")
                path, expanded_nodes = self.a_star_step3_ii(epsilon)

                elapsed_time = time.time() - start_time

                # Check for time limit
                if elapsed_time >= time_limit:
                    print(f"Time limit {time_limit}s reached.")
                    break
                
                if path:  # If a path was found
                    #print(f"Found path with {expanded_nodes} nodes expanded.")
                    #print(f"Path length: {len(path)}")
                    break
                
                # Decay epsilon
                new_epsilon = epsilon - 0.5 * (epsilon - 1)
                if new_epsilon < 1.001:
                    new_epsilon = 1
                epsilon = new_epsilon

            print(f"Initial epsilon: {epsilon}, Nodes expanded: {expanded_nodes}, Path length: {len(path)}, Elapsed time: {elapsed_time:.4f}, Time limit: {time_limit}s")

if __name__ == "__main__":
    mazes = [
        (Maze2D.from_pgm('maze1.pgm'), "Maze 1"),
        (Maze2D.from_pgm('maze2.pgm'), "Maze 2"), 
    ]

    # Setup and preprocessing for each maze
    for maze, maze_name in mazes:
        maze.start_state = (1, 1)
        maze.goal_state = (maze.cols - 1, maze.rows - 1)
        maze.start_index = maze.index_from_state(maze.start_state)
        maze.goal_index = maze.index_from_state(maze.goal_state)

    # Step 2 part i: Run A* on each maze
    for maze, maze_name in mazes:
        path = maze.a_star_step2_i()
        print(f"Found path for {maze_name}")
        maze.plot_path(path, f"A* Solution: {maze_name}")

    # Step 2 part ii: Run with epsilon decay on each maze
    for maze, maze_name in mazes:
        print(f"Testing {maze_name} with epsilon decay...")
        maze.run_with_epsilon_decay(initial_epsilon=10, time_limits=[0.05, 0.25, 1])

    # Step 2 part iii: Run RRT on each maze
    for maze, maze_name in mazes:
        print(f"Running RRT with step size 1 and max iterations 1000 on {maze_name}...")
        path, length, runtime = maze.rrt(max_iter=1000, step_size=1)
        if path:
            print(f"RRT Path Found for {maze_name}. Path Length: {length}, Runtime: {runtime} seconds.")
            maze.plot_path(path, f"{maze_name} RRT Path", runtime=runtime, path_length=length)
        else:
            print(f"RRT failed to find a path for {maze_name}")

    # Step 3
    mazes_4D = [
        (Maze4D.from_pgm('maze1.pgm'), "Maze 1 4D"),
        (Maze4D.from_pgm('maze2.pgm'), "Maze 2 4D")
    ]
    # Setup and preprocessing for each 4D maze
    for maze, maze_name in mazes_4D:
        # Test index_from_state and state_from_index for consistency
        for x in range(maze.cols):
            for y in range(maze.rows):
                for dx in range(3):
                    for dy in range(3):
                        state = (x, y, dx, dy)
                        assert maze.state_from_index(maze.index_from_state(state)) == state, f"Mapping incorrect for state: {state}"

    for maze, maze_name in mazes_4D:
        # Step 3 part i: Solve using A* search
        path = Maze4D.a_star_search_step3_i(maze, maze.start_state, maze.goal_state)
        if path:
            maze.plot_path(path, f'A* Solution: {maze_name}')
        else:
            print("No path found!")

    for maze, maze_name in mazes_4D:
        # Step 3 part ii: Run with epsilon decay on each 4D maze
        print(f"Testing {maze_name} with epsilon decay...")
        maze.run_with_epsilon_decay(initial_epsilon=1, time_limits=[0.05, 0.25, 1])

