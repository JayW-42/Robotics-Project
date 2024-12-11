###################################################################################################################################
# Clearing the Terminal before the code begins
###################################################################################################################################
import os

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

clear()

# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import random

# Define the Grid class
class Grid:
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((size, size), dtype=object)

    def is_cell_empty(self, x, y):
        return self.grid[x, y] == 0

# Define the Robot class
class Robot:
    def __init__(self, x, y, goal_x, goal_y, robot_type):
        self.x = x
        self.y = y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.robot_type = robot_type

    def distance_to_goal(self):
        return abs(self.x - self.goal_x) + abs(self.y - self.goal_y)

    def move(self, direction):
        if direction == "up" and self.x > 0:
            self.x -= 1
        elif direction == "down" and self.x < 4:
            self.x += 1
        elif direction == "left" and self.y > 0:
            self.y -= 1
        elif direction == "right" and self.y < 4:
            self.y += 1

# Subclasses for specific robot types
class Quadrotor(Robot):
    def __init__(self, x, y, goal_x, goal_y):
        super().__init__(x, y, goal_x, goal_y, "Quadrotor")

class DifferentialDrive(Robot):
    def __init__(self, x, y, goal_x, goal_y):
        super().__init__(x, y, goal_x, goal_y, "DifferentialDrive")

class Humanoid(Robot):
    def __init__(self, x, y, goal_x, goal_y):
        super().__init__(x, y, goal_x, goal_y, "Humanoid")

# Randomly sample robots and goals
def initialize_robots(grid, num_robots=10):
    robots = []
    positions = set()

    for _ in range(num_robots):
        while True:
            x, y = random.randint(0, grid.size - 1), random.randint(0, grid.size - 1)
            if (x, y) not in positions:
                positions.add((x, y))
                break

        while True:
            goal_x, goal_y = random.randint(0, grid.size - 1), random.randint(0, grid.size - 1)
            if (goal_x, goal_y) not in positions:
                positions.add((goal_x, goal_y))
                break

        # Randomly assign robot type
        robot_type = random.choice([Quadrotor, DifferentialDrive, Humanoid])
        robots.append(robot_type(x, y, goal_x, goal_y))

    return robots

# Visualize the grid
def visualize(grid, robots):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, grid.size - 0.5)
    ax.set_ylim(-0.5, grid.size - 0.5)
    ax.set_xticks(range(grid.size))
    ax.set_yticks(range(grid.size))
    ax.grid(True)
    ax.set_title("Robot Grid Navigation")

    colors = {"Quadrotor": "red", "DifferentialDrive": "green", "Humanoid": "blue"}

    for robot in robots:
        # Plot robot
        ax.plot(robot.y, robot.x, "o", color=colors[robot.robot_type], label=robot.robot_type)
        # Plot goal
        ax.plot(robot.goal_y, robot.goal_x, "s", color=colors[robot.robot_type], alpha=0.6)
        # Connect robot to goal
        ax.plot([robot.y, robot.goal_y], [robot.x, robot.goal_x], "--", color=colors[robot.robot_type], alpha=0.5)

    # Avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.show()

# Main simulation setup
grid = Grid(size=5)
robots = initialize_robots(grid)
visualize(grid, robots)


# Simulate robot movements
def simulate(grid, robots, steps=10):
    def choose_direction(robot):
        """Choose the direction that minimizes the Manhattan distance to the goal."""
        options = []
        if robot.x > 0:  # Up
            options.append(("up", abs(robot.x - 1 - robot.goal_x) + abs(robot.y - robot.goal_y)))
        if robot.x < grid.size - 1:  # Down
            options.append(("down", abs(robot.x + 1 - robot.goal_x) + abs(robot.y - robot.goal_y)))
        if robot.y > 0:  # Left
            options.append(("left", abs(robot.x - robot.goal_x) + abs(robot.y - 1 - robot.goal_y)))
        if robot.y < grid.size - 1:  # Right
            options.append(("right", abs(robot.x - robot.goal_x) + abs(robot.y + 1 - robot.goal_y)))
        
        # Choose the direction with the minimum distance
        options.sort(key=lambda x: x[1])  # Sort by Manhattan distance
        return options[0][0] if options else None

    for step in range(steps):
        # Store next positions and handle collisions
        proposed_positions = {}
        for robot in robots:
            if (robot.x, robot.y) == (robot.goal_x, robot.goal_y):
                continue  # Skip if already at the goal

            direction = choose_direction(robot)
            if direction:
                # Predict the new position
                new_x, new_y = robot.x, robot.y
                if direction == "up":
                    new_x -= 1
                elif direction == "down":
                    new_x += 1
                elif direction == "left":
                    new_y -= 1
                elif direction == "right":
                    new_y += 1
                
                proposed_positions[(new_x, new_y)] = proposed_positions.get((new_x, new_y), []) + [robot]

        # Resolve conflicts at each position
        for position, robots_at_pos in proposed_positions.items():
            if len(robots_at_pos) == 1:  # No conflict
                robots_at_pos[0].x, robots_at_pos[0].y = position
            else:  # Resolve conflict
                robots_at_pos.sort(key=lambda r: r.distance_to_goal(), reverse=True)
                # Robot with the higher distance moves, others wait
                moving_robot = robots_at_pos[0]
                moving_robot.x, moving_robot.y = position

        # Visualize the state after this step
        print(f"Step {step + 1}")
        visualize(grid, robots)

# Run the simulation
simulate(grid, robots, steps=10)


# Generalize to an nxn grid (with direct input as a function parameter)
def generalized_simulation_fixed(n):
    if not (3 <= n <= 10):
        raise ValueError("Grid size must be between 3 and 10.")

    # Create nxn grid and initialize robots
    grid = Grid(size=n)
    num_robots = 2 * n  # Number of robots is 2 * n
    robots = initialize_robots(grid, num_robots=num_robots)

    # Visualize initial state and simulate movement
    print(f"Simulating a {n}x{n} grid with {num_robots} robots...")
    visualize(grid, robots)
    simulate(grid, robots, steps=10)

# Specify the grid size (example: 6)
grid_size = 5
generalized_simulation_fixed(grid_size)
