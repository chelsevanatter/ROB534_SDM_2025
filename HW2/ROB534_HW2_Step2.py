import numpy as np
import random
import time
import heapq
from RobotClass import Robot
from GameClass import Game
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork

class AStarNavigator:
    def __init__(self):
        self.uNet = None  # Neural network for map estimation
        self.classNet = None  # Neural network for digit classification
        self.prediction = None  # Stores last digit prediction
        self.visited = set()
    
    def setNetworks(self, uNet, classNet):
        self.uNet = uNet
        self.classNet = classNet
    
    def heuristic(self, x, y, goal):
        return abs(x - goal[0]) + abs(y - goal[1])  # Manhattan distance heuristic
    
    def getValidMoves(self, robot, x, y, exploredMap):
        moves = {'left': (x - 1, y), 'right': (x + 1, y), 'up': (x, y - 1), 'down': (x, y + 1)}
        valid_moves = []
        
        for move, (nx, ny) in moves.items():
            if robot.checkValidMove(move) and (nx, ny) not in self.visited:
                valid_moves.append((move, (nx, ny)))
        
        return valid_moves
    
    def getAction(self, robot, exploredMap):
        x, y = robot.getLoc()
        self.visited.add((x, y))
        
        estimated_world = self.uNet.runNetwork(exploredMap, (exploredMap != 128).astype(int))
        digit_probs = self.classNet.runNetwork(estimated_world)
        self.prediction = digit_probs.argmax()
        goal = self.getGoal(self.prediction)
        
        priority_queue = []
        for move, (nx, ny) in self.getValidMoves(robot, x, y, exploredMap):
            score = self.heuristic(nx, ny, goal) + random.uniform(0.1, 0.3)
            heapq.heappush(priority_queue, (score, move))
        
        if priority_queue:
            return heapq.heappop(priority_queue)[1]
        else:
            return random.choice(['left', 'right', 'up', 'down'])  # Fallback move
    
    def getGoal(self, prediction):
        if prediction in [0, 1, 2]:
            return (0, 27)
        elif prediction in [3, 4, 5]:
            return (27, 27)
        else:
            return (27, 0)

map_data = Map()
all_scores = []
times = []

for i in range(10):
    print(f"Running Simulation for Map Number: {i + 1}")
    robot = Robot(0, 0)
    navigator = AStarNavigator()
    
    uNet = WorldEstimatingNetwork()
    classNet = DigitClassificationNetwork()
    navigator.setNetworks(uNet, classNet)
    
    game = Game(map_data.map, map_data.number, navigator, robot)
    
    start_time = time.time()
    time_limit = 15 * 60  # 15 minutes in seconds
    
    for step in range(1000):
        if time.time() - start_time > time_limit:
            print("Terminating simulation: exceeded 15-minute limit.")
            break
        
        reached_goal = game.tick()
        if reached_goal:
            break
    
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)
    
    print(f"Final Score: {game.score}, Time Taken: {elapsed_time:.2f} seconds")
    all_scores.append(game.score)
    map_data.getNewMap()

print(f"All Scores: {all_scores}")
print(f"Average Score: {sum(all_scores) / len(all_scores):.2f}")
print(f"Total Time Taken: {sum(times):.2f} seconds")
print(f"Average Time Per Solution: {sum(times) / len(times):.2f} seconds")
