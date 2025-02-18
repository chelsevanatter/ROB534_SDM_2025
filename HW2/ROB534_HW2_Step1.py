__author__ = 'Chelse VanAtter'

import gzip
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random
from random import randint
from RobotClass import Robot
from GameClass import Game
from RandomNavigator import RandomNavigator
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork

__author__ = 'Chelse VanAtter'

import gzip
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random
from random import randint
from RobotClass import Robot
from GameClass import Game
from RandomNavigator import RandomNavigator
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork

class GreedyNavigator:
    def __init__(self):
        self.uNet = None  # Neural network for map estimation
        self.classNet = None  # Neural network for digit classification
        self.prediction = None  # Stores last digit prediction
        self.path_history = []  # Track robot's movement

    def setNetworks(self, uNet, classNet):
        # Assign the neural networks after object creation 
        self.uNet = uNet
        self.classNet = classNet
    
    def getAction(self, robot, exploredMap):
        if not self.uNet or not self.classNet:
            raise ValueError("Both neural networks must be initialized before calling getAction.")
        
        mask = (exploredMap != 128).astype(int)  # Identifies explored areas
        estimated_world = self.uNet.runNetwork(exploredMap, mask)  # Predict missing areas
        digit_probs = self.classNet.runNetwork(estimated_world)
        self.prediction = digit_probs.argmax()
        #print(f"Digit identified: {self.prediction}")
        
        x, y = robot.getLoc()
        self.path_history.append((x, y))  # Log movement
        
        possible_moves = {'left': (x - 1, y), 'right': (x + 1, y), 'up': (x, y - 1), 'down': (x, y + 1)}
        
        move_options = []
        
        for move, (nx, ny) in possible_moves.items():
            if robot.checkValidMove(move):
                if exploredMap[nx, ny] == 128:  # Unvisited location
                    info_gain = self.calculateInfoGain(estimated_world, nx, ny)
                    move_score = random.uniform(0.5, 0.7) * info_gain
                    move_options.append((move, move_score))
        
        if move_options:
            move_options.sort(key=lambda x: x[1], reverse=True)
            best_choice = move_options[0][0] if random.random() < 0.8 else random.choice(move_options[:min(3, len(move_options))])[0]
        else:
            best_choice = random.choice(list(possible_moves.keys()))  # If no good move, pick a random one
        
        if best_choice in possible_moves and robot.checkValidMove(best_choice):
            new_x, new_y = possible_moves[best_choice]
            exploredMap[new_x, new_y] = 0
        
        return best_choice
    
    def calculateInfoGain(self, estimated_world, x, y):
        """ Calculate information gain at a location """
        return np.abs(estimated_world[x, y] - 0)


# Simulation Setup
map_data = Map()
all_scores = []

for i in range(10):
    print(f"Running Simulation for Map Number: {i + 1}")
    robot = Robot(0, 0)
    navigator = GreedyNavigator()
    
    uNet = WorldEstimatingNetwork()
    classNet = DigitClassificationNetwork()
    navigator.setNetworks(uNet, classNet)
    
    game = Game(map_data.map, map_data.number, navigator, robot)
    last_prediction = None
    consistent_count = 0
    
    for step in range(1000):
        reached_goal = game.tick()
        
        if navigator.prediction == last_prediction:
            consistent_count += 1
        else:
            consistent_count = 0
        
        if step %100 == 0:
            print(f"Step {step}: Robot at {robot.getLoc()}, Score: {game.getScore()}, Prediction: {navigator.prediction}")
        
        last_prediction = navigator.prediction
        
        if reached_goal:
            print(f"Goal reached in {game.getIteration()} steps!")
            break
    
    print(f"Final Score: {game.score}")
    all_scores.append(game.score)
    
    estimated_mask = (game.exploredMap != 128).astype(int)
    estimated_world = uNet.runNetwork(game.exploredMap, estimated_mask)
    
    color_map = np.stack((estimated_world,) * 3, axis=-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(color_map, cmap='gray')
    
    if navigator.path_history:
        x_vals, y_vals = zip(*navigator.path_history)
        ax.plot(y_vals, x_vals, color='red', linewidth=1.5, marker='o', markersize=2, label="Path")
        ax.scatter([y_vals[-1]], [x_vals[-1]], color='blue', label="Final Position", s=40)
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f"Path on Map {i + 1}")
    plt.legend()
    plt.show()
    
    predicted_digit = classNet.runNetwork(estimated_world).argmax()
    print(f"Final Predicted Digit: {predicted_digit}")
    
    map_data.getNewMap()
    print(f"Simulation for Map Number {i + 1} Complete\n")

print(f"All Scores: {all_scores}")
print(f"Average Score: {sum(all_scores) / len(all_scores)}")
