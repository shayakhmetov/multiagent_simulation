import numpy as np
import collections, itertools
import time 
import pylab as pl
from enum import IntEnum

class Cell(IntEnum):
    EMPTY = 0
    RED = 1
    BLUE = 2
    RESOURCE = 3

    def opposite(breed):
        if breed == Cell.RED:
            return Cell.BLUE
        elif breed == Cell.BLUE:
            return Cell.RED

class World:
    def __init__(self, size, update_rate=0.01, number_of_resources=3):
        self.size = size
        self.number_of_resources = number_of_resources
        self.grid = np.zeros((size, size))
        self.smells = np.zeros((size, size))
        self.red_ants = []
        self.blue_ants = []
        self.red_center = (size//3, size//2)
        self.blue_center = (2*size//3 + 1, size//2)
        self.ants_positions = {}
        self.marker_size = 25000/size**2
        self.update_rate = update_rate
        self.iteration = 0
        self.resources = []
        self.smell_decay_rate = 80./self.size
        self.fig, self.ax = pl.subplots(figsize=(9,9))
        self.ax.set_xlim([-1, self.size])
        self.ax.set_ylim([-1, self.size])
        pl.ion()
        # for i in range(self.size):
        #     for j in range(self.size):
        #         self.ax.scatter(i,j,marker='.', c='w', s=self.marker_size)
        # self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox) 


    def draw(self):
        self.ax.clear()
        # self.ax.grid()
        self.ax.axis('off')
        # self.fig.canvas.restore_region(self.background)
        
        self.ax.plot([0, 0], [0, self.size-1], color='k', linestyle='--')
        self.ax.plot([0, self.size-1], [self.size-1, self.size -1], color='k', linestyle='--')
        self.ax.plot([0, self.size-1], [0, 0], color='k', linestyle='--')
        self.ax.plot([self.size-1, self.size-1], [0, self.size-1], color='k', linestyle='--')

        self.ax.scatter(*self.red_center, marker='D', c='r', s=self.marker_size, alpha=0.8)
        self.ax.scatter(*self.blue_center, marker='D', c='b', s=self.marker_size, alpha=0.8)

        for i in range(self.size):
            for j in range(self.size):
                if self.smells[i][j] > 0:
                    self.ax.scatter(i,j,marker='s', c='g', s=3.4*self.marker_size, alpha=self.smells[i][j]/100.*0.9)
                if self.grid[i][j] == Cell.EMPTY:
                    continue
                    # self.ax.scatter(i,j,marker='.', c='w', s=self.marker_size)
                elif self.grid[i][j] == Cell.RED:
                    self.ax.scatter(i,j,marker='o', c='r', s=2.5*self.marker_size, alpha=0.1 + self.ants_positions[(i,j)].power/100.*0.8)
                elif self.grid[i][j] == Cell.BLUE:
                    self.ax.scatter(i,j,marker='o', c='b', s=2.5*self.marker_size, alpha=0.1 + self.ants_positions[(i,j)].power/100.*0.8)
                elif self.grid[i][j] == Cell.RESOURCE:
                    self.ax.scatter(i,j,marker='*', c='y', s=1.5*self.marker_size)
                
        for x, y in self.resources:
            self.ax.scatter(x,y,marker='D', c='y', s=self.marker_size, alpha=0.8)

    def move_ant(self, ant, position):
        self.grid[ant.x][ant.y] = Cell.EMPTY
        del self.ants_positions[(ant.x, ant.y)]
        
        x, y = position
        if x < 0:
            x += self.size
        elif x >= self.size:
            x -= self.size
        if y < 0:
            y += self.size
        elif y >= self.size:
            y -= self.size
        position = (x, y)
        if self.grid[x][y] == Cell.RESOURCE:
            ant.eat()
        self.ants_positions[position] = ant
        self.grid[x][y] = ant.breed
        ant.x, ant.y = x, y
      

    def erase_ant(self, ant):
        self.grid[ant.x][ant.y] = Cell.EMPTY
        del self.ants_positions[(ant.x, ant.y)]
        ant.death = True 


    def add_ants(self, probability=0.4):
        if self.red_center not in self.ants_positions:
            if np.random.random() < probability:
                ant = Ant(self, self.red_center, Cell.RED)
                self.ants_positions[self.red_center] = ant
                self.red_ants.append(ant)
                self.grid[self.red_center[0]][self.red_center[1]] = Cell.RED
        if self.blue_center not in self.ants_positions:
            if np.random.random() < probability:
                ant = Ant(self, self.blue_center, Cell.BLUE)
                self.ants_positions[self.blue_center] = ant
                self.blue_ants.append(ant)
                self.grid[self.blue_center[0]][self.blue_center[1]] = Cell.BLUE


    def add_resources(self, probability=0.8, iterations_to_change=60):
        if self.iteration % iterations_to_change == 0:
            self.resources = [(np.random.randint(0, self.size), 
                np.random.choice(list(range(self.size//2-1)) + list(range(self.size//2+1, self.size))))
                for i in range(self.number_of_resources)]
        for cell in self.resources:
            if cell not in self.ants_positions:
                if np.random.random() < probability:
                    self.grid[cell[0]][cell[1]] = Cell.RESOURCE


    def add_smell(self, ant, number=100):
        x, y = ant.x, ant.y
        self.smells[x][y] += number - ant.steps_after_found*self.smell_decay_rate
        if self.smells[x][y] > 100:
            self.smells[x][y] = 100


    def get_environment(self, position):
        x,y = position
        x_minus = x - 1
        if x_minus < 0:
            x_minus += self.size
        x_plus = x + 1
        if x_plus >= self.size:
            x_plus -= self.size
        y_minus = y - 1
        if y_minus < 0:
            y_minus += self.size
        y_plus = y + 1
        if y_plus >= self.size:
            y_plus -= self.size
        x_indices = [x_minus]*3 +  [x]*3 + [x_plus]*3
        y_indices = [y_minus, y, y_plus]*3
        del x_indices[4], y_indices[4]

        return (np.array(x_indices), np.array(y_indices),
            self.grid[x_indices, y_indices], self.smells[x_indices, y_indices])


    def one_step(self):
        self.add_ants()
        self.add_resources()
        def iterate_and_action(ants):
            died = 0
            for i in range(len(ants)):
                if ants[i-died].death:
                    del ants[i-died]
                    died += 1
                else:
                    ants[i-died].action()
        iterate_and_action(self.red_ants)
        self.draw()
        pl.pause(self.update_rate)
        iterate_and_action(self.blue_ants)
        self.draw()
        pl.pause(self.update_rate)
        if self.smells.sum() > 0:
            self.smells -= self.smell_decay_rate
            self.smells[self.smells < 0] = 0
        self.iteration += 1



class AntState(IntEnum):
    WANDER_AND_FIGHT = 1
    FOUND_RESOURCE = 2


class Ant:
    def __init__(self, world, position, breed):
        self.world = world
        self.x, self.y = position
        self.breed = breed
        self.death = False
        self.state = AntState.WANDER_AND_FIGHT
        self.power = 90.
        self.steps_after_found = 0


    def choose_position(self, smell_threshold=10):
        x_indices, y_indices, grid_values, smells = world.get_environment((self.x, self.y))
        
        if self.state == AntState.WANDER_AND_FIGHT:    
            resources = grid_values == Cell.RESOURCE 
            if resources.sum() > 0:
                index = np.random.choice(np.where(resources)[0])
            else:
                opponents = grid_values == Cell.opposite(self.breed)
                if opponents.sum() > 0:
                    index = np.random.choice(np.where(opponents)[0])
                else:
                    empty = grid_values == Cell.EMPTY
                    if empty.sum() > 0:
                        if smells.max() <= smell_threshold:
                            index = np.random.choice(np.where(empty)[0])
                        else:
                            index = np.random.choice(np.where(smells == smells[empty].max())[0])
                    else:
                        index = np.random.randint(0, grid_values.shape[0])
        elif self.state == AntState.FOUND_RESOURCE:
            self.world.add_smell(self)
            center = self.world.red_center if self.breed == Cell.RED else self.world.blue_center
            if max(abs(self.x - center[0]), abs(self.y - center[1])) <= 1:
                self.state = AntState.WANDER_AND_FIGHT
                index = np.random.randint(0, grid_values.shape[0])
            else:
                empty = grid_values == Cell.EMPTY
                if empty.sum() > 0:
                    distances = np.maximum(
                        np.minimum(
                            np.abs(x_indices - center[0]),
                            np.abs(x_indices + self.world.size - center[0]), 
                            np.abs(x_indices - self.world.size - center[0])),
                        np.minimum(
                            np.abs(y_indices - center[1]),
                            np.abs(y_indices + self.world.size - center[1]), 
                            np.abs(y_indices - self.world.size - center[1]))
                        )
                    index = np.random.choice(np.where(distances == distances[empty].min())[0])
                else:
                    index = np.random.randint(0, grid_values.shape[0])
                self.steps_after_found += 1

        position = x_indices[index], y_indices[index]
        return position, grid_values[index]


    def action(self):
        position, cell = self.choose_position()
        if position != (self.x, self.y):
            if cell not in {Cell.RED, Cell.BLUE}:
                self.world.move_ant(self, position)
            else:
                ant = self.world.ants_positions[position]
                if cell == Cell.opposite(self.breed):
                    ant.decrease_power()
                    self.decrease_power(number=5)
                    if self.death:
                        return
                    if ant.death:
                        self.world.move_ant(self, position)
                elif cell == self.breed:
                    average_power = (self.power + ant.power)/2
                    self.power = average_power
                    ant.power = average_power 

    def decrease_power(self, number=20):
        self.power -= number
        if self.power <= 0:
            self.world.erase_ant(self)

    def eat(self, number=40):
        self.power += number
        if self.power > 100:
            self.power = 100
        self.state = AntState.FOUND_RESOURCE
        self.steps_after_found = 0

                    

np.random.seed(251192)
world = World(40)
world.draw()
pl.pause(0.01)
while True:
    world.one_step()
