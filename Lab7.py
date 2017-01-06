import numpy as np
import collections
import pylab as pl
from enum import IntEnum
import pandas as pd

class Cell(IntEnum):
    """
    Enumeration class for a type of cells or a breed of ants (RED and BLUE) 
    """
    EMPTY = 0
    RED = 1
    BLUE = 2
    RESOURCE = 3

    def opposite(breed):
        """
        Return an opposite of an ant's breed  
        """
        if breed == Cell.RED:
            return Cell.BLUE
        elif breed == Cell.BLUE:
            return Cell.RED

class World:
    """
    Class provides access to the simulation (simulating one step and drawing).
    It also probides the environment for ants (2D grid, ants' neighbourhood, movement of ants, resources addition, etc.)
    """
    def __init__(self, size=40, update_rate=0.001, number_of_resources=3, draw_mode=True, different=False):
        """
        Initialization of model parameters and creation of a graphical canvas. 
        """
        self.size = size # the size of the world (=grid)
        self.number_of_resources = number_of_resources # number of resource sources
        self.grid = np.zeros((size, size)) # grid, that will be drawn according to Cell class encoding
        self.smells = np.zeros((size, size)) # grid of pheromons' (="smells") concentration in the environment  
        self.red_ants = [] # a list for red ants which determines their priority (the order of actions)
        self.blue_ants = [] # the same list for blue ants
        self.red_center = (size//3, size//2) # center of red ants, where new ants appear
        self.blue_center = (2*size//3 + 1, size//2) # center of blue ants
        self.ants_positions = {} # a dictionary (hash) for quick access to the ant, given grid coordinates
        self.marker_size = 25000/size**2 # default marker size for a scatter plot
        self.update_rate = update_rate # Pause interval for animation (in seconds)
        self.iteration = 0 # iteration number (change of resource sources occurs periodically)
        self.resources = [] # list of resource sources (coordinates)
        self.different = different # Modification of simulation when red ants have double power and blue one two tries to add ants 
        self.smell_decay_rate = 20./np.sqrt(self.size) 
        # "smell" has a value between 0 and 100 and decreases by decay rate until zero each step  

        self.draw_mode = draw_mode
        # Setting a window for drawing the grid
        if self.draw_mode:
            self.fig, self.ax = pl.subplots(figsize=(10,10))
            self.ax.set_xlim([-1, self.size])
            self.ax.set_ylim([-1, self.size])
            pl.ion()
            self.fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, hspace=0.01, wspace=0.01)


    def draw(self):
        """
        Draw the current grid on the created window (in the initialization)
        """
        # Clear previous canvas
        self.ax.clear()
        # Remove axises 
        self.ax.axis('off')
        
        # Draw a rectangle with dotted borders of the world
        self.ax.plot([0, 0], [0, self.size-1], color='k', linestyle='--')
        self.ax.plot([0, self.size-1], [self.size-1, self.size -1], color='k', linestyle='--')
        self.ax.plot([0, self.size-1], [0, 0], color='k', linestyle='--')
        self.ax.plot([self.size-1, self.size-1], [0, self.size-1], color='k', linestyle='--')

        # Draw each cell: its "smell" and type 
        for i in range(self.size):
            for j in range(self.size):
                if self.smells[i][j] > 0:
                    self.ax.scatter(i,j,marker='s', c='g', s=3.5*self.marker_size, 
                        alpha=self.smells[i][j]/100.*0.9)
                if self.grid[i][j] == Cell.EMPTY: # empty cell
                    continue
                elif self.grid[i][j] == Cell.RED: # cell, occupied by a red ant
                    self.ax.scatter(i,j,marker='o', c='r', s=2.5*self.marker_size,
                     alpha=0.1 + self.ants_positions[(i,j)].power/(100. + 100*int(self.different))*0.8)
                elif self.grid[i][j] == Cell.BLUE: # cell, occupied by a blue ant
                    self.ax.scatter(i,j,marker='o', c='b', s=2.5*self.marker_size,
                     alpha=0.1 + self.ants_positions[(i,j)].power/(100. + 100*int(self.different))*0.8)
                elif self.grid[i][j] == Cell.RESOURCE: # resource (a star)
                    self.ax.scatter(i,j,marker='*', c='w', s=2*self.marker_size)
                
        # Plot sources of resources in a form of a yellow diamond
        for x, y in self.resources:
            self.ax.scatter(x,y,marker='D', c='y', s=0.7*self.marker_size, alpha=0.9)
        # Plot red and blue centers in a form of a diamond 
        self.ax.scatter(*self.red_center, marker='D', c='r', s=self.marker_size, alpha=0.8)
        self.ax.scatter(*self.blue_center, marker='D', c='b', s=self.marker_size, alpha=0.8)


    def move_ant(self, ant, position):
        """
        Request from an ant to move it in the new position
        """
        self.grid[ant.x][ant.y] = Cell.EMPTY
        del self.ants_positions[(ant.x, ant.y)]
        
        # The world is a torus, so need to assure that it moves correctly
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
        # if the new position has a resource, the ant will immediatly eat it (gain power)
        if self.grid[x][y] == Cell.RESOURCE:
            ant.eat()
        self.ants_positions[position] = ant
        self.grid[x][y] = ant.breed
        ant.x, ant.y = x, y
      

    def erase_ant(self, ant):
        """
        When an ant dies it must be removed from the world
        """
        self.grid[ant.x][ant.y] = Cell.EMPTY
        del self.ants_positions[(ant.x, ant.y)]
        ant.death = True # This will be handy for removing the ant in the priority list 


    def add_ants(self, probability=0.4,):
        """
        Each step this function tries to generate new ants.
        First, the center of ants must not be occupied. Otherwise, it won't be able to generate new ones.
        If the center is not occupied, random drawing with a defined probability determines whether to add an ant or not
        different parameters tells whether to create equal breeds or different.
        in a case of different breeds red have twice much power, but blue ones can add at most 2 ants in one turn
        """
        if self.red_center not in self.ants_positions:
            if np.random.random() < probability:
                ant = Ant(self, self.red_center, Cell.RED, different=self.different)
                self.ants_positions[self.red_center] = ant
                self.red_ants.append(ant) # The new ant goes the last to the priority list
                self.grid[self.red_center[0]][self.red_center[1]] = Cell.RED
        for center in [self.blue_center, (self.blue_center[0]+1, self.blue_center[0])][:int(self.different)+1]:
            if center not in self.ants_positions:
                    if np.random.random() < probability:
                        ant = Ant(self, center, Cell.BLUE)
                        self.ants_positions[center] = ant
                        self.blue_ants.append(ant)
                        self.grid[center[0]][center[1]] = Cell.BLUE


    def add_resources(self, probability=0.8, iterations_to_change=100):
        '''
        Each turn all sources of resources try to generate a resource with a defined probability.
        In order to generate a resource the source must not be occupied by any ant.
        Each number (iterations_to_change) of steps all resource sources are changed (they receive new locations)
        '''
        if self.iteration % iterations_to_change == 0:
            self.resources = [(np.random.randint(0, self.size), 
                np.random.choice(list(range(self.size//2-1)) + list(range(self.size//2+1, self.size))))
                for i in range(self.number_of_resources)]
        for cell in self.resources:
            if cell not in self.ants_positions:
                if np.random.random() < probability:
                    self.grid[cell[0]][cell[1]] = Cell.RESOURCE


    def add_smell(self, ant, number=100):
        '''
        An ant adds pheromones to his current location.
        This can happen when the ant found a resource. The ant goes straight (using empty cells) to his center.
        As he progresses the strength of a "smell" reduces in such way that the "smell" increases approaching to the resourse.
        If the ant stands on a cell, which already has a "smell", it cannot increase it greater than it supposed to increase.
        This allows ants to find resources using simple rule of following a maximum "smell" direction.
        '''
        x, y = ant.x, ant.y
        inc_n = number - ant.steps_after_found*(self.smell_decay_rate + 0.01)
        if self.smells[x][y] < inc_n:
            self.smells[x][y] = inc_n


    def get_environment(self, position):
        """
        An ant calls get_environment in order to get the information about his neighbourhood.
        His neighbourhood does not include his current position.
        Returns x and y indices of his neighbour cells with their type and "smell" 
        """
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
        """
        Simulates one step of a model.
        One step consists of adding new ants, resources and sequencial execution of ants' actions.
        First, all red ants execute their actions according to the priority list (the oldest is first).
        Only then all blue ants execute their actions according to their priority list.
        After all actions the "smell" of all cells decreases according to the model's parameter.
        """
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
            return died

        self.red_killed = iterate_and_action(self.red_ants)
        # Update the drawing after each team finishes their actions.
        if self.draw_mode:
            self.draw()
            pl.pause(self.update_rate)

        self.blue_killed = iterate_and_action(self.blue_ants)
        if self.draw_mode:
            self.draw()
            pl.pause(self.update_rate)
        
        # The strength of "smell" decreases until 0
        if self.smells.sum() > 0:
            self.smells -= self.smell_decay_rate
            self.smells[self.smells < 0] = 0
        self.iteration += 1


    def simulate(self, number_of_steps=100000):
        """
        simulates the whole experiment and tracks statistics through all steps
        returns pandas DataFrame with statistics
        """

        self.red_population_statistic = [] # Number of red ants in at particular time
        self.blue_population_statistic = [] # Number of blue ants in at particular time
        self.red_eat_statistic = [] # Number of eaten resources by red ants in at particular time
        self.blue_eat_statistic = [] # Number of eaten resources by blue ants in at particular time
        self.red_killed_statistic = [] # Number of killed red ants in at particular time
        self.blue_killed_statistic = [] # Number of killed blue ants in at particular time
        if self.draw_mode:
            world.draw()
            pl.pause(0.5)
        for i in range(number_of_steps):
            self.red_eat, self.blue_eat = 0, 0
            self.one_step()
            self.red_population_statistic.append(len(self.red_ants))
            self.blue_population_statistic.append(len(self.blue_ants))
            self.red_eat_statistic.append(self.red_eat)
            self.blue_eat_statistic.append(self.blue_eat)
            self.red_killed_statistic.append(self.red_killed)
            self.blue_killed_statistic.append(self.blue_killed)

        statistics = pd.DataFrame([
            self.red_population_statistic, self.blue_population_statistic,
            self.red_eat_statistic, self.blue_eat_statistic,
            self.red_killed_statistic, self.blue_killed_statistic
            ]).T
        statistics.index.name = 'Time'
        statistics.columns = ['Population_R', 'Population_B', 'CumEaten_R', 'CumEaten_B', 'CumDeaths_R', 'CumDeaths_B']
        for column in statistics.columns:
            if column.startswith('Cum'):
                statistics[column] = statistics[column].cumsum()

        return statistics






class AntState(IntEnum):
    """
    Each ant has two states of his behaviour: wandering/fighting and constructing a path (leaving "smells") to his center   
    """
    WANDER_AND_FIGHT = 1
    FOUND_RESOURCE = 2


class Ant:
    """
    Each ant is an object of this class. It contains all information about an ant.
    """
    def __init__(self, world, position, breed, different=False):
        self.world = world # world\environment reference 
        self.x, self.y = position # ant's position in the world
        self.breed = breed # ant's type (Red or Blue)
        self.death = False # if the ant dies this helps to delete him from a priority list 
        self.state = AntState.WANDER_AND_FIGHT # ant's state
        self.different = different # Whether blue and read are different in power
        if not different or self.breed == Cell.BLUE:
            self.power = 90. # ant's power/health (from 0 to 100). If it reaches 0 or below, the ant is considered dead 
        else:
            self.power = 180.
        self.steps_after_found = 0 # in a state of FOUND_RESOURCE this variable counts the number of steps from a resource.


    def choose_position(self):
        """
        In the first state of wandering/fighting each ant seeks for a resourse,
        attacks opposite ants, follows "smells" or wanders randomly otherwise.
        The second state triggers when the ant found a resource (which gives him power/health).
        In that state the ant goes directly (as the environment can afford him) to his team's center and leave "smells" every step.
        The ant also tries to choose those empty cells that have maximum "smell" in order to maintain old routes.
        If the ant has no empty cells surrounding him, he chooses a random direction. 
        This function chooses a position to move and also returns a cell type in that position. 
        """
        # Get information about his neighbourhood
        x_indices, y_indices, grid_values, smells = world.get_environment((self.x, self.y))
        # The first state 
        if self.state == AntState.WANDER_AND_FIGHT: 
            # First, try to find some resource   
            resources = grid_values == Cell.RESOURCE
            if resources.sum() > 0:
                index = np.random.choice(np.where(resources)[0])
            else:
                # If there is no resource, try to find opposite ants
                opponents = grid_values == Cell.opposite(self.breed)
                if opponents.sum() > 0:
                    index = np.random.choice(np.where(opponents)[0])
                else:
                    # If there is no opposite ants
                    empty = grid_values == Cell.EMPTY
                    if empty.sum() > 0:
                        # If there are some empty cells, choose from those who has the maximum "smell"
                        index = np.random.choice(np.where(smells == smells[empty].max())[0])
                    else:
                        # choose randomly
                        index = np.random.randint(0, grid_values.shape[0])
        # The second state, found a resource
        elif self.state == AntState.FOUND_RESOURCE:
            # Add a smell to the current location
            self.world.add_smell(self)
            # location of his center
            center = self.world.red_center if self.breed == Cell.RED else self.world.blue_center
            # If an ant can reach his center in a move, his mission is over, switch to the defaul behaviour
            if max(abs(self.x - center[0]), abs(self.y - center[1])) <= 1:
                self.state = AntState.WANDER_AND_FIGHT
                index = np.random.randint(0, grid_values.shape[0])
                # go randomly
            else:
                # The ant is not near the center
                empty = grid_values == Cell.EMPTY
                if empty.sum() > 0:
                    # If there are some empty cells in the neighbourhood
                    # compute distances to the center, according to the Minkowski distance (p=infinity)
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
                    distances_mask = distances == distances[empty].min()
                    # If some empty cells have the same distance choose those with the maximum "smell"
                    if smells[distances_mask].max() > 0:
                        index = np.random.choice(np.where(
                            (smells == smells[distances_mask].max()) & distances_mask)[0])
                    else:
                        index = np.random.choice(np.where(distances_mask)[0])
                else:
                    index = np.random.randint(0, grid_values.shape[0])
                self.steps_after_found += 1

        position = x_indices[index], y_indices[index]
        return position, grid_values[index]


    def action(self):
        """
        This function determines the interaction between ant and some chosen cell (by choose_position).
        If the chosen cell not occupied by any ant, the actor can ask the world to move it there.
        If the chosen cell is occupied by an opposite ant, it engages to a fight.
        The actor attacks the opposite ant by decreasing power/health (by 20).
        The attacker looses smaller value of power/health (by 5).   

        If the chosen cell is occupied by an ant from his own team, 
        the actor initiates a "kiss", where they are sharing their power/health.
        Two of them have the same amount of power/health (average) after this action.
        If the actor had less health than the other, the other will heal him (and vice versa) 
        """
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
        """
        Decreases power of the ant and erases it from the world if his power reached 0 or below
        """
        self.power -= number
        if self.power <= 0:
            self.world.erase_ant(self)

    def eat(self, number=40):
        """
        This function is called when the ant found a resource.
        The resource heals his power a little. The power cannot be increased above 100.
        It triggers the new state of found_resource.
        """
        self.power += number
        if self.power > 100 + 100*int(self.different):
            self.power = 100 + 100*int(self.different)
        self.state = AntState.FOUND_RESOURCE
        self.steps_after_found = 0
        if self.breed == Cell.RED:
            self.world.red_eat += 1
        else:
            self.world.blue_eat += 1

                    

np.random.seed(64925)
world = World(size=40)  
world.simulate(number_of_steps=100000)

# number_of_experiments = 40
# statistics = pd.DataFrame()
# for i in range(number_of_experiments):
#     world = World(size=40, draw_mode=False, different=True)
#     statistics = statistics.append(world.simulate(number_of_steps=10000))
# statistics.to_csv('statistics.csv')