# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 00:55:58 2018

@author: yyy
"""

# clone of the 2048 game

import poc_2048_gui
import random

# directions
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

# offsets for computing tile indices in each direction
# do not modify this dictionary

OFFSETS = {UP: (1,0), DOWN: (-1,0), LEFT: (0,1), RIGHT: (0, -1)}

def merge(line):
    # helper function that merges a single row or column in 2048
    
    # merge phase 1-1 create list of 0s with the same length as line
    align_list = [0] * len(line)
    
    # merge phase 1-2 replace 0 with non-zero elements in next available slot
    accumulate = 0
    for elem in range(0, len(line)):
        if line[elem] != 0:
            align_list[accumulate] = line[elem]
            accumulate += 1
            
    # result list intialized to zeroes
    result_list = [0] * len(line)
    
    # merge phase 2 algorithm for 2048, combining titles
    last_merged = 0
    
    # this is an iteration
    for elem in range(0, len(line)):
        if result_list[last_merged] == 0:
            result_list[last_merged] = align_list[elem]
        elif result_list[last_merged] == align_list[elem]:
            result_list[last_merged] += align_list[elem]
            last_merged += 1
        else:
            last_merged += 1
            result_list[last_merged] = align_list[elem]
            
    return result_list

class TwentyFortyEight:
    # class to run the game logic
    
    global OFFSETS
    
    def __init__(self, grid_height, grid_width):
        # stores variables and initiates new game based on reset
        self.rows = grid_height
        self.columns = grid_width
        self.grid = self.reset()
        
        # intiate directionary for each direction
        # each direction has a set of initial tiles
        self.directions = {UP: [(0,0), (0,1), (0,2), (0,3)],
                            DOWN: [(3,0), (3,1), (3,2), (3,3)],
                            LEFT: [(0,0), (1,0), (2,0), (3,0)],
                            RIGHT: [(0,3), (1,3), (2,3), (3,3)]}
    
    def reset(self):
        # reset the game so the grid is empty
        # clears grid and initiates cells to 0
        self.cells = []
        for i in range(0, self.rows):
            self.cells.append([0] * self.columns)
    
    def __str__(self):
        # return a string representation of the grid for debugging
        
        # return a string representation of current cells
        return str(self.cells)
    
    def get_grid_height(self):
        # get the height of the board
        
        # replace with your code
        if self.columns > 0:
            return self.rows
        else:
            return 0
    
    def get_grid_width(self):
        # get the width of the board
        
        # given neither are 0, returns grid width
        if self.rows > 0 and self.columns > 0:
            return self.columns
        else:
            return 0
    
    def move(self, direction):
        # move all tiles in the given direction and add a new tile if any tiles moved
        
        # add new tile
        add_new_tile = False
        
        # logic of 2048, moves and merges tiles across the board
        initial = self.directions[direction]
        
        for init in range(0, len(initial)):
            offset_index = []
            tempset = []
            
            # determine if direction is up or down to figure out whether
            # to use height or width
            if direction == UP or direction == DOWN:
                
                # creates a list of indices for the initial index based on offset
                for i in range(0, self.rows):
                    offset_index.append((initial[init][0] + i * OFFSETS[direction][0],
                                         initial[init][0] + i * OFFSETS[direction][1]))
                    
                # get values of all tiles at the indices
                for i in offset_index:
                    tempset.append(self.get_tile(i[0], i[1]))
                
                # merge tempset
                merged_set = merge(tempset)
                
                # put merged values back into selected indices
                for i in range(0, len(offset_index)):
                    if add_new_tile == False:
                        if (self.get_tile(offset_index[i][0], offset_index[i][1]) != merged_set[i]):
                            add_new_tile = True
                        
                    self.set_tile(offset_index[i][0], offset_index[i][1], merged_set[i])
            
            # repeat for other directions, except use columns
            if direction == LEFT or direction == RIGHT:

                # creates a list of indices for the inital index based on offset
                for i in range(0, self.columns):
                    offset_index.append((initial[init][0] + i * OFFSETS[direction][0],
                                         initial[init][1] + i * OFFSETS[direction][1]))

                # get values of all the tiles at the indices
                for i in offset_index:
                    tempset.append(self.get_tile(i[0], i[1]))
                
                # merge tempset
                merged_set = merge(tempset)

                # put merged values back into selected indices
                for i in range(0, len(offset_index)):
                    if add_new_tile == False:
                        if self.get_tile(offset_index[i][0], offset_index[i][1]) != merged_set[i]:
                            add_new_tile = True
                    self.set_tile(offset_index[i][0], offset_index[i][1], merged_set[i])

        if add_new_tile == True:
            self.new_tile()
        
        
    def new_tile(self):
        """
        Create a new tile in a randomly selected empty 
        square.  The tile should be 2 90% of the time and
        4 10% of the time.
        """
        # generator for 2s and 4s
        number = random.randrange(0, 9)
        tile_value = 2
        if number > 0:
            tile_value = 2
        else:
            tile_value = 4

        # generate rows
        row_numbers = []
        for i in range(0, self.rows):
            row_numbers.append(i)
        random.shuffle(row_numbers)
        for i in range(0, self.rows):
            row_index = row_numbers.pop()
            if 0 in self.cells[row_index]:
                self.set_tile(row_index, self.cells[row_index].index(0), tile_value)
                break
        
        
    def set_tile(self, row, col, value):
        """
        Set the tile at position row, col to have the given value.
        """        
        # assigns a cell a value
        self.cells[row][col] = value

    def get_tile(self, row, col):
        """
        Return the value of the tile at position row, col.
        """        
        # returns index based on given row and column
        return self.cells[row][col]
 
    
poc_2048_gui.run_gui(TwentyFortyEight(4, 4))



































