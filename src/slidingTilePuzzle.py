from __future__ import division
from __future__ import print_function
# import resource
import sys
import math
import time
import queue as Q


#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """

        """ 
        new_config = self.cofig changed to new_config = self.config[:] because 
        python lists are mutable and previous line of code was creating reference of self.config.
        self.config[:] will create a shallow copy of self.config instead of creating reference to self.config.
        """
        new_config = self.config[:]
        if self.blank_index >= 3:
            new_config[self.blank_index],new_config[self.blank_index - 3] = new_config[self.blank_index - 3],new_config[self.blank_index]
            return new_config
      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        new_config = self.config[:]
        if self.blank_index < 6:
            new_config[self.blank_index],new_config[self.blank_index + 3] = new_config[self.blank_index + 3],new_config[self.blank_index]
            return new_config
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        new_config = self.config[:]
        if self.blank_index % 3 != 0:
            new_config[self.blank_index],new_config[self.blank_index - 1] = new_config[self.blank_index - 1],new_config[self.blank_index]
            return new_config

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        new_config = self.config[:]
        if self.blank_index % 3 != 2:
            new_config[self.blank_index],new_config[self.blank_index + 1] = new_config[self.blank_index + 1],new_config[self.blank_index]
            return new_config
      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children
    
    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __eq__(self, other):
        return self.cost == other.cost

    def __ne__(self, other):
        return self.cost != other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __ge__(self, other):
        return self.cost >= other.cost
# Function that Writes to output.txt

# The Class to keep track of stats
class Statistics(object):
    def __init__(self, nodes_expanded = 0, max_search_depth = 0, running_time=0.0, memory=0.0):
        self.nodes_expanded = nodes_expanded
        self.max_search_depth = max_search_depth
        self.running_time = running_time
        self.memory = memory

### Students need to change the method to have the corresponding parameters
def writeOutput(solution_state: PuzzleState, stats: Statistics):
    ### Student Code Goes here
    path_to_goal = []
    current_state = solution_state
    while current_state.parent:
        path_to_goal.insert(0, current_state.action)
        current_state = current_state.parent
    
    with open('output.txt', 'w') as f:
        f.write("path_to_goal: {}\n".format(path_to_goal))
        f.write("cost_of_path: {}\n".format(solution_state.cost))
        f.write("nodes_expanded: {}\n".format(stats.nodes_expanded))
        f.write("search_depth: {}\n".format(solution_state.cost))
        f.write("max_search_depth: {}\n".format(stats.max_search_depth))
        f.write("running_time: {}\n".format(stats.running_time))
        f.write("max_ram_usage: {}\n".format(stats.memory))

        # You would need to compute the running_time and max_ram_usage and write them here as well.

def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###
    frontier = [initial_state]
    explored = set()
    frontier_set = {tuple(initial_state.config)}
    stats = Statistics()
    while frontier:
        state = frontier.pop(0)
        frontier_set.remove(tuple(state.config))
        stats.max_search_depth = max(stats.max_search_depth, state.cost)
        # Goal test
        if test_goal(state):
            stats.max_search_depth += 1
            return (state, stats)
        stats.nodes_expanded += 1 
        explored.add(tuple(state.config))
        
        # Child states generation from the expand function
        for child_config in state.expand():
            if tuple(child_config) not in explored and tuple(child_config) not in frontier_set:
                move = calc_move(state, child_config)
                child_state = PuzzleState(child_config, state.n, parent=state, action=move, cost=state.cost + 1)
                frontier.append(child_state)
                frontier_set.add(tuple(child_config))
    # No solution found
    return (None, stats)

def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    stack = [initial_state]
    stack_set = {tuple(initial_state.config)}  # Keep track of states in stack
    explored = set()
    stats = Statistics()
    while stack:
        state = stack.pop()
        stack_set.remove(tuple(state.config))
        stats.max_search_depth = max(stats.max_search_depth, state.cost)
        # Test the state as soon as it's popped off the stack
        if test_goal(state):
            return (state, stats)

        stats.nodes_expanded += 1  # Increment nodes expanded count
        explored.add(tuple(state.config))

        # Expand the neighbors (children) of the state
        for child_config in reversed(state.expand()):
            # Use the set for efficient membership test
            if tuple(child_config) not in explored and tuple(child_config) not in stack_set:
                move = calc_move(state, child_config)
                child_state = PuzzleState(child_config, state.n, parent=state, action=move, cost=state.cost + 1)
                stack.append(child_state)
                stack_set.add(tuple(child_config))  # Update our stack set

    return (None, stats)

def A_star_search(initial_state):
    """A * search"""
    ### STUDENT CODE GOES HERE ###
    pq = Q.PriorityQueue()
    pq.put((calculate_total_cost(initial_state), initial_state))
    pq_set = {tuple(initial_state.config)}
    explored = set()
    stats = Statistics()
    while not pq.empty():
        _, state = pq.get()
        if tuple(state.config) in pq_set:
            pq_set.remove(tuple(state.config))

        if tuple(state.config) in explored:
            continue
        explored.add(tuple(state.config))
        
        stats.max_search_depth = max(stats.max_search_depth, state.cost)
        
        if test_goal(state):
            return (state, stats)
        
        stats.nodes_expanded += 1

        for child_config in state.expand():
            child_state = PuzzleState(child_config, state.n, parent=state, action=calc_move(state, child_config), cost = state.cost + 1)
            total_cost = child_state.cost + calculate_total_cost(child_state)
            if tuple(child_config) not in explored and tuple(child_config) not in pq_set:
                pq.put((total_cost, child_state))
                pq_set.add(tuple(child_config))
            # elif tuple(child_config) in pq_set and tuple(child_config) not in explored:
            #     pq.put((total_cost, child_state))
    return (None, stats)

def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    total_cost = 0
    for idx, value in enumerate(state.config):
        if value != 0:
            total_cost += calculate_manhattan_dist(idx, value, state.n)
    return total_cost

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    goal_idx = value
    return abs(idx // n - goal_idx // n) + abs(idx % n - goal_idx % n)

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    goal_config = list(range(len(puzzle_state.config)))
    return puzzle_state.config == goal_config

"""
Function to determine the move made from initial 
state to child state (had to write this because 
expand method does not have the capability to give move and 
modifying expand method is restricted.
"""
def calc_move(state, child_config):
    move = ""
    if state.blank_index - child_config.index(0) == state.n:
        move = "Up"
    elif child_config.index(0) - state.blank_index == state.n:
        move = "Down"
    elif state.blank_index - child_config.index(0) == 1:
        move = "Left"
    else:
        move = "Right"
    return move

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    
    begin_state = [8,6,4,2,1,3,5,7,0]
    board_size = 3
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    search_mode = "ast" # Set the search mode to use different search techniques. (ast, dfs, bfs)

    # dfsstartram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if search_mode == "bfs": 
        solution, stats = bfs_search(hard_state)
    elif search_mode == "dfs": 
        solution, stats = dfs_search(hard_state)
    elif search_mode == "ast": 
        solution, stats = A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")
        
    # dfsram = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - dfsstartram)/(2**20)
    end_time = time.time()
    print("Program completed in %.3f second(s)"%(end_time-start_time))
    solution.display()
    stats.running_time = end_time-start_time
    # stats.memory = dfsram
    writeOutput(solution, stats)

if __name__ == '__main__':
    main()
