import numpy
import time

from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum

def calc_min_dist_from_fruit(state: GameState, player_index: int) ->float:
    min=-1
    for location in state.fruits_locations:
        dist = manhattan_distance(state.snakes[player_index].head,location)
        if min == (-1):
            min=dist
        elif dist<min:
            min=dist
    return min


def manhattan_distance (x1 : tuple, x2: tuple):
    return abs( x1[0] - x2[0] ) + abs( x1[1] - x2[1] )

def sumOfOpponenetsLen(state: GameState, player_index):
    len = 0
    for snake in state.snakes:
        if snake.index != player_index:
            len += snake.length
    return len


def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    # Insert your code here...

    if not state.snakes[player_index].alive:
        return state.snakes[player_index].length
    else:
        my_length = state.snakes[player_index].length

        sum_of_all_lengths = sumOfOpponenetsLen(state, player_index)

        # min distance from fruit
        next_fruit_dist = calc_min_dist_from_fruit(state, player_index)

        discount_factor = 0.5
        max_possible_fruits = len(state.fruits_locations) + sum([s.length for s in state.snakes
                                                                 if s.index != player_index and s.alive])
        turns_left = (state.game_duration_in_turns - state.turn_number)
        max_possible_fruits = min(max_possible_fruits, turns_left)
        optimistic_future_reward = discount_factor * (1 - discount_factor ** max_possible_fruits) / (
                    1 - discount_factor)

        heuristic_value = my_length + (1/next_fruit_dist) + (1/sum_of_all_lengths) + optimistic_future_reward

        return heuristic_value

    pass


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """
    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """
        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    counter = 0
    time_sum = 0

    def rb_minimax (self, depth, tree_node: TurnBasedGameState) -> float:
        if tree_node.game_state.is_terminal_state or depth == 0:
            return heuristic(tree_node.game_state, self.player_index)
        #if tree_node.agent_action != None:
        if tree_node.turn == self.Turn.OPPONENTS_TURN:
            curr_min = np.inf
            for opponents_actions in tree_node.game_state.get_possible_actions_dicts_given_action(tree_node.agent_action,
                                                                                   player_index=self.player_index):
                opponents_actions[self.player_index] = tree_node.agent_action
                next_state = get_next_state(tree_node.game_state, opponents_actions)
                oppnents_node = self.TurnBasedGameState(next_state, agent_action=None)
                val = self.rb_minimax(depth-1, oppnents_node)
                if(val < curr_min):
                    curr_min = val
            return curr_min
        else:
            curr_max = -np.inf
            for action in tree_node.game_state.get_possible_actions(player_index=self.player_index):
                agent_node = self.TurnBasedGameState(tree_node.game_state,action)
                val = self.rb_minimax(depth, agent_node)
                if (val > curr_max):
                    curr_max = val
            return curr_max

        pass

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        #use the heuristic function to choose best action
        #start = time.time()
        curr_max = -np.inf
        temp_action = np.random.choice(state.get_possible_actions(player_index=self.player_index))
        #temp_action = None
        for action in state.get_possible_actions(player_index=self.player_index):
            agent_node = self.TurnBasedGameState(state, action)
            val = self.rb_minimax(2, agent_node)
            if (val > curr_max):
                curr_max = val
                temp_action = action
        #stop = time.time()
        #self.time_sum += (stop - start)
        #self.counter += 1
        #print(self.time_sum / self.counter)
        return temp_action
        pass


class AlphaBetaAgent(MinimaxAgent):
    def rb_alpha_beta(self, depth, tree_node: MinimaxAgent.TurnBasedGameState, alpha, beta) -> float:
        if tree_node.game_state.is_terminal_state or depth == 0:
            return heuristic(tree_node.game_state, self.player_index)
        #if tree_node.agent_action != None:
        if tree_node.turn == self.Turn.OPPONENTS_TURN:
            curr_min = np.inf
            for opponents_actions in tree_node.game_state.get_possible_actions_dicts_given_action(tree_node.agent_action,
                                                                                   player_index=self.player_index):
                opponents_actions[self.player_index] = tree_node.agent_action
                next_state = get_next_state(tree_node.game_state, opponents_actions)
                oppnents_node = self.TurnBasedGameState(next_state, agent_action=None)
                val = self.rb_alpha_beta(depth-1, oppnents_node, alpha, beta)
                if(val < curr_min):
                    curr_min = val
                if (curr_min < beta):
                    beta = curr_min
                if curr_min <= alpha:
                    return -np.inf
            return curr_min
        else:
            curr_max = -np.inf
            for action in tree_node.game_state.get_possible_actions(player_index=self.player_index):
                agent_node = self.TurnBasedGameState(tree_node.game_state,action)
                val = self.rb_alpha_beta(depth, agent_node, alpha, beta)
                if (val > curr_max):
                    curr_max = val
                if curr_max > alpha:
                    alpha = curr_max
                if curr_max >= beta:
                    return np.inf
            return curr_max

        pass

    def get_action(self, state: GameState) -> GameAction:
        #start = time.time()
        # Insert your code here...
        #use the heuristic function to choose best action
        curr_max = -np.inf
        temp_action = np.random.choice(state.get_possible_actions(player_index=self.player_index))
        #temp_action = None
        for action in state.get_possible_actions(player_index=self.player_index):
            agent_node = self.TurnBasedGameState(state, action)
            val = self.rb_alpha_beta(2, agent_node, -np.inf, np.inf)
            if (val > curr_max):
                curr_max = val
                temp_action = action
        #stop = time.time()
        #self.time_sum += (stop - start)
        #self.counter += 1
        #print(self.time_sum / self.counter)
        return temp_action
        pass



def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    n=50
    game_action = [GameAction.LEFT, GameAction.RIGHT, GameAction.STRAIGHT]
    steps = np.random.choice(game_action, n)
    curr_benefit = get_fitness(steps)
    for i in range(50):
        if( steps[i] == GameAction.LEFT):
            steps[i] = GameAction.RIGHT
            benefit1 = get_fitness(steps)
            steps[i] = GameAction.STRAIGHT
            benefit2 = get_fitness(steps)
            if(benefit1>curr_benefit):
                steps[i] = GameAction.RIGHT
                curr_benefit = benefit1
            else:
                steps[i] = GameAction.LEFT
            if(benefit2>curr_benefit):
                steps[i] = GameAction.STRAIGHT
                curr_benefit = benefit2


        if (steps[i] == GameAction.RIGHT):
            steps[i] = GameAction.LEFT
            benefit1 = get_fitness(steps)
            steps[i] = GameAction.STRAIGHT
            benefit2 = get_fitness(steps)
            if (benefit1 > curr_benefit):
                steps[i] = GameAction.LEFT
                curr_benefit = benefit1
            else:
                steps[i] = GameAction.RIGHT
            if (benefit2 > curr_benefit):
                steps[i] = GameAction.STRAIGHT
                curr_benefit = benefit2

        if (steps[i] == GameAction.STRAIGHT):
            steps[i] = GameAction.RIGHT
            benefit1 = get_fitness(steps)
            steps[i] = GameAction.LEFT
            benefit2 = get_fitness(steps)
            if (benefit1 > curr_benefit):
                steps[i] = GameAction.RIGHT
                curr_benefit = benefit1
            else:
                steps[i] = GameAction.STRAIGHT
            if (benefit2 > curr_benefit):
                steps[i] = GameAction.LEFT
                curr_benefit = benefit2

    print(steps)
    print(get_fitness(steps))
    pass

def getMaxThreeValues(tupels_list):
    if len(tupels_list)<=3:
        return tupels_list
    to_return = []
    assert len(tupels_list) <= 9
    tmp_max = (None, -np.inf)
    i = tupels_list[0]
    for k in range(3):
        for i in tupels_list:
            if i[1] >= tmp_max[1]:
                tmp_max = i

        to_return.append((tmp_max[0].copy(), tmp_max[1]))
        for i in range(len(tupels_list)):
            v = tupels_list[i][0]==tmp_max[0]
            if v.all() and tupels_list[i][1]==tmp_max[1]:
                tupels_list.pop(i)
                break
        tmp_max = (None, -np.inf)
    return to_return


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm]
    3) print the best moves vector you found.
    :return:
    """
    n = 50
    game_action = [GameAction.LEFT, GameAction.RIGHT, GameAction.STRAIGHT]
    steps = np.random.choice(game_action, n)
    best_three_states = []
    for i in game_action:
        steps[0] = i
        best_three_states.append((steps.copy(), get_fitness(steps)))
    tmp_results = []

    for i in range(1, 50):
        for j in best_three_states:
            for k in game_action:
                j[0][i] = k
                tmp_results.append((j[0].copy(), get_fitness(j[0])))
        assert len(tmp_results) <= 9

        best_three_states.clear()
        best_three_states = getMaxThreeValues(tmp_results)
        tmp_results.clear()

    max11 = (None, -np.inf)
    for i in best_three_states:
        if (max11[1] <= i[1]):
            max11 = i
    print(max11[0])
    print(get_fitness(max11[0]))
    pass


class TournamentAgent(MinimaxAgent):

    total_time=0

    def get_action(self, state: GameState) -> GameAction:
        start = time.time()
        # Insert your code here...
        # use the heuristic function to choose best action
        curr_max = -np.inf
        temp_action = np.random.choice(state.get_possible_actions(player_index=self.player_index))
        # temp_action = None
        for action in state.get_possible_actions(player_index=self.player_index):
            agent_node = self.TurnBasedGameState(state, action)
            val = self.rb_minimax(2, agent_node)
            if (val > curr_max):
                curr_max = val
                temp_action = action

        stop = time.time()
        self.total_time += (stop-start)
        print(self.total_time)
        if(self.total_time>=60):
            self.alive = False
        return temp_action
        pass


if __name__ == '__main__':
    SAHC_sideways()
    local_search()

