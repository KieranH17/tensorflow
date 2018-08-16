
import tensorflow as tf
import chess
import time
from random import randrange, choice
from math import sqrt, log
import state


def monte_python_search(root_state, timer, legal_move_memo):
    root_node = Node(root_state, legal_move_memo)
    i = 0
    timeout = time.time() + timer
    while time.time() < timeout:
        next_node = tree_policy(root_node, legal_move_memo)
        outcome = default_policy(next_node.state)
        backup(next_node, outcome)
        i += 1
    print(i)
    return best_child(root_node, 0).parent_action


def tree_policy(node, legal_move_memo):
    curr_node = node
    while not curr_node.state.is_terminal():
        if curr_node.unexplored_actions:
            return expand(curr_node, legal_move_memo)
        else:
            curr_node = best_child(curr_node)
    return curr_node


def expand(node, legal_move_memo):
    unexplored_actions = node.unexplored_actions
    rand_ind = randrange(len(unexplored_actions))
    unexplored_actions[rand_ind], unexplored_actions[-1] = \
        unexplored_actions[-1], unexplored_actions[rand_ind]
    action = unexplored_actions.pop()
    new_state = state.State((node.state, action))
    new_child_node = Node(new_state, legal_move_memo, parent=node, parent_action=action)
    node.add_child(new_child_node)
    return new_child_node


def best_child(node, c_value=sqrt(2)):
    max_uct = state.MIN_VALUE
    best_child_node = None
    for child_node in node.children:
        child_uct = (child_node.reward / child_node.visits) \
                    + c_value * sqrt(log(node.visits) / child_node.visits)
        if child_uct > max_uct:
            max_uct = child_uct
            best_child_node = child_node
    return best_child_node


def default_policy(curr_state):
    curr_state = curr_state
    while not curr_state.is_terminal():
        action = choice(curr_state.legal_actions)
        curr_state = state.State((curr_state, action))
    return terminal_state_to_outcome(curr_state)


def backup(node, reward):
    curr_node = node
    if curr_node.parent_turn == chess.BLACK:
        reward = - reward
    while curr_node:
        curr_node.visits += 1
        curr_node.reward += reward
        reward = -reward
        curr_node = curr_node.parent


class Node:
    # parent_action is the Move taken by the parent to get to this node
    def __init__(self, my_state, legal_move_memo, parent=None, parent_action=None):
        self.state = my_state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.visits = 0
        self.reward = 0
        self.parent_turn = not self.state.turn
        self.legal_move_memo = legal_move_memo

        if self.state.board.epd() in self.legal_move_memo:
            self.unexplored_actions = self.legal_move_memo[self.state.board.epd()]
        else:
            self.unexplored_actions = [action for action in self.state.legal_actions]
            self.legal_move_memo[self.state.board.epd()] = self.unexplored_actions

    def update_and_backprop(self, outcome_value):
        self.reward += outcome_value
        if self.parent:
            self.parent.update_and_backprop(outcome_value)

    def add_child(self, child):
        self.children.append(child)


def terminal_state_to_outcome(state_to_check):
    if state_to_check.rep_count[state_to_check.board.epd()] > 2:
        result = "1/2-1/2"
    else:
        result = state_to_check.board.result(claim_draw=True)

    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    if result == "1/2-1/2":
        return 0
