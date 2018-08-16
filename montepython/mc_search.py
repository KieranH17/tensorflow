
import tensorflow as tf
import chess
from collections import defaultdict
import time
from random import randrange, choice
from sys import maxsize
from math import sqrt, log


def monte_python_search(state, timer, legal_move_memo):
    root_node = Node(state, legal_move_memo)
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
    new_state = State((node.state, action))
    new_child_node = Node(new_state, legal_move_memo, parent=node, parent_action=action)
    node.add_child(new_child_node)
    return new_child_node


def best_child(node, c_value=sqrt(2)):
    max_uct = MIN_VALUE
    best_child_node = None
    for child_node in node.children:
        child_uct = (child_node.reward / child_node.visits) \
                    + c_value * sqrt(log(node.visits) / child_node.visits)
        if child_uct > max_uct:
            max_uct = child_uct
            best_child_node = child_node
    return best_child_node


def default_policy(state):
    curr_state = state
    while not curr_state.is_terminal():
        action = choice(curr_state.legal_actions)
        curr_state = State((curr_state, action))
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
    def __init__(self, state, legal_move_memo, parent=None, parent_action=None):
        self.state = state
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


class State:
    def __init__(self, old_state_and_action=None):
        if old_state_and_action:
            old_state = old_state_and_action[0]
            move = chess.Move.from_uci(old_state_and_action[1])
            move_start_ind = move.from_square
            move_end_ind = move.to_square
            piece_type_taken = old_state.board.piece_type_at(move_end_ind)
            piece_type_moved = old_state.board.piece_type_at(move_start_ind)

            self.my_pieces = []
            for i in range(len(old_state.opp_pieces)):
                self.my_pieces.append(old_state.opp_pieces[i][::-1])

            self.opp_pieces = []
            for i in range(len(old_state.my_pieces)):
                self.opp_pieces.append(old_state.my_pieces[i][::-1])

            update_piece_planes(self, old_state, move)

            self.board = old_state.board.copy()
            self.board.push(move)

            self.legal_actions = [move.uci() for move in self.board.legal_moves]

            self.rep_count = old_state.rep_count.copy()
            self.rep_count[self.board.epd()] += 1

            self.turn = self.board.turn
            self.full_mc = self.board.fullmove_number

            self.my_castling = get_bit_castles(self.board, self.turn)
            self.opp_castling = get_bit_castles(self.board, not self.turn)

            if piece_type_moved == chess.PAWN or piece_type_taken:
                self.np_count = 0
            else:
                self.np_count = old_state.np_count + 1/2

        else:
            self.board = chess.Board()

            self.legal_actions = [move.uci() for move in self.board.legal_moves]

            self.my_pieces = get_piece_planes(self.board, self.board.turn)
            self.opp_pieces = get_piece_planes(self.board, self.board.turn)

            self.rep_count = defaultdict(int)
            self.rep_count[self.board.epd()] += 1

            self.turn = self.board.turn
            self.full_mc = self.board.fullmove_number
            self.my_castling = get_bit_castles(self.board, self.turn)
            self.opp_castling = get_bit_castles(self.board, not self.turn)
            self.np_count = 0

    def is_terminal(self):
        if self.np_count > 49.5 or\
                self.rep_count[self.board.epd()] > 2 or\
                self.board.is_game_over():
            return True
        return False


# called after board perspective switches
def update_piece_planes(new_state, old_state, move):
    start_ind = move.from_square
    end_ind = move.to_square
    piece = old_state.board.piece_at(start_ind)
    if move.promotion:
        new_piece_type = move.promotion
    else:
        new_piece_type = piece.piece_type

    new_state.opp_pieces[new_piece_type - 1][63 - start_ind] = 0
    new_state.opp_pieces[new_piece_type - 1][63 - end_ind] = 1
    for i in range(NUM_PIECE_TYPES):
        new_state.my_pieces[i][63 - end_ind] = 0


def get_piece_planes(board, turn):
    pawn_list = [0] * 64
    knight_list = [0] * 64
    bishop_list = [0] * 64
    rook_list = [0] * 64
    queen_list = [0] * 64
    king_list = [0] * 64
    player_piece_lists = [pawn_list, knight_list, bishop_list, rook_list, queen_list, king_list]

    for i in range(NUM_PIECE_TYPES):
        piece_indices = board.pieces(i+1, turn)
        for j in piece_indices:
            player_piece_lists[i][j] = 1

    return player_piece_lists


def get_bit_castles(board, turn):
    castles = [0] * 2
    if board.has_kingside_castling_rights(turn):
        castles[0] = 1
    if board.has_queenside_castling_rights(turn):
        castles[1] = 1
    return castles


def terminal_state_to_outcome(state):
    if state.rep_count[state.board.epd()] > 2:
        result = "1/2-1/2"
    else:
        result = state.board.result(claim_draw=True)

    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    if result == "1/2-1/2":
        return 0
    else:
        print(result)
        print(state.np_count)
        print(state.rep_count[state.board.epd()])
        print(state.board.epd())
        return None


NUM_PIECE_TYPES = 6
MAX_VALUE = maxsize
MIN_VALUE = - MAX_VALUE
