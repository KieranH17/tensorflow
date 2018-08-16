import magnus_engine
import chess
import mc_search
import pickle


class Player:
    def __init__(self, board):
        self.board = board

    # guarantees a legal and valid move
    def my_move(self):
        raise NotImplementedError


class Human(Player):
    def my_move(self):
        print("Your move.")

        move_uci = input()
        if move_uci.lower() == "quit" or move_uci.lower() == "resign":
            return None

        move = None
        while not move:
            try:
                move = self.board.parse_uci(move_uci)
            except ValueError:
                print("Invalid move. Try again.")
                move_uci = input()
                if move_uci.lower() == "quit" or move_uci.lower() == "resign":
                    return None
        return move


class MontePython(Player):
    def __init__(self, board, timer):
        Player.__init__(self, board)
        self.timer = timer
        with open("seen_pos_legmov/seen_pos.pkl", "rb") as read_file:
            try:
                self.seen_pos_legmov_dict = pickle.load(read_file)
            except EOFError:
                self.seen_pos_legmov_dict = {}

        self.state = mc_search.State()

    def my_move(self, timer=None):
        if not timer:
            timer = self.timer
        move_uci = mc_search.monte_python_search(self.state, timer, self.seen_pos_legmov_dict)
        self.state = mc_search.State((self.state, move_uci))
        self.write_legmov_file()
        return chess.Move.from_uci(move_uci)

    def update_state(self, opp_move_uci):
        self.state = mc_search.State((self.state, opp_move_uci))

    def write_legmov_file(self):
        with open("seen_pos_legmov/seen_pos.pkl", "wb") as write_file:
            pickle.dump(self.seen_pos_legmov_dict, write_file)
