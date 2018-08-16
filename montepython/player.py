import chess
import mc_search


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
        self.state = mc_search.State()
        self.legal_move_memo = {}

    def my_move(self, timer=None):
        if not timer:
            timer = self.timer
        move_uci = mc_search.monte_python_search(self.state, timer, self.legal_move_memo)
        self.update_state(move_uci)
        return chess.Move.from_uci(move_uci)

    def update_state(self, move_uci):
        self.state = mc_search.State((self.state, move_uci))
