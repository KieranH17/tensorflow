import chess
import player
import state


def train(num_sims, move_time, opening=None):
    board = chess.Board()
    state_history = state.StateHistoryKeeper()

    white = player.MontePython(board, move_time)
    black = player.MontePython(board, move_time)

    if opening in OPENING_FEN_DICT:
        for action in OPENING_FEN_DICT[opening]:
            board.push(chess.Move.from_uci(action))
            state_history.update_state_history(action)
            white.update_state(action)
            black.update_state(action)

    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            move = white.my_move()
            if not move:
                print("no move?")
            board.push(move)
            state_history.update_state_history(move.uci())
            black.update_state(move.uci())
        else:
            move = black.my_move()
            if not move:
                if not move:
                    print("no move?")
            board.push(move)
            state_history.update_state_history(move.uci())
            white.update_state(move.uci())
    else:
        if board.is_checkmate():
            print("Checkmate.")
        elif board.is_insufficient_material():
            print("Insufficient material.")
        elif board.is_stalemate():
            print("Stalemate.")
        outcome = board.result(claim_draw=True)
        # FIX_ME
        # for state in state_history
        #   train cnn on (state, outcome)


def is_int_as_string(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


OPENING_FEN_DICT = {"Queen's Gambit": ["d2d4", "d7d5", "c2c4"],
                    "English Opening": ["c2c4"],
                    "French Defense": ["e2e4", "e7e6", "d2d4", "d7d5"],
                    "Ruy Lopez": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]}
