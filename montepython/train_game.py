import chess
import player
import state


def run_train_game(
        move_time=0,
        board_classifier=None,
        opening=None,
        import_game_and_result=None,
        game_from_move=None):
    board = chess.Board()
    state_history = state.StateHistoryKeeper()

    white = player.MontePython(board, move_time, board_classifier=board_classifier)
    black = player.MontePython(board, move_time, board_classifier=board_classifier)

    if opening in OPENING_FEN_DICT:
        for action in OPENING_FEN_DICT[opening]:
            board.push(chess.Move.from_uci(action))
            state_history.update_state_history(action)
            white.update_state(action)
            black.update_state(action)

    if import_game_and_result:
        outcome_vector = []
        if import_game_and_result[1] == "1-0":
            outcome = 1
        elif import_game_and_result[1] == "0-1":
            outcome = -1
        else:
            outcome = 0

        outcome_vector.append(outcome)
        for move in import_game_and_result[0].main_line():
            board.push(move)
            state_history.update_state_history(move.uci())
            outcome_vector.append(outcome)

        states_to_return = state_history.get_state_history()
        outcomes_to_return = [x+1 for x in outcome_vector]

        if game_from_move:
            states_to_return = states_to_return[game_from_move:]
            outcomes_to_return = outcomes_to_return[game_from_move:]

        return states_to_return, outcomes_to_return

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
        result = board.result(claim_draw=True)
        outcome = result_to_int(result)

        outcome_vector = []
        for _ in state_history.get_state_history():
            outcome_vector.append(outcome)
            outcome = - outcome

        return state_history.get_state_history(), [x+1 for x in outcome_vector]


def is_int_as_string(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def result_to_int(result):
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    if result == "1/2-1/2":
        return 0


OPENING_FEN_DICT = {"Queen's Gambit": ["d2d4", "d7d5", "c2c4"],
                    "English Opening": ["c2c4"],
                    "French Defense": ["e2e4", "e7e6", "d2d4", "d7d5"],
                    "Ruy Lopez": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]}
