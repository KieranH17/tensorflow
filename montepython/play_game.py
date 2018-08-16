import chess
import player


def main():
    board = chess.Board()
    print("White or black?")

    player_color = input()
    resign = False
    while player_color.lower() != "white" and player_color.lower() != "black":
        print("Please select a valid side.")
        player_color = input()

    print("How long per computer turn in seconds? "
          "(E.g. input '3' gives 3 second turns).")
    comp_time = input()
    while not is_int_as_string(comp_time) \
            or int(comp_time) not in range(1, 20000):
        print("Please enter a valid number between 1 and 20000.")
        comp_time = input()

    comp_time = int(comp_time)

    if player_color.lower() == "white":
        white = player.Human(board)
        black = player.MontePython(board, comp_time)
        update_black = True
        update_white = False
    else:
        white = player.MontePython(board, comp_time)
        black = player.Human(board)
        update_white = True
        update_black = False
    while not board.is_game_over(claim_draw=True):
        print(board)
        if board.is_check():
            print("Check.")
        print()
        if board.turn == chess.WHITE:
            move = white.my_move()
            if not move:
                resign = True
                break
            board.push(move)
            if update_black:
                black.update_state(move.uci())
        else:
            move = black.my_move()
            if not move:
                resign = True
                break
            board.push(move)
            if update_white:
                white.update_state(move.uci())
    if resign:
        print("You resigned, good game.")
    else:
        if board.is_checkmate():
            print("Checkmate.")
        elif board.is_insufficient_material():
            print("Insufficient material.")
        elif board.is_stalemate():
            print("Stalemate.")
        print("Good game.")


def is_int_as_string(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
