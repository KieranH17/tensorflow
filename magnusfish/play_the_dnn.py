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

    print("How long per computer turn in milliseconds? "
          "(E.g. input '3000' gives 3 second turns).")
    comp_time = input()
    while not is_int_as_string(comp_time) \
            or int(comp_time) not in range(100, 1000000):
        print("Please enter a valid number between 1000 and 1000000.")
        comp_time = input()

    comp_time = int(comp_time)

    if player_color.lower() == "white":
        white = player.Human(board)
        black = player.MagnusFish(board, comp_time)
    else:
        white = player.MagnusFish(board, comp_time)
        black = player.Human(board)
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
        else:
            move = black.my_move()
            if not move:
                resign = True
                break
            board.push(move)
    if resign:
        print("You resigned, good game.")
    else:
        if board.is_checkmate():
            print("Checkmate.")
        elif board.is_insufficient_material():
            print("Insufficient material.")
        elif board.is_stalemate():
            print("Stalemate.")
        print(str(board.result))
        print("Good game.")


def is_int_as_string(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
