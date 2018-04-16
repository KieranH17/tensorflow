import chess
import player


def main():
    board = chess.Board()
    print("white or black")

    player_color = input()
    resign = False
    while player_color.lower() != "white" and player_color.lower() != "black":
        print("Please select a valid side.")
        player_color = input()
    if player_color.lower() == "white":
        white = player.Human(board)
        black = player.MagnusFish(board)
    else:
        white = player.MagnusFish(board)
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


if __name__ == "__main__":
    main()
