import json
from random import choice
import tkinter as tk
import numpy as np


class TicTacToe(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.players = ["X", "O"]
        self.turn = 0
        self.winner = None

        # Opening JSON file
        with open('model_x_and_o.json') as json_file:
            self.Q = json.load(json_file)

        self.create_widgets()

    def create_widgets(self):
        self.board_frame = tk.Frame(self.master)
        self.board_frame.pack(padx=10, pady=10)
        self.buttons = []
        for row in range(3):
            button_row = []
            for col in range(3):
                button = tk.Button(self.board_frame, text=" ", width=6, height=3,
                                   command=lambda row=row, col=col: self.play_move(row, col, isAI=False))
                button.grid(row=row, column=col, padx=3, pady=3)
                button_row.append(button)
            self.buttons.append(button_row)

        self.normal_bg_color = self.buttons[0][0]['bg']

        self.status_label = tk.Label(
            self.master, text="Player " + self.players[self.turn % 2] + "'s turn")
        self.status_label.pack(pady=5)

        self.reset_button = tk.Button(
            self.master, text="New game", command=self.reset_game)
        self.reset_button.pack(pady=10)

    def play_move(self, row, col, isAI=True):
        print("Move "+self.players[self.turn %
                                   2]+": " + str(row)+" - "+str(col))
        if not self.winner and self.board[row][col] == " ":

            self.board[row][col] = self.players[self.turn % 2]
            self.buttons[row][col].configure(text=self.players[self.turn % 2])
            self.turn += 1
            self.winner, winning_line = self.check_win()
            if self.winner:
                if isAI:
                    color = 'red'
                    text = "Player " + self.winner + "(AI) wins!"
                else:
                    color = 'green'
                    text = "Player " + self.winner + " wins!"
                for i, j in winning_line:
                    self.buttons[i][j].configure(
                        bg=color)  # Change color to green
                self.status_label.configure(
                    text=text)
            elif self.turn == 9:
                for button_row in self.buttons:
                    for button in button_row:
                        button.configure(bg='yellow')
                self.status_label.configure(text="It's a tie.")
            else:
                self.status_label.configure(
                    text="Player " + self.players[self.turn % 2] + "'s turn")

            new_board = []
            for array in self.board:
                for el in array:
                    new_board.append(el.replace(' ', '_'))

            if not isAI and self.get_empty_spaces(new_board) != []:
                self.AI_move()

    def check_win(self):
        # Check rows
        for i, row in enumerate(self.board):
            if row[0] != " " and len(set(row)) == 1:
                return row[0], [(i, j) for j in range(3)]
        # Check columns
        for i in range(3):
            col = [self.board[j][i] for j in range(3)]
            if col[0] != " " and len(set(col)) == 1:
                return col[0], [(j, i) for j in range(3)]
        # Check diagonals
        if self.board[0][0] != " " and len(set([self.board[i][i] for i in range(3)])) == 1:
            return self.board[0][0], [(i, i) for i in range(3)]
        if self.board[0][2] != " " and len(set([self.board[i][2-i] for i in range(3)])) == 1:
            return self.board[0][2], [(i, 2-i) for i in range(3)]
        # No winner yet
        return None, []

    def reset_game(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.turn = 0
        self.winner = None
        for button_row in self.buttons:
            for button in button_row:
                button.configure(bg=self.normal_bg_color)
                button.configure(text=" ")
        self.status_label.configure(
            text="Player " + self.players[self.turn % 2] + "'s turn")

    def get_empty_spaces(self, board):
        return [i for i, x in enumerate(board) if x == '_']

    def best_action(self, state):
        new_board = []
        for array in self.board:
            for el in array:
                new_board.append(el.replace(' ', '_'))

        if state not in self.Q:
            print("random")
            return choice(self.get_empty_spaces(new_board))
        else:
            print("choice")
            empty_spaces = self.get_empty_spaces(new_board)
            state_copy = self.Q[state].copy()
            for empty_space in empty_spaces:
                state_copy[empty_space] = state_copy[empty_space]+1000
            ar = np.array(state_copy)
            return np.random.choice(np.flatnonzero(ar == ar.max()))

    def get_state(self, board):
        return "".join(board)

    def AI_move(self):
        # 1 replace the = with X since the model is trained with X as a player in mind
        my_str = ''.join([str(elem) for row in self.board for elem in row])

        my_list = list(my_str)
        s = self.get_state(my_list)
        s = s.replace(' ', '_')
        print("AI move considering board state: " + s)
        a = self.best_action(s)
        row = int(a/3)
        col = a % 3
        self.play_move(row, col)


root = tk.Tk()
root.title("Tic-Tac-Toe")

game = TicTacToe(root)


root.mainloop()
