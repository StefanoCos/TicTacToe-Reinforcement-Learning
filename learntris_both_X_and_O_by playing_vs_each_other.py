import json
import numpy as np
from random import choice

preload_table = False

# Tic-Tac-Toe board
# '_' for empty, 'X' for agent, 'O' for player
board = ['_'] * 9

# Possible winning combinations
wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
        (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

# Initialize Q-table as a dictionary
Q = {}

if preload_table:
    # Opening JSON file
    with open('model_x_and_o.json') as json_file:
        Q = json.load(json_file)

# Hyperparameters
alpha = 1  # 1 for deterministic 0.5
gamma = 0.9
# Initial epsilon and decay rate
epsilon_x = 1.0
epsilon_o = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99999


def print_board(board):
    print(board[:3])
    print(board[3:6])
    print(board[6:])


def get_state(board):
    return "".join(board)


def get_empty_spaces(board):
    return [i for i, x in enumerate(board) if x == '_']


def check_win(board):
    for i, j, k in wins:
        if board[i] == board[j] == board[k] != '_':
            return board[i]
    return False


def update_Q(s, s_, r, a):
    if s not in Q:
        Q[s] = [0]*9
    if s_ not in Q:
        Q[s_] = [0]*9
    Q[s][a] = Q[s][a] + alpha * (r + gamma * max(Q[s_]) - Q[s][a])


def best_action(state, board):
    if state not in Q:
        return choice(get_empty_spaces(board))
    else:
        empty_spaces = get_empty_spaces(board)
        state_copy = Q[state].copy()
        for empty_space in empty_spaces:
            state_copy[empty_space] = state_copy[empty_space]+1000
        ar = np.array(state_copy)
        return np.random.choice(np.flatnonzero(ar == ar.max()))


def select_action(agent, state, board):
    """Selects an action using an epsilon-greedy strategy."""
    if agent == 'X':
        epsilon = epsilon_x
    else:
        epsilon = epsilon_o
    if np.random.rand() < epsilon:
        return np.random.choice(get_empty_spaces(board))
    else:
        return best_action(state, board)


games = []
won_games = []

episodes = 10000000
eval_interval = 1000
convergence = 0.999

# Performance tracking
win_count = 0
loss_count = 0
draw_count = 0

# Training loop
for episode in range(episodes):
    # Reset the board
    board = ['_'] * 9
    done = False
    game = []

    # Decide which agent goes first
    current_agent = 'X'  # choice(['X', 'O'])

    while not done:
        game.append(board.copy())
        s = get_state(board)
        a = select_action(current_agent, s, board)

        board[a] = current_agent

        win = check_win(board)
        if win:
            game.append(board.copy())
            won_games.append(game.copy())
            update_Q(s, get_state(board), 100, a)
            if win == 'X':
                win_count += 1
            else:
                loss_count += 1
            break

        update_Q(s, get_state(board), -1, a)

        empty_spaces = get_empty_spaces(board)
        if not empty_spaces:  # If no empty spaces, game is a draw
            draw_count += 1
            break

        # Switch to the other agent
        current_agent = 'O' if current_agent == 'X' else 'X'

    # Same code as before for updating win_count, loss_count, etc.

    epsilon_x = max(epsilon_min, epsilon_decay * epsilon_x)
    epsilon_o = max(epsilon_min, epsilon_decay * epsilon_o)
    games.append(game)

    # Print out win/loss/draw counts every 1000 episodes for tracking progress
    # Print stats every eval_interval episodes
    if episode % eval_interval == 0 and episode != 0:
        print(
            f"After {episode} episodes, Wins: {win_count / episode * 100:.2f}% ({win_count}), Losses: {loss_count / episode * 100:.2f}% ({loss_count}), Draws: {draw_count / episode * 100:.2f}% ({draw_count})")
        # print(
        #     f"Win percentage: {win_count / episode * 100:.2f}%, Loss percentage: {loss_count / episode * 100:.2f}%, Draw percentage: {draw_count / episode * 100:.2f}%")


# print("Saving won games...")
# with open("won_games.json", "w") as write_file:
#     json.dump(won_games, write_file, indent=4)
print("Saving model...")
with open("model_x_and_o.json", "w") as write_file:
    json.dump(Q, write_file, indent=4)

print("Model saved...")
