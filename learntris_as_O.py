import json
import numpy as np
from random import choice

# Tic-Tac-Toe board
# '_' for empty, 'X' for agent, 'O' for player
board = ['_'] * 9

# Possible winning combinations
wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
        (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

# Initialize Q-table as a dictionary
Q = {}

# Hyperparameters
alpha = 1  # 1 for deterministic 0.5
gamma = 0.9
# Initial epsilon and decay rate
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9999


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


def select_action(state, board):
    """Selects an action using an epsilon-greedy strategy."""
    # With probability epsilon, choose a random action
    if np.random.rand() < epsilon:
        return np.random.choice(get_empty_spaces(board))
    # Otherwise, choose the action with the highest Q-value
    else:
        return best_action(state, board)


games = []
won_games = []

episodes = 500000
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

    # Opponent's turn
    board[choice(get_empty_spaces(board))] = 'X'

    while not done:
        game.append(board.copy())
        s = get_state(board)
        a = select_action(s, board)

        board[a] = 'O'

        win = check_win(board)
        if win:
            game.append(board.copy())
            won_games.append(game.copy())
            update_Q(s, get_state(board), 100, a)
            break

        update_Q(s, get_state(board), +1, a)

        empty_spaces = get_empty_spaces(board)
        if not empty_spaces:  # If no empty spaces, game is a draw
            break

        board[choice(empty_spaces)] = 'X'
        win = check_win(board)
        if win:
            update_Q(s, get_state(board), -100, a)
            break

        empty_spaces = get_empty_spaces(board)
        if not empty_spaces:  # If no empty spaces, game is a draw
            break

    # Record the result of this game
    if win == 'O':
        win_count += 1
    elif win == 'X':
        loss_count += 1
    else:
        draw_count += 1

    # Evaluate agent performance
    if (episode + 1) % eval_interval == 0:
        win_rate = win_count / eval_interval
        loss_rate = loss_count / eval_interval
        draw_rate = draw_count / eval_interval

        print(
            f"Episode {episode + 1}: Win rate: {win_rate}, Loss rate: {loss_rate}, Draw rate: {draw_rate}")

        # Reset the counts
        win_count = 0
        loss_count = 0
        draw_count = 0

        # Check for convergence
        if win_rate >= convergence:
            print("Agent converged!")
            break

    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    games.append(game)

# print("Saving won games...")
# with open("won_games.json", "w") as write_file:
#     json.dump(won_games, write_file, indent=4)
print("Saving model...")
with open("modelO.json", "w") as write_file:
    json.dump(Q, write_file, indent=4)

print("Model saved...")
