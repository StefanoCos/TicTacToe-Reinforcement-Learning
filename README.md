# TicTacToe-Reinforcement-Learning
This project is essentially an implementation of a reinforcement learning algorithm (Q-Learning) for the game of Tic Tac Toe. The aim of the project is to train an AI agent to play the game optimally.

Here's a breakdown of how learntris_both_X_and_O_by playing_vs_each_other.py script works (the other two are just semplified versions):

1. **Initialization**: The Tic Tac Toe board is a list of 9 items (3x3 grid) where '_' represents an empty space. The script also initializes a Q-table as an empty dictionary, which will be filled as the AI learns. The `wins` variable contains all possible winning combinations in the game.

2. **Q-Learning Parameters**: The script sets several hyperparameters including alpha (learning rate), gamma (discount factor), epsilon (exploration rate), and epsilon_decay (rate at which epsilon decreases). 

3. **Main Loop**: The AI plays a large number of games (as defined by `episodes`) against itself. On each move, it decides whether to make a random move (exploration) or to use the Q-table to decide the best move (exploitation). This decision is based on the value of epsilon. 

4. **Updating Q-Table**: After each move, it updates the Q-value of the state-action pair using the Q-learning update rule. When a game ends, it gives a reward of +100 for a win and updates the Q-values accordingly.

5. **Track Performance**: It keeps track of the number of games won, lost, and drawn. It prints these stats after every `eval_interval` episodes.

6. **Model Saving**: At the end of training, the script saves the final Q-table as a JSON file named 'model_x_and_o.json'.
