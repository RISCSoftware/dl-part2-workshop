import os
import math
import random
import time
import torch
import random
import logging
import coloredlogs

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.notebook import tqdm

import sys

sys.path.append("../nma_rl_games/alpha-zero")

import Arena

from utils import *
from Game import Game
from MCTS import MCTS
from NeuralNet import NeuralNet

# from othello.OthelloPlayers import *
from othello.OthelloLogic import Board

# from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as NNet


def set_seed(seed=None, seed_torch=True):
    """
    Function that controls randomness. NumPy and random modules must be imported.

    Args:
      seed : Integer
        A non-negative integer that defines the random state. Default is `None`.
      seed_torch : Boolean
        If `True` sets the random seed for pytorch tensors, so pytorch module
        must be imported. Default is `True`.

    Returns:
      Nothing.
    """
    if seed is None:
        seed = np.random.choice(2**32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f"Random seed {seed} has been set.")


# In case that `DataLoader` is used
def seed_worker(worker_id):
    """
    DataLoader will reseed workers following randomness in
    multi-process data loading algorithm.

    Args:
      worker_id: integer
        ID of subprocess to seed. 0 means that
        the data will be loaded in the main process
        Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details

    Returns:
      Nothing
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# @title Set device (GPU or CPU). Execute `set_device()`
# especially if torch modules used.

# Inform the user if the notebook uses GPU or CPU.


def set_device():
    """
    Set the device. CUDA if available, CPU otherwise

    Args:
      None

    Returns:
      Nothing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(
            "WARNING: For this notebook to perform best, "
            "if possible, in the menu under `Runtime` -> "
            "`Change runtime type.`  select `GPU` "
        )
    else:
        print("GPU is enabled in this notebook.")

    return device


args = dotdict(
    {
        "numIters": 1,  # In training, number of iterations = 1000 and num of episodes = 100
        "numEps": 1,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  # To control exploration and exploitation
        "updateThreshold": 0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 200,  # Number of game examples to train the neural networks.
        "numMCTSSims": 15,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 10,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "maxDepth": 5,  # Maximum number of rollouts
        "numMCsims": 5,  # Number of monte carlo simulations
        "mc_topk": 3,  # Top k actions for monte carlo rollout
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("/dev/models/8x100x50", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
        # Define neural network arguments
        "lr": 0.001,  # learning rate
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 64,
        "device": set_device(),
        "num_channels": 512,
    }
)


def loadTrainExamples(folder, filename):
    """
    Helper function to load training examples

    Args:
      folder: string
        Path specifying training examples
      filename: string
        File name of training examples

    Returns:
      trainExamplesHistory: list
        Returns examples based on the model were already collected (loaded)
    """
    trainExamplesHistory = []
    modelFile = os.path.join(folder, filename)
    examplesFile = modelFile + ".examples"
    if not os.path.isfile(examplesFile):
        print(f'File "{examplesFile}" with trainExamples not found!')
        r = input("Continue? [y|n]")
        if r != "y":
            sys.exit()
    else:
        print("File with train examples found. Loading it...")
        with open(examplesFile, "rb") as f:
            trainExamplesHistory = Unpickler(f).load()
        print("Loading done!")
        return trainExamplesHistory


def save_model_checkpoint(folder, filename, nnet):
    filepath = os.path.join(folder, filename)

    if not os.path.exists(folder):
        print("Checkpoint Directory does not exist! Making directory {}".format(folder))
        os.mkdir(folder)
    else:
        print("Checkpoint Directory exists!")

    torch.save({"state_dict": nnet.state_dict()}, filepath)
    print("Model saved!")


def load_model_checkpoint(folder, filename, nnet, device):
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError("No model in path {}".format(filepath))

    checkpoint = torch.load(filepath, map_location=device)
    nnet.load_state_dict(checkpoint["state_dict"])


class OthelloGame(Game):
    """
    Othello game board
    """

    square_content = {-1: "X", +0: "-", +1: "O"}

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        # Return number of actions, n is the board size and +1 is for no-op action
        return self.n * self.n + 1

    def getCanonicalForm(self, board, player):
        # Return state if player==1, else return -state if player==-1
        return player * board

    def stringRepresentation(self, board):
        return board.tobytes()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")  # Print the row
            for x in range(n):
                piece = board[y][x]  # Get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")
        print("-----------------------")

    @staticmethod
    def displayValidMoves(moves):
        A = np.reshape(moves[0:-1], board.shape)
        n = board.shape[0]
        print("  ")
        print("possible moves")
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")  # Print the row
            for x in range(n):
                piece = A[y][x]  # Get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")
        print("-----------------------")

    def getNextState(self, board, player, action):
        """
        Make valid move. If player takes action on board, return next (board,player)
        and action must be a valid move

        Args:
          board: np.ndarray
            Board of size n x n [6x6 in this case]
          player: Integer
            ID of current player
          action: np.ndarray
            Space of actions

        Returns:
          (board, player): tuple
            Next state representation
        """
        if action == self.n * self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        """
        Get all valid moves for player

        Args:
          board: np.ndarray
            Board of size n x n [6x6 in this case]
          player: Integer
            ID of current player
          action: np.ndarray
            Space of action

        Returns:
          valids: np.ndarray
            Valid moves for player
        """
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Check if game ended

        Args:
          board: np.ndarray
            Board of size n x n [6x6 in this case]
          player: Integer
            ID of current player

        Returns:
          0 if not ended, 1 if player 1 won, -1 if player 1 lost
        """
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getSymmetries(self, board, pi):
        """
        Get mirror/rotational configurations of board

        Args:
          board: np.ndarray
            Board of size n x n [6x6 in this case]
          pi: np.ndarray
            Dimension of board

        Returns:
          l: list
            90 degree of board, 90 degree of pi_board
        """
        assert len(pi) == self.n**2 + 1  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l


class RandomPlayer:

    def __init__(self, game):
        self.game = game

    def play(self, board):
        """
        Simulates game play

        Args:
          board: np.ndarray
            Board of size n x n [6x6 in this case]

        Returns:
          a: int
            Randomly chosen move
        """

        # Compute the valid moves using getValidMoves()
        valids = self.game.getValidMoves(board, 1)

        # Compute the probability of each move being played (random player means this should
        # be uniform for valid moves, 0 for others)
        prob = valids / valids.sum()

        # Pick an action based on the probabilities (hint: np.choice is useful)
        a = np.random.choice(self.game.getActionSize(), p=prob)

        return a


class OthelloNNet(nn.Module):

    def __init__(self, game, args):
        """
        Initialise game parameters

        Args:
          game: OthelloGame instance
            Instance of the OthelloGame class above;
          args: dictionary
            Instantiates number of iterations and episodes, controls temperature threshold, queue length,
            arena, checkpointing, and neural network parameters:
            learning-rate: 0.001, dropout: 0.3, epochs: 10, batch_size: 64,
            num_channels: 512
        """
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(OthelloNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=args.num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=args.num_channels, out_channels=args.num_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(in_channels=args.num_channels, out_channels=args.num_channels, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=args.num_channels, out_channels=args.num_channels, kernel_size=3, stride=1)

        self.bn1 = nn.BatchNorm2d(num_features=args.num_channels)
        self.bn2 = nn.BatchNorm2d(num_features=args.num_channels)
        self.bn3 = nn.BatchNorm2d(num_features=args.num_channels)
        self.bn4 = nn.BatchNorm2d(num_features=args.num_channels)

        self.fc1 = nn.Linear(in_features=args.num_channels * (self.board_x - 4) * (self.board_y - 4), out_features=1024)
        self.fc_bn1 = nn.BatchNorm1d(num_features=1024)

        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc_bn2 = nn.BatchNorm1d(num_features=512)

        self.fc3 = nn.Linear(in_features=512, out_features=self.action_size)

        self.fc4 = nn.Linear(in_features=512, out_features=1)

    def forward(self, s):
        """
        Controls forward pass of OthelloNNet

        Args:
          s: np.ndarray
            Array of size (batch_size x board_x x board_y)

        Returns:
          prob, v: tuple of torch.Tensor
            Probability distribution over actions at the current state and the value
            of the current state.
        """
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training
        )  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class ValueNetwork(NeuralNet):

    def __init__(self, game):
        """
        Args:
          game: OthelloGame
            Instance of the OthelloGame class above
        """
        self.nnet = OthelloNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.nnet.to(args.device)

    def train(self, games):
        """
        Args:
          games: list
            List of examples with each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        for examples in games:
            for epoch in range(args.epochs):
                print("EPOCH ::: " + str(epoch + 1))
                self.nnet.train()
                v_losses = []  # To store the losses per epoch
                batch_count = int(len(examples) / args.batch_size)  # len(examples)=200, batch-size=64, batch_count=3
                t = tqdm(range(batch_count), desc="Training Value Network")
                for _ in t:
                    sample_ids = np.random.randint(
                        len(examples), size=args.batch_size
                    )  # Read the ground truth information from MCTS simulation using the loaded examples
                    boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))  # Length of boards, pis, vis = 64
                    boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                    target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                    # Predict
                    # To run on GPU if available
                    boards, target_vs = boards.contiguous().to(args.device), target_vs.contiguous().to(args.device)

                    # Compute output
                    _, out_v = self.nnet(boards)
                    l_v = self.loss_v(target_vs, out_v)  # Total loss

                    # Record loss
                    v_losses.append(l_v.item())
                    t.set_postfix(Loss_v=l_v.item())

                    # Compute gradient and do SGD step
                    optimizer.zero_grad()
                    l_v.backward()
                    optimizer.step()

    def predict(self, board):
        """
        Args:
          board: np.ndarray
            Board of size n x n [6x6 in this case]

        Returns:
          v: OthelloNet instance
            Data of the OthelloNet class instance above;
        """
        # Timing
        start = time.time()

        # Preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        board = board.contiguous().to(args.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            _, v = self.nnet(board)
        return v.data.cpu().numpy()[0]

    def loss_v(self, targets, outputs):
        """
        Args:
          targets: np.ndarray
            Ground Truth variables corresponding to input
          outputs: np.ndarray
            Predictions of Network

        Returns:
          MSE Loss averaged across the whole dataset
        """
        # Mean squared error (MSE)
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        save_model_checkpoint(folder, filename, self.nnet)

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        load_model_checkpoint(folder, filename, self.nnet, args.device)


class ValueBasedPlayer:

    def __init__(self, game, vnet):
        """
        Args:
          game: OthelloGame instance
            Instance of the OthelloGame class
          vnet: Value Network instance
            Instance of the Value Network class
        """
        self.game = game
        self.vnet = vnet

    def play(self, board):
        """
        Args:
          board: np.ndarray
            Board of size n x n [6x6 in this case]

        Returns:
          candidates: List
            Collection of tuples describing action and values of future predicted
            states
        """
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        max_num_actions = 4
        va = np.where(valids)[0]
        va_list = va.tolist()
        random.shuffle(va_list)
        for a in va_list:
            # Return next board state using getNextState() function
            nextBoard, _ = self.game.getNextState(board, 1, a)
            # Predict the value of next state using value network
            value = self.vnet.predict(nextBoard)
            # Add the value and the action as a tuple to the candidate lists, note that you might need to change the sign of the value based on the player
            candidates += [(-value, a)]

            if len(candidates) == max_num_actions:
                break

        # Sort by the values
        candidates.sort()

        # Return action associated with highest value
        return candidates[0][1]


class PolicyNetwork(NeuralNet):

    def __init__(self, game):
        """
        Args:
          game: OthelloGame
            Instance of the OthelloGame class
        """
        self.nnet = OthelloNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.nnet.to(args.device)

    def train(self, games):
        """
        Args:
          games: list
            List of examples where each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for examples in games:
            for epoch in range(args.epochs):
                print("EPOCH ::: " + str(epoch + 1))
                self.nnet.train()
                pi_losses = []

                batch_count = int(len(examples) / args.batch_size)

                t = tqdm(range(batch_count), desc="Training Policy Network")
                for _ in t:
                    sample_ids = np.random.randint(len(examples), size=args.batch_size)
                    boards, pis, _ = list(zip(*[examples[i] for i in sample_ids]))
                    boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                    target_pis = torch.FloatTensor(np.array(pis))

                    # Predict
                    boards, target_pis = boards.contiguous().to(args.device), target_pis.contiguous().to(args.device)

                    # Compute output
                    out_pi, _ = self.nnet(boards)
                    l_pi = self.loss_pi(target_pis, out_pi)

                    # Record loss
                    pi_losses.append(l_pi.item())
                    t.set_postfix(Loss_pi=l_pi.item())

                    # Compute gradient and do SGD step
                    optimizer.zero_grad()
                    l_pi.backward()
                    optimizer.step()

    def predict(self, board):
        """
        Args:
          board: np.ndarray
            Board of size n x n [6x6 in this case]

        Returns:
          Data from the OthelloNet instance
        """
        # Timing
        start = time.time()

        # Preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        board = board.contiguous().to(args.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, _ = self.nnet(board)
        return torch.exp(pi).data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        """
        Calculates Negative Log Likelihood(NLL) of Targets

        Args:
          targets: np.ndarray
            Ground Truth variables corresponding to input
          outputs: np.ndarray
            Predictions of Network

        Returns:
          Negative Log Likelihood calculated as: When training a model, we aspire to
          find the minima of a loss function given a set of parameters (in a neural
          network, these are the weights and biases).
          Sum the loss function to all the correct classes. So, whenever the network
          assigns high confidence at the correct class, the NLL is low, but when the
          network assigns low confidence at the correct class, the NLL is high.
        """
        ## For more information, here is a reference that connects the expression to
        # the neg-log-prob: https://gombru.github.io/2018/05/23/cross_entropy_loss/
        return -torch.sum(targets * outputs) / targets.size()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        save_model_checkpoint(folder, filename, self.nnet)

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        load_model_checkpoint(folder, filename, self.nnet, args.device)


class PolicyBasedPlayer:

    def __init__(self, game, pnet, greedy=True):
        """
        Args:
          game: OthelloGame instance
            Instance of the OthelloGame class above;
          pnet: Policy Network instance
            Instance of the Policy Network class above
          greedy: Boolean
            If true, implement greedy approach
            Else, implement random sample policy based player
        """
        self.game = game
        self.pnet = pnet
        self.greedy = greedy

    def play(self, board):
        """
        Args:
          board: np.ndarray
            Board of size n x n [6x6 in this case]

        Returns:
          a: np.ndarray
            If greedy, implement greedy policy player
            Else, implement random sample policy based player
        """
        valids = self.game.getValidMoves(board, 1)
        action_probs = self.pnet.predict(board)
        vap = action_probs * valids  # Masking invalid moves
        sum_vap = np.sum(vap)

        if sum_vap > 0:
            vap /= sum_vap  # Renormalize
        else:
            # If all valid moves were masked we make all valid moves equally probable
            print("All valid moves were masked, doing a workaround.")
            vap = vap + valids
            vap /= np.sum(vap)

        if self.greedy:
            # Greedy policy player
            a = np.where(vap == np.max(vap))[0][0]
        else:
            # Sample-based policy player
            a = np.random.choice(self.game.getActionSize(), p=vap)

        return a


class MonteCarlo:

    def __init__(self, game, nnet, args):
        """
        Args:
          game: OthelloGame instance
            Instance of the OthelloGame class above;
          nnet: OthelloNet instance
            Instance of the OthelloNNet class above;
          args: dictionary
            Instantiates number of iterations and episodes, controls temperature threshold, queue length,
            arena, checkpointing, and neural network parameters:
            learning-rate: 0.001, dropout: 0.3, epochs: 10, batch_size: 64,
            num_channels: 512
        """
        self.game = game
        self.nnet = nnet
        self.args = args

        self.Ps = {}  # Stores initial policy (returned by neural net)
        self.Es = {}  # Stores game.getGameEnded ended for board s

    # Call this rollout
    def simulate(self, canonicalBoard):
        """
        Simulate one Monte Carlo rollout

        Args:
          canonicalBoard: np.ndarray
            Canonical Board of size n x n [6x6 in this case]

        Returns:
          temp_v:
            Terminal State
        """
        s = self.game.stringRepresentation(canonicalBoard)
        init_start_state = s
        temp_v = 0
        isfirstAction = None
        current_player = -1  # opponent's turn (the agent has already taken an action before the simulation)
        self.Ps[s], _ = self.nnet.predict(canonicalBoard)

        for i in range(self.args.maxDepth):  # maxDepth

            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
            if self.Es[s] != 0:
                # Terminal state
                temp_v = self.Es[s] * current_player
                break

            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # Masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])

            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # Renormalize
            else:
                # If all valid moves were masked make all valid moves equally probable
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            # Choose action according to the policy distribution
            a = np.random.choice(self.game.getActionSize(), p=self.Ps[s])
            # Find the next state and the next player
            next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
            canonicalBoard = self.game.getCanonicalForm(next_s, next_player)
            s = self.game.stringRepresentation(next_s)
            current_player *= -1
            # Initial policy
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            temp_v = v.item() * current_player

        return temp_v


class MonteCarloBasedPlayer:
    """
    Simulate Player based on Monte Carlo Algorithm
    """

    def __init__(self, game, nnet, args):
        """
        Args:
          game: OthelloGame instance
            Instance of the OthelloGame class above;
          nnet: OthelloNet instance
            Instance of the OthelloNNet class above;
          args: dictionary
            Instantiates number of iterations and episodes, controls temperature threshold, queue length,
            arena, checkpointing, and neural network parameters:
            learning-rate: 0.001, dropout: 0.3, epochs: 10, batch_size: 64,
            num_channels: 512
        """
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mc = MonteCarlo(game, nnet, args)
        self.K = self.args.mc_topk

    def play(self, canonicalBoard):
        """
        Simulate Play on Canonical Board

        Args:
          canonicalBoard: np.ndarray
            Canonical Board of size n x n [6x6 in this case]

        Returns:
          best_action: tuple
            (avg_value, action) i.e., Average value associated with corresponding action
            i.e., Action with the highest topK probability
        """
        self.qsa = []
        s = self.game.stringRepresentation(canonicalBoard)
        Ps, v = self.nnet.predict(canonicalBoard)
        valids = self.game.getValidMoves(canonicalBoard, 1)
        Ps = Ps * valids  # Masking invalid moves
        sum_Ps_s = np.sum(Ps)

        if sum_Ps_s > 0:
            Ps /= sum_Ps_s  # Renormalize
        else:
            # If all valid moves were masked make all valid moves equally probable
            # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
            # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
            log = logging.getLogger(__name__)
            log.error("All valid moves were masked, doing a workaround.")
            Ps = Ps + valids
            Ps /= np.sum(Ps)

        num_valid_actions = np.shape(np.nonzero(Ps))[1]

        if num_valid_actions < self.K:
            top_k_actions = np.argpartition(Ps, -num_valid_actions)[-num_valid_actions:]
        else:
            top_k_actions = np.argpartition(Ps, -self.K)[-self.K :]  # To get actions that belongs to top k prob

        for action in top_k_actions:
            next_s, next_player = self.game.getNextState(canonicalBoard, 1, action)
            next_s = self.game.getCanonicalForm(next_s, next_player)

            values = []

            # Do some rollouts
            for rollout in range(self.args.numMCsims):
                value = self.mc.simulate(next_s)
                values.append(value)

            # Average out values
            avg_value = np.mean(values)
            self.qsa.append((avg_value, action))

        self.qsa.sort(key=lambda a: a[0])
        self.qsa.reverse()
        best_action = self.qsa[0][1]
        return best_action

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Get probabilities associated with each action

        Args:
          canonicalBoard: np.ndarray
            Canonical Board of size n x n [6x6 in this case]
          temp: Integer
            Signifies if game is in terminal state

        Returns:
          action_probs: List
            Probability associated with corresponding action
        """
        if self.game.getGameEnded(canonicalBoard, 1) != 0:
            return np.zeros((self.game.getActionSize()))

        else:
            action_probs = np.zeros((self.game.getActionSize()))
            best_action = self.play(canonicalBoard)
            action_probs[best_action] = 1

        return action_probs
