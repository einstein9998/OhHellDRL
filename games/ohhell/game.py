import numpy as np
import random
from copy import deepcopy, copy

from games.ohhell import Dealer, Player, Judge, Round

class OhHellGame():
    def __init__(self, allow_step_back=False, num_players=4, n_cards=10, verbose=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = num_players
        self.n_cards = n_cards
        self.payoffs = [0 for _ in range(num_players)]
        self.verbose = verbose

    def configure(self, game_config):
        pass

    def init_game(self):
        self.dealer = Dealer(self.np_random)

        self.players = [Player(i, self.np_random) for i in range(self.num_players)]

        self.training_index = random.randint(0, 3)
        self.players[self.training_index].isTraining = True
        if self.verbose: print([player.isTraining for player in self.players])

        self.judge = Judge(self.np_random)

        for i in range(self.n_cards * self.num_players):
            self.players[i % self.num_players].hand.append(self.dealer.deal_card())

        self.trump_card = self.dealer.flip_trump_card()
        if self.verbose: print(f"{self.trump_card=}")

        starting_player = random.randint(0, self.num_players - 1)
        if self.verbose: print(f"{starting_player=}")

        self.round = Round(self.players, self.n_cards, self.num_players, self.np_random, self.dealer, 
                           self.trump_card, starting_player=starting_player, current_player=starting_player, verbose=self.verbose)

        self.history = []

        self.trick_count = 0

        return self.get_state()
    
    def step(self, action):
        if self.allow_step_back:
            r = deepcopy(self.round)
            tc = self.trick_count
            d = deepcopy(self.dealer)
            ps = deepcopy(self.players)
            self.history.append((r, tc, d, ps))

        self.round.proceed_round(action)
        if self.round.trick_over():
            
            winner_index = self.judge.judge_round(self.round.trick, self.trump_card.suit)
            trick_winner = (self.round.starting_player + winner_index) % self.num_players
            self.round.starting_player = trick_winner
            self.round.current_player = trick_winner
            self.players[trick_winner].tricks_won += 1
            self.round.trick = []
            self.trick_count += 1
            if self.verbose: print(f"Player {trick_winner} wins the trick\n")
            for player in self.players:
                player.acted = False

        return self.get_state()
    
    def step_back(self):
        if len(self.history):
            self.round, self.trick_count, self.dealer, self.players = self.history.pop()
            return True
        return False
    
    def get_state(self):
        if self.is_over():
            return {'trump_suit': self.trump_card.suit} | self.round.get_state()# | {'tricks_won': self.judge.judge_game(self.players)}
        else:
            return {'trump_suit': self.trump_card.suit} | self.round.get_state()
    
    def is_over(self):
        return self.trick_count == self.n_cards