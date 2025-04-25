import numpy as np

class OhHellRound():
    def __init__(self, players, cards_per_player, num_players, np_random, dealer, trump_card, starting_player=0, current_player=0, verbose=False):
        self.np_random = np_random
        self.dealer = dealer
        self.trump_card = trump_card
        self.players = players

        self.trick = []

        self.cards_per_player = cards_per_player
        self.num_players = num_players
        self.starting_player = starting_player
        self.current_player = current_player

        self.verbose = verbose

        self.bids = [-1 for _ in range(self.num_players)]

    def get_num_bids(self):
        return sum(b >= 0 for b in self.bids)

    def proceed_round(self, action):
        legal_actions = self.get_legal_actions()
        if action not in legal_actions:
            raise Exception(f"{action} is not a legal action. {legal_actions=}")
        if isinstance(action, np.int32) or isinstance(action, int):
            self.players[self.current_player].bid = action
            self.players[self.current_player].acted = True
            self.bids[self.current_player] = action
            if self.verbose: print(f"Player {self.current_player} bids {action}\n")
            if self.get_num_bids() == self.num_players:
                for player in self.players:
                    player.acted = False
        else:
            self.players[self.current_player].played_cards.append(action)
            self.trick.append(action)
            self.players[self.current_player].hand.remove(action)
            self.players[self.current_player].acted = True
            if self.verbose: print(f"Player {self.current_player} plays {action}\n")

        self.current_player += 1
        self.current_player %= self.num_players

    def get_legal_actions(self):
        if self.players[self.current_player].bid == -1:
            full = list(range(self.cards_per_player + 1))
            if self.get_num_bids() == self.num_players - 1:
                return [b for b in full if b != self.cards_per_player - sum(self.bids)]
            else:
                return full
        else:
            full = self.players[self.current_player].hand
            if self.current_player == self.starting_player:
                return full
            else:
                led_suit = self.trick[0].suit
                follow_suit = [card for card in full if card.suit == led_suit]
                if follow_suit:
                    return follow_suit
                else:
                    return full
                
    def get_state(self):
        return {
            'current_player': self.current_player,
            'current_hand': self.players[self.current_player].hand,
            'bids': self.bids,
            'tricks_won': [player.tricks_won for player in self.players],
            'played_cards': [player.played_cards for player in self.players],
            'trick': self.trick,
            'acted': [player.acted for player in self.players]
        }
        
    def trick_over(self):
        return len(self.trick) == self.num_players