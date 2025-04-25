from games.base import Card

class OhHellDealer():
    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = Card.get_standard_deck()
        self.shuffle()

    def shuffle(self):
        self.np_random.shuffle(self.deck)

    def flip_trump_card(self):
        return self.deck.pop()
    
    def deal_cards(self, player, num):
        player.hand += [self.deck.pop() for _ in range(num)]

    def deal_card(self):
        return self.deck.pop()