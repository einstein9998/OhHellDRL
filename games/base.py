class Card():
    suits = ('s', 'h', 'd', 'c')
    ranks = tuple(str(n) for n in range(2, 10)) + ('T', 'J', 'Q', 'K', 'A')
    @staticmethod
    def get_standard_deck():
        return [Card(r, s) for s in Card.suits for r in Card.ranks]

    @staticmethod
    def get_index_52():
        return dict((c, i) for i, c in enumerate(Card.get_standard_deck()))

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    @classmethod
    def from_string(cls, str):
        if len(str) == 2:
            return cls(str[0], str[1])
        else:
            raise ValueError("Invalid string format")

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        else:
            raise NotImplemented
    
    def __hash__(self):
        suit_index = Card.suits.index(self.suit)
        rank_index = Card.ranks.index(self.rank)
        return rank_index + 100 * suit_index
    
    def __repr__(self):
        return self.rank + self.suit