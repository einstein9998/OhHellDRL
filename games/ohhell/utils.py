from utils.utils import rank2int, int2rank
from games.base import Card

def determine_winner(trick, trump_suit):
    led_suit = trick[0].suit
    trump_played = [rank2int(card.rank) for card in trick if card.suit == trump_suit]
    follow_suit = [rank2int(card.rank) for card in trick if card.suit == led_suit]

    if trump_played:
        highest = max(trump_played)
        highest_rank = int2rank(highest) # TODO: make rank class with ordering
        return trick.index(Card(highest_rank, trump_suit))
    else:
        highest = max(follow_suit)
        highest_rank = int2rank(highest)
        return trick.index(Card(highest_rank, led_suit))