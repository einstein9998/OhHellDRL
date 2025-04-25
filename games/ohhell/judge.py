from games.ohhell.utils import determine_winner

class OhHellJudge():
    def __init__(self, np_random):
        self.np_random = np_random

    def judge_round(self, trick, trump_suit):
        return determine_winner(trick, trump_suit)

    def judge_game(self, players):
        return list(player.tricks_won + 10 * int(player.tricks_won == player.bid) for player in players)