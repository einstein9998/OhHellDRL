class OhHellPlayer():
    def __init__(self, player_id, np_random, isTraining=False):
        self.np_random = np_random
        self.player_id = player_id
        self.hand = []
        self.played_cards = []
        self.bid = None
        self.isTraining = isTraining
        self.tricks_won = 0
        self.acted = False
