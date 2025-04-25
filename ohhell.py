import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from collections import OrderedDict

from games.ohhell import Game
from games.base import Card
from utils.utils import rank2int, one_bit_set, int_to_binary_array

class OhHellEnv(gym.Env):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.game = Game(verbose=verbose)
        self.game.init_game()
        self.seed()
        self.action_recorder = []
        self.timestep = 0

        self.ACTION_SPACE = Card.get_index_52() | dict((str(i), i + 52) for i in range(self.game.n_cards + 1))
        self.ACTION_LIST = list(self.ACTION_SPACE.keys())

        self.observation_space = gym.spaces.MultiBinary(260)
        self.action_space = gym.spaces.Discrete(52 + self.game.n_cards + 1)

        self.card2index = Card.get_standard_deck()

        self.was_action_available = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _extract_state(self, state):
        obs_list = []

        obs_list.append(one_bit_set(4, Card.suits.index(state['trump_suit']))) # trump suit
        if state['current_hand']:
            obs_list.append(one_bit_set(52, np.array([self.card2index.index(card) for card in state['current_hand']]))) # cards in hand
        else:
            obs_list.append(np.zeros(52))
        obs_list.append(one_bit_set(self.game.num_players, state['current_player'])) # our index
        for i in range(self.game.num_players): # player information
            # obs_list.append(int_to_binary_array(state['bids'][i] if state['bids'][i] >= 0 else 0, num_bits=5)) # player bid
            # obs_list.append(int_to_binary_array(state['tricks_won'][i], num_bits=5)) # player tricks taken
            if state['bids'][i] >= 0:
                obs_list.append(one_bit_set(11, state['bids'][i]))
            else:
                obs_list.append(np.zeros(11))
            obs_list.append(one_bit_set(11, state['tricks_won'][i]))
            # if state['played_cards'][i]:
            #     obs_list.append(one_bit_set(52, np.array([self.card2index.index(card) for card in state['played_cards'][i]]))) # player cards played
            # else:
            #     obs_list.append(np.zeros(52))
        if any(state['played_cards'][i] for i in range(self.game.num_players)):
            obs_list.append(one_bit_set(52, np.array([self.card2index.index(card) for i in range(self.game.num_players) for card in state['played_cards'][i]])))
        else:
            obs_list.append(np.zeros(52))
        if state['trick']: 
            obs_list.append(one_bit_set(4, Card.suits.index(state['trick'][0].suit))) # led suit
            obs_list.append(one_bit_set(52, np.array([self.card2index.index(card) for card in state['trick']]))) # cards played in trick
        else:
            obs_list.append(np.zeros(4))
            obs_list.append(np.zeros(52))
        
        if any(state['acted']):
            obs_list.append(one_bit_set(4, np.array([i for i in range(self.game.num_players) if state['acted'][i]])))
        else:
            obs_list.append(np.zeros(4))

        return np.concatenate(obs_list)
    
    def _get_info(self):
        return {
            "action_mask": self.get_action_mask()
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.game.init_game()

        # while not self.game.players[self.game.round.current_player].isTraining:
        #     # current_obs = self._extract_state(self.game.round.get_state())
        #     # action, _ = self.trained_model.predict(current_obs)
        #     # state = self.game.step(self._decode_action(action))
        #     action = np.random.choice(self.game.round.get_legal_actions())
        #     state = self.game.step(action)
        
        self.action_recorder = []
        observation = self._extract_state(state)
        info = self._get_info()

        return observation, info
    
    def _decode_action(self, action_id):
        legal_ids = list(self._get_legal_actions())
        if self.game.round.get_num_bids() == self.game.num_players: # TODO: make this a helper fn in round
            if action_id in legal_ids:
                self.was_action_available = True
                return self.ACTION_LIST[action_id]
            else:
                self.was_action_available = False
                # self.game.players[self.game.round.current_player].wrong_actions_chosen += 1
                return self.ACTION_LIST[np.random.choice(legal_ids)]
        else:
            if action_id in legal_ids:
                self.was_action_available = True
                return int(self.ACTION_LIST[action_id])
            else:
                self.was_action_available = False
                # self.game.players[self.game.current_player].wrong_actions_chosen += 1
                return int(self.ACTION_LIST[np.random.choice(legal_ids)])
    
    def step(self, action, raw_action=False):
        if not raw_action:
            action = self._decode_action(action)
        
        agent = self.game.round.current_player
        current_tricks_won = self.game.players[agent].tricks_won
        next_state = self.game.step(action)

        agent_action_was_available = self.was_action_available

        # while not self.game.players[self.game.round.current_player].isTraining and not self.game.is_over():
        #     # current_obs = self._extract_state(self.game.round.get_state())
        #     # action, _ = self.trained_model.predict(current_obs)
        #     # state, _ = self.game.step(self._decode_action(action))
        #     action = np.random.choice(self.game.round.get_legal_actions())
        #     next_state = self.game.step(action)

        new_tricks_won = self.game.players[agent].tricks_won

        #reward = new_tricks_won - current_tricks_won # TODO: is this even correct?
        if agent_action_was_available: # first train to make legal moves
            reward = 1
        else:
            reward = -1 - 9 * (1 - np.exp(-5 * self.timestep/12000000))
        done = self.game.is_over()
        truncated = False
        info = self._get_info()
        obs = self._extract_state(next_state)

        self.timestep += 1

        return obs, reward, done, truncated, info
    
    def get_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=np.uint8)
        legal_actions = list(self._get_legal_actions())
        mask[legal_actions] = 1
        return mask
        

    def _get_legal_actions(self):
        legal_actions = self.game.round.get_legal_actions()
        if self.game.round.get_num_bids() == self.game.num_players: # TODO: make this a helper fn in round
            legal_ids = {self.ACTION_SPACE[action]: None for action in legal_actions}
        else:
            legal_ids = {self.ACTION_SPACE[str(action)]: None for action in legal_actions}
        return OrderedDict(legal_ids)


if __name__ == '__main__':
    env = OhHellEnv()
    obs = env.reset()