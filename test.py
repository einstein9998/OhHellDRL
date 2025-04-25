from ohhell import OhHellEnv
from games.base import Card

if __name__ == '__main__':
    env = OhHellEnv(verbose=True)
    state = env.reset()
    print(f"your hand: {state}\n")
    done = False
    while not done:
        action = input("\nWhat is your action? ")
        try: 
            action = int(action)
        except:
            action = Card.from_string(action)
        next_state, _, done, _ = env.step(action, raw_action=True)
        print(f"your hand: {next_state}\n")