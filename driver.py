from utils import constants
from utils.sdnenvironment import SdnEnvironment
from utils.rl_agent import DQNAgent
import numpy as np


def main():
    env = SdnEnvironment()
    state_size = constants.STATE_SIZE
    action_size = constants.ACTION_SIZE
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32
    episodes = constants.EPISODES
    for e in range(episodes):
        state = env.reset()
        print("EPISODE", e)
        state = np.reshape(state, [1, state_size])
        for time in range(constants.NUM_SIMULATIONS):
            print("Episode", e, "Time", time)
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, episodes, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)


if __name__ == "__main__":
    main()
