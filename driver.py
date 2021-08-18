from utils import constants
from utils.sdnenvironment import SdnEnvironment
from utils.rl_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt


def main():
    sdn_env = SdnEnvironment()
    state_size = constants.STATE_SIZE
    action_size = constants.ACTION_SIZE
    rl_agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32
    episodes = constants.EPISODES
    for e in range(episodes):
        state = sdn_env.reset()
        print("EPISODE", e)
        state = np.reshape(state, [1, state_size])

        for time in range(constants.NUM_SIMULATIONS):
            sdn_env.render()
            action = rl_agent.act(state)
            next_state, reward, done, _ = sdn_env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            rl_agent.memorize(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, episodes, time, rl_agent.epsilon))
                break
            if len(rl_agent.memory) > batch_size:
                rl_agent.replay(batch_size)

        plt.figure(1)
        plt.title("Buffer sizes, episode {}".format(e + 1))
        plt.plot(np.array(sdn_env.bsizes))
        plt.savefig("graphs/bsizes_episode{}.png".format(e + 1))
        plt.show()

        plt.figure(2)
        plt.title("Delay, episode {}".format(e+1))
        plt.plot(np.array(sdn_env.delays))
        plt.savefig("graphs/delay_episode{}.png".format(e+1))
        plt.show()

        plt.figure(3)
        plt.title("Packet drop %, episode {}".format(e+1))
        plt.plot(np.array(sdn_env.pdrops))
        plt.savefig("graphs/pdrop_episode{}.png".format(e+1))
        plt.show()

        plt.figure(4)
        plt.title("Throughput, episode {}".format(e+1))
        plt.plot(np.array(sdn_env.throughputs))
        plt.savefig("graphs/throughput_episode{}.png".format(e+1))
        plt.show()


if __name__ == "__main__":
    main()
