import os
from utils import constants
from utils.sdnenvironment import SdnEnvironment
from utils.rl_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt


def main():
    alter_gamma = False
    alter_initial_bsize = False
    if alter_gamma:
        gamma_values = np.arange(0.1, 1.0, 0.05)
    else:
        gamma_values = [1.0]

    if alter_initial_bsize:
        initial_bsizes = [(i*50) for i in range(constants.EPISODES)]
    else:
        initial_bsizes = [constants.INITIAL_BSIZE for i in range(constants.EPISODES)]

    for gamma in gamma_values:
        sdn_env = SdnEnvironment()
        state_size = constants.STATE_SIZE
        action_size = constants.ACTION_SIZE
        rl_agent = DQNAgent(state_size, action_size,gamma)

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

            script_dir = os.path.dirname(os.path.abspath(__file__))
            graphs_dir = os.path.join(script_dir, "graphs")
            results_dir = os.path.join(graphs_dir, "gamma {}".format(gamma))


            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            plt.figure(1)
            plt.title("Buffer sizes, episode {}".format(e + 1))
            plt.plot(np.array(sdn_env.bsizes))
            filename = "bsizes_episode{}.png".format(e+1)
            plt.savefig(os.path.join(results_dir, filename))
            plt.show()

            plt.figure(2)
            plt.title("Delay, episode {}".format(e+1))
            plt.plot(np.array(sdn_env.delays))
            filename = "delay_episode{}.png".format(e+1)
            plt.savefig(os.path.join(results_dir, filename))
            plt.show()

            plt.figure(3)
            plt.title("Packet drop %, episode {}".format(e+1))
            plt.plot(np.array(sdn_env.pdrops))
            filename = "pdrop_episode{}.png".format(e + 1)
            plt.savefig(os.path.join(results_dir, filename))
            plt.show()

            plt.figure(4)
            plt.title("Throughput, episode {}".format(e+1))
            plt.plot(np.array(sdn_env.throughputs))
            filename = "throughput_episode{}.png".format(e + 1)
            plt.savefig(os.path.join(results_dir, filename))
            plt.show()

            plt.figure(5)
            plt.title("Reward, episode {}".format(e+1))
            plt.plot(np.array(sdn_env.rewards))
            filename = "reward_episode{}.png".format(e + 1)
            plt.savefig(os.path.join(results_dir, filename))
            plt.show()

if __name__ == "__main__":
    main()
