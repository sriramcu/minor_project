import os
from utils import constants
from utils.sdnenvironment import SdnEnvironment
from utils.rl_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd


def main():
    alter_gamma = True
    demo_mode = False
    if len(sys.argv) == 2 and sys.argv[1] == 'demo':
        # Colab demo
        alter_gamma = False
        demo_mode = True
        print("Demo Mode")

    alter_initial_bsize = False
    if alter_gamma:
        gamma_values = np.arange(0.1, 1.0, 0.15)
        # gamma_values = [0.7, 0.4, 0.85]
    else:
        gamma_values = [0.7]

    if alter_initial_bsize:
        initial_bsizes = [(i * 50) for i in range(constants.EPISODES)]
    else:
        initial_bsizes = [constants.INITIAL_BSIZE for _ in range(constants.EPISODES)]

    data = {'Gamma': [],
            'Min Delay': [], 'Max Delay': [], 'Avg Delay': [], 'StdDev Delay': [],
            'Min Throughput': [], 'Max Throughput': [], 'Avg Throughput': [], 'StdDev Throughput': [],
            'Min Packet Drop': [], 'Max Packet Drop': [], 'Avg Packet Drop': [], 'StdDev Packet Drop': [],
            'Min Reward': [], 'Max Reward': [], 'Avg Reward': [], 'StdDev Reward': []
            }

    for gamma in gamma_values:
        print("Gamma={}".format(gamma))
        sdn_env = SdnEnvironment()
        state_size = constants.STATE_SIZE
        action_size = constants.ACTION_SIZE
        rl_agent = DQNAgent(state_size, action_size, gamma)

        batch_size = 32
        episodes = constants.EPISODES

        for e in range(episodes):
            state = sdn_env.reset()
            print("EPISODE", (e + 1))
            state = np.reshape(state, [1, state_size])
            simulations = constants.NUM_SIMULATIONS
            if demo_mode:
                simulations = constants.DEMO_SIMULATIONS
            for time in range(simulations):
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

            if demo_mode:
                results_dir = os.path.join(graphs_dir, "demo")

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            data["Gamma"].append(gamma)
            plt.figure(1)
            plt.title("Buffer sizes, episode {}".format(e + 1))
            plt.plot(np.array(sdn_env.bsizes))
            filename = "bsizes_episode{}.png".format(e + 1)
            plt.savefig(os.path.join(results_dir, filename))
            # plt.show()
            plt.close()

            plt.figure(2)
            plt.title("Delay, episode {}".format(e + 1))
            plt.plot(np.array(sdn_env.delays))
            filename = "delay_episode{}.png".format(e + 1)
            plt.savefig(os.path.join(results_dir, filename))
            if demo_mode:
                plt.show()
            data["Min Delay"].append(min(sdn_env.delays))
            data["Max Delay"].append(max(sdn_env.delays))
            data["Avg Delay"].append(np.average(sdn_env.delays))
            data["StdDev Delay"].append(np.std(sdn_env.delays))
            plt.close()

            plt.figure(3)
            plt.title("Packet drop %, episode {}".format(e + 1))
            plt.plot(np.array(sdn_env.pdrops))
            filename = "pdrop_episode{}.png".format(e + 1)
            plt.savefig(os.path.join(results_dir, filename))
            # plt.show()
            if demo_mode:
                plt.show()
            data["Min Packet Drop"].append(min(sdn_env.pdrops))
            data["Max Packet Drop"].append(max(sdn_env.pdrops))
            data["Avg Packet Drop"].append(np.average(sdn_env.pdrops))
            data["StdDev Packet Drop"].append(np.std(sdn_env.pdrops))
            plt.close()

            plt.figure(4)
            plt.title("Throughput, episode {}".format(e + 1))
            plt.plot(np.array(sdn_env.throughputs))
            filename = "throughput_episode{}.png".format(e + 1)
            plt.savefig(os.path.join(results_dir, filename))
            # plt.show()
            if demo_mode:
                plt.show()
            data["Min Throughput"].append(min(sdn_env.throughputs))
            data["Max Throughput"].append(max(sdn_env.throughputs))
            data["Avg Throughput"].append(np.average(sdn_env.throughputs))
            data["StdDev Throughput"].append(np.std(sdn_env.throughputs))
            plt.close()

            plt.figure(5)
            plt.title("Reward, episode {}".format(e + 1))
            plt.plot(np.array(sdn_env.rewards))
            filename = "reward_episode{}.png".format(e + 1)
            plt.savefig(os.path.join(results_dir, filename))
            # plt.show()
            if demo_mode:
                plt.show()
            data["Min Reward"].append(min(sdn_env.rewards))
            data["Max Reward"].append(max(sdn_env.rewards))
            data["Avg Reward"].append(np.average(sdn_env.rewards))
            data["StdDev Reward"].append(np.std(sdn_env.rewards))
            plt.close()

    if not demo_mode:
        df = pd.DataFrame(data)
        print(df)
        df.to_csv('gamma_analysis.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
