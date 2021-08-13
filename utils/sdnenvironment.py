import QueueNet2
import constants


class SdnEnvironment:
    def __init__(self) -> None:
        self.bsize = constants.INITIAL_BSIZE
        self.reward = 0
        self.pdrops = []
        self.delays = []
        self.throughputs = []

    def step(self, action):
        done = False
        if len(self.delays) < 2:
            delay_reward = 0
            pdrop_reward = 0
            throughput_reward = 0
        else:
            delay_reward = (self.delays[-2] - self.delays[-1]) * constants.DELAY_REWARD
            pdrop_reward = (self.throughputs[-1] - self.throughputs[-2]) * constants.PACKET_DROP_REWARD
            throughput_reward = (self.pdrops[-2] - self.pdrops[-1]) * constants.THROUGHPUT_REWARD

        self.reward = self.reward + delay_reward + throughput_reward + pdrop_reward

        self.bsize += int(self.reward * constants.BSIZE_MULTIPLIER)
        self.bsize = max(self.bsize, 1)
        print(delay_reward, pdrop_reward, throughput_reward)
        print("Reward=", self.reward)
        print("Action=", action)
        print("Buffer size is now ", self.bsize)

        delay, pdrop, throughput = QueueNet2.simulate_network(self.bsize)
        self.delays.append(delay)
        self.pdrops.append(pdrop)
        self.throughputs.append(throughput)

        return [delay, pdrop, throughput], self.reward, done, 0

    def reset(self):
        delay, pdrop, throughput = QueueNet2.simulate_network(self.bsize)
        self.delays.append(delay)
        self.pdrops.append(pdrop)
        self.throughputs.append(throughput)
        return [delay, pdrop, throughput]
