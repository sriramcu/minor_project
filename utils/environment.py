import QueueNet2
import constants


class environment:
    def __init__(self) -> None:
        self.bsize = 10
        self.pdrops = []
        self.delays = []
        self.throughputs = []
        self.delay = constants.INITIAL_DELAY
        self.pdrop = constants.INITIAL_PDROP
        self.throughput = constants.INITIAL_THROUGHPUT

    def step(self, action):
        reward = 0

        delay_reward = (self.delays[-1] - self.delay) * constants.DELAY_REWARD
        pdrop_reward = (self.throughput - self.throughputs[-1]) * constants.PACKET_DROP_REWARD
        throughput_reward = (self.pdrops[-1] - self.pdrop) * constants.THROUGHPUT_REWARD

        reward = reward + delay_reward + throughput_reward + pdrop_reward

        self.bsize += int(reward * constants.BSIZE_MULTIPLIER)
        self.delay, self.pdrop, self.throughput = QueueNet2.simulate_network(self.bsize)
        self.delays.append(self.delay)
        self.pdrops.append(self.pdrop)
        self.throughputs.append(self.throughput)
        print("Buffer size is now ", self.bsize)
        done = False
        return [self.delay, self.pdrop, self.throughput], reward, done, 0

    def reset(self):
        self.delay, self.pdrop, self.throughput = QueueNet2.simulate_network(self.bsize)
        self.delays.append(self.delay)
        self.pdrops.append(self.pdrop)
        self.throughputs.append(self.throughput)
        return [self.delay, self.pdrop, self.throughput]
