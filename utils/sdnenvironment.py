import QueueNet2
import constants


class SdnEnvironment:
    def __init__(self) -> None:
        self.bsize = constants.INITIAL_BSIZE
        self.reward = 0
        self.pdrops = []
        self.delays = []
        self.throughputs = []
        self.bsizes = []
        self.rewards = []
        self.headers = []
        self.lengths = []


    def step(self, action):
        # if action is 1 then increase buffer size
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


        if action == 1:
            self.bsize += constants.BSIZE_CHANGE
        else:
            self.bsize -= constants.BSIZE_CHANGE

        self.bsize = max(self.bsize, 1)

        delay, pdrop, throughput = QueueNet2.simulate_network(self.bsize)
        self.delays.append(delay)
        self.pdrops.append(pdrop)
        self.throughputs.append(throughput)
        self.bsizes.append(self.bsize)
        self.rewards.append(self.reward)

        return [delay, pdrop, throughput], self.reward, done, 0

    def reset(self):
        delay, pdrop, throughput = QueueNet2.simulate_network(self.bsize)
        self.delays.append(delay)
        self.pdrops.append(pdrop)
        self.throughputs.append(throughput)
        self.bsizes.append(self.bsize)
        return [delay, pdrop, throughput]

    def render(self):
        if len(self.delays) == 1:
            self.headers = ["Sl No", "Buffer Size", "Delay", "Packet Drop", "Throughput"]
            self.lengths = [len(x)+5 for x in self.headers]
            for i in range(len(self.headers)):
                print(self.headers[i].ljust(self.lengths[i]), end='')
            print()

        print(str(len(self.delays)).ljust(self.lengths[0]),  end='')
        print(str(self.bsizes[-1]).ljust(self.lengths[1]), end='')
        print(str(self.delays[-1]).ljust(self.lengths[2]), end='')
        print(str(self.pdrops[-1]).ljust(self.lengths[3]), end='')
        print(str(self.throughputs[-1]).ljust(self.lengths[4]), end='')
        print()
