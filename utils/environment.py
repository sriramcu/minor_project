import QueueNet2

class environment:
    def __init__(self) -> None:
        self.bsize = 10
        self.pdrops = []
        self.delay1s = []
        self.delay2s = []
        self.sw_pdrops = []
        self.throughputs = []
        

    def step(self, action):

        reward = 0
        if (self.delay1 + self.delay2) > (self.delay1s[-1]+self.delay2s[-1]):
            reward -= 10
        else:
            reward += 10

        if self.throughput < self.throughputs[-1]:
            reward -= 20
        else:
            reward += 20

        self.bsize += (reward * 5)
        self.delay1, self.delay2, self.pdrop, self.sw_pdrop, self.throughput = QueueNet2.main(self.bsize)
        self.pdrops.append(self.pdrop)
        self.delay1s.append(self.delay1)
        self.delay2s.append(self.delay2)
        self.sw_pdrops.append(self.sw_pdrop)
        self.throughputs.append(self.throughput)
        print("Buffer size is now ", bsize)

        return [self.delay1, self.delay2, self.pdrop, self.sw_pdrop, self.throughput], reward, False, 0



    def reset(self):
        
        self.delay1, self.delay2, self.pdrop, self.sw_pdrop, self.throughput = QueueNet2.main(self.bsize)
        self.pdrops.append(self.pdrop)
        self.delay1s.append(self.delay1)
        self.delay2s.append(self.delay2)
        self.sw_pdrops.append(self.sw_pdrop)
        self.throughputs.append(self.throughput)
        return [self.delay1, self.delay2, self.pdrop, self.sw_pdrop, self.throughput]


