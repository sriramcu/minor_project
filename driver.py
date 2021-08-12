import os
import time
import matplotlib.pyplot as plt

from utils import QueueNet2
import numpy as np



def main():

    # bsizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    pdrops = []
    delays = []
    throughputs = []
    mode = "dynamic"
    if mode == "static":
        bsizes = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 1000, 3000, 10000, 20000]
        for bsize in bsizes:
            delay, pdrop, throughput = QueueNet2.simulate_network(bsize)
            pdrops.append(pdrop)
            delays.append(delay)
            throughputs.append(throughput)

    else:
        bsize = 10  # initial
        bsizes = []
        for i in range(12):
            delay, pdrop, throughput = QueueNet2.simulate_network(bsize)
            pdrops.append(pdrop)
            delays.append(delay)
            throughputs.append(throughput)
            reward = 0
            if delay > delays[-1]:
                reward -= 10
            else:
                reward += 10

            if throughput < throughputs[-1]:
                reward -= 20
            else:
                reward += 20

            bsize += (reward * 5)

            print("Buffer size is now ", bsize)
            bsizes.append(bsize)

    plt.figure(1)
    plt.title("Delay")
    plt.plot(np.array(bsizes), np.array(delays))
    plt.savefig("graphs/delay.png")
    plt.show()

    plt.figure(2)
    plt.title("Packet drop %")
    plt.plot(np.array(bsizes), np.array(pdrops))
    plt.savefig("graphs/pdrop.png")
    plt.show()

    plt.figure(3)
    plt.title("Throughput")
    plt.plot(np.array(bsizes), np.array(throughputs))
    plt.savefig("graphs/throughput.png")
    plt.show()


if __name__ == '__main__':
    main()
