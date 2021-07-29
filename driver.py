import os
import time
import matplotlib.pyplot as plt

import QueueNet2
import numpy as np



def main():

    # bsizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


    pdrops = []
    delay1s = []
    delay2s = []
    sw_pdrops = []
    throughputs = []
    mode = "dynamic"
    if mode == "static":
        bsizes = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 1000, 3000, 10000, 20000]
        for bsize in bsizes:
            delay1, delay2, pdrop, sw_pdrop, throughput = QueueNet2.main(bsize)
            pdrops.append(pdrop)
            delay1s.append(delay1)
            delay2s.append(delay2)
            sw_pdrops.append(sw_pdrop)
            throughputs.append(throughput)

    else:
        bsize = 100  # initial
        bsizes = []
        for i in range(12):
            delay1, delay2, pdrop, sw_pdrop, throughput = QueueNet2.main(bsize)
            pdrops.append(pdrop)
            delay1s.append(delay1)
            delay2s.append(delay2)
            sw_pdrops.append(sw_pdrop)
            throughputs.append(throughput)
            reward = 0
            if (delay1 + delay2) > (delay1s[-1]+delay2s[-1]):
                reward -= 10
            else:
                reward += 10

            if throughput < throughputs[-1]:
                reward -= 20
            else:
                reward += 20

            bsize += (reward * 50)

            print("Buffer size is now ", bsize)
            bsizes.append(bsize)

    # print(delay1s, delay2s, pdrops, sw_pdrops)

    plt.figure(1)
    plt.title("Delay 1")
    plt.plot(np.array(bsizes), np.array(delay1s))
    plt.savefig("delay1.png")
    plt.show()


    plt.figure(2)
    plt.title("Delay 2")
    plt.plot(np.array(bsizes), np.array(delay2s))
    plt.savefig("delay2.png")
    plt.show()


    plt.figure(3)
    plt.title("Packet drop %")
    plt.plot(np.array(bsizes), np.array(pdrops))
    plt.savefig("pdrop.png")
    plt.show()

    plt.figure(4)
    plt.title("Switch packet drop %")
    plt.plot(np.array(bsizes), np.array(sw_pdrops))
    plt.savefig("sw_pdrop.png")
    plt.show()

    plt.figure(5)
    plt.title("Throughput")
    plt.plot(np.array(bsizes), np.array(throughputs))
    plt.savefig("throughput.png")
    plt.show()


if __name__ == '__main__':
    main()
