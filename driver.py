import os
import time
import matplotlib.pyplot as plt

import QueueNet2
import numpy as np



def main():

    # bsizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    bsizes = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    pdrops = []
    delay1s = []
    delay2s = []
    for bsize in bsizes:
        delay1, delay2, pdrop = QueueNet2.main(bsize)
        pdrops.append(pdrop)
        delay1s.append(delay1)
        delay2s.append(delay2)

    print(delay1s, delay2s, pdrops)

    plt.figure(1)
    plt.title("Delay 1")
    plt.plot(np.array(bsizes), np.array(delay1s))
    plt.show()
    plt.figure(2)
    plt.title("Delay 2")
    plt.plot(np.array(bsizes), np.array(delay2s))
    plt.show()
    plt.figure(3)
    plt.title("Packet drop %")
    plt.plot(np.array(bsizes), np.array(pdrops))
    plt.show()

if __name__ == '__main__':
    main()
