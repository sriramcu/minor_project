"""
Use of SimComponents to simulate the network of queues from Homework #6 problem 1, Fall 2014.
See corresponding solution set for mean delay calculation based on Burkes theorem.

Copyright 2014 Dr. Greg M. Bernstein
Released under the MIT license
"""
import random
import functools

import simpy

from SimComponents import PacketGenerator, PacketSink, SwitchPort, RandomBrancher


def main(custom_queue=300):
    # Set up arrival and packet size distributions
    # Using Python functools to create callable functions for random variates with fixed parameters.
    # each call to these will produce a new random value.
    mean_pkt_size = 100.0  # in bytes
    adist1 = functools.partial(random.expovariate, 2.0)
    adist2 = functools.partial(random.expovariate, 0.5)
    adist3 = functools.partial(random.expovariate, 0.6)
    sdist = functools.partial(random.expovariate, 1.0 / mean_pkt_size)
    samp_dist = functools.partial(random.expovariate, 0.50)
    port_rate = 2.2 * 8 * mean_pkt_size  # want a rate of 2.2 packets per second

    # Create the SimPy environment. This is the thing that runs the simulation.
    env = simpy.Environment()

    # Create the packet generators and sink
    def selector(pkt):
        return pkt.src == "SJSU1"

    def selector2(pkt):
        return pkt.src == "SJSU2"

    def selector3(pkt):
        return pkt.src == "SJSU1" or pkt.src == "SJSU2" or pkt.src == "SJSU3"

    ps1 = PacketSink(env, debug=False, rec_arrivals=True, selector=selector3)
    ps2 = PacketSink(env, debug=False, rec_waits=True, selector=selector3)
    pg1 = PacketGenerator(env, "SJSU1", adist1, sdist)
    pg2 = PacketGenerator(env, "SJSU2", adist2, sdist)
    pg3 = PacketGenerator(env, "SJSU3", adist3, sdist)
    branch1 = RandomBrancher(env, [0.75, 0.25])
    branch2 = RandomBrancher(env, [0.65, 0.35])

    switch_port1 = SwitchPort(env, port_rate, qlimit=custom_queue)
    switch_port2 = SwitchPort(env, port_rate, qlimit=custom_queue)
    switch_port3 = SwitchPort(env, port_rate, qlimit=custom_queue)
    switch_port4 = SwitchPort(env, port_rate, qlimit=custom_queue)

    # Wire packet generators, switch ports, and sinks together
    pg1.out = switch_port1
    switch_port1.out = branch1
    branch1.outs[0] = switch_port2
    switch_port2.out = branch2
    branch2.outs[0] = switch_port3
    branch2.outs[1] = switch_port4
    pg3.out = switch_port3
    pg2.out = switch_port4
    switch_port3.out = ps1
    switch_port4.out = ps2
    # Run it
    env.run(until=4000)
    print(ps2.waits[-10:])
    # print pm.sizes[-10:]
    # print ps.arrivals[-10:]
    delay1 = sum(ps1.waits) / len(ps1.waits)
    print("average wait source 1 to output 3 = {}".format(delay1))
    delay2 = sum(ps2.waits) / len(ps2.waits)
    print("average wait source 2 to output 4 = {}".format(delay2))
    print("packets sent {}".format(pg1.packets_sent + pg2.packets_sent))
    print("packets received: {}".format(len(ps2.waits)))
    packets_sent = pg1.packets_sent + pg2.packets_sent + pg3.packets_sent
    packets_received = len(ps1.waits) + len(ps2.waits)
    pdrop = ((packets_sent-packets_received)/packets_sent) * 100

    pdrop = round(pdrop, 2)
    print("Packet drop = {} %".format(pdrop))

    return delay1, delay2, pdrop
    # print "average system occupancy: {}".format(float(sum(pm.sizes))/len(pm.sizes))



if __name__ == '__main__':
    main(30000)


