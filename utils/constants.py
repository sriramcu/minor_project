STATE_SIZE = 3
ACTION_SIZE = 2
EPISODES = 1000
TIME_SLICE = 4000
# Assumed network average values to start,
# do not assume extreme values
# so that reward doesn't get lopsided
INITIAL_DELAY = 5000
INITIAL_PDROP = 20
INITIAL_THROUGHPUT = 100
INITIAL_BSIZE = 50
# Multiplier for obtaining resultant reward
DELAY_REWARD = 1
THROUGHPUT_REWARD = 1000
PACKET_DROP_REWARD = 50
BSIZE_MULTIPLIER = 5  # multiply reward by this amount and add to the buffer size
BSIZE_CHANGE = 50  # add or subtract buffer size by this amount based on action
NUM_SIMULATIONS = 500  # number of simulations per episode
OPTIMAL_DELAY = 100
OPTIMAL_PDROP = 25
OPTIMAL_THROUGHPUT = 1
