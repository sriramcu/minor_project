STATE_SIZE = 3
ACTION_SIZE = 2
EPISODES = 1
TIME_SLICE = 4000
# Assumed network average values to start,
# do not assume extreme values
# so that reward doesn't get lopsided
INITIAL_DELAY = 5000
INITIAL_PDROP = 20
INITIAL_THROUGHPUT = 100
INITIAL_BSIZE = 50
# Rewards for the simpler absolute change system
DELAY_REWARD = 50
PDROP_REWARD = 50
THROUGHPUT_REWARD = 50
# Multiplier for obtaining resultant reward for relative change system
DELAY_REWARD_MULTIPLIER = 1
THROUGHPUT_REWARD_MULTIPLIER = 50
PACKET_DROP_REWARD_MULTIPLIER = 50
BSIZE_MULTIPLIER = 5  # multiply reward by this amount and add to the buffer size
BSIZE_CHANGE = 10  # add or subtract buffer size by this amount based on action
NUM_SIMULATIONS = 300  # number of simulations per episode
OPTIMAL_DELAY = 100
OPTIMAL_PDROP = 25
OPTIMAL_THROUGHPUT = 1
