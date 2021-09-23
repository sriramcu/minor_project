STATE_SIZE = 3
ACTION_SIZE = 2
EPISODES = 2
TIME_SLICE = 4000
DEMO_SIMULATIONS = 75
NUM_SIMULATIONS = 350  # number of simulations per episode

# Assumed network average values to start,
# do not assume extreme values
# so that reward doesn't get lopsided
INITIAL_DELAY = 5000
INITIAL_PDROP = 20
INITIAL_THROUGHPUT = 100
INITIAL_BSIZE = 250
# Rewards
DELAY_REWARD = 60
PDROP_REWARD = 0
THROUGHPUT_REWARD = 40


BSIZE_CHANGE = 20  # add or subtract buffer size by this amount based on action


# Below values are used for normalisation
MAX_DELAY = 10000
MAX_THROUGHPUT = 3
MAX_PDROP = 100
