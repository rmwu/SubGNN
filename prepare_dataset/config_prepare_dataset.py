from pathlib import Path
import sys

sys.path.insert(0, '..') # add config to path
import config as general_config

# Output directory ('density' as an example)
DATASET_DIR = Path(general_config.PROJECT_ROOT) / "covid_pairs"

# Flags
GENERATE_SYNTHETIC_G = False  # whether to generate synthetic graph with below specified properties
GENERATE_NODE_EMB = True  # whether to generate node embeddings

# Random Seed
RANDOM_SEED = 226

# Parameters for training node embeddings for base graph
CONV = "gcn_graphsaint"
MINIBATCH = "GraphSaint"
POSSIBLE_BATCH_SIZES = [64, 128, 256]
POSSIBLE_HIDDEN = [128]
POSSIBLE_OUTPUT = [64]
POSSIBLE_LR = [0.001]
POSSIBLE_WD = [5e-4]
POSSIBLE_DROPOUT = [0.1]
POSSIBLE_NB_SIZE = [-1]
POSSIBLE_NUM_HOPS = [1]
POSSIBLE_WALK_LENGTH = [32]
POSSIBLE_NUM_STEPS = [32]
EPOCHS = 100

# Flags for precomputing similarity metrics
CALCULATE_SHORTEST_PATHS = True # Calculate pairwise shortest paths between all nodes in the graph
CALCULATE_DEGREE_SEQUENCE = True # Create a dictionary containing degrees of the nodes in the graph
CALCULATE_EGO_GRAPHS = True # Calculate the 1-hop ego graph associated with each node in the graph
OVERRIDE = False # Overwrite a similarity file even if it exists
N_PROCESSSES = 30 # Number of cores to use for multi-processsing when precomputing similarity metrics


