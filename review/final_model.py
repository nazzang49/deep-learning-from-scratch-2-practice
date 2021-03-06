# page 66

import sys
sys.path.append("..")
import numpy as np
from review import softmax_function
from dataset import spiral
import matplotlib.pyplot as plt
from review import neural_net
import seaborn as sns

# load data
x, t = spiral.load_data()

# setting hyper parameter
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 0.01
