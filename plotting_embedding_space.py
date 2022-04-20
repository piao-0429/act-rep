import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import time
from datetime import datetime
from PIL import Image
import csv
import sys

sys.path.append(".") 
import torch
import os
import time
import os.path as osp
import numpy as np
from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env


import torchrl.policies as policies
import torchrl.networks as networks
import gym

from math import pi, cos, sin


args = get_args()
params = get_params(args.config)
env_name = params['env_name']

experiment_id = str(args.id)
experiment_id_v2 = experiment_id + "_embedding_space"






device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")


torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.backends.cudnn.deterministic=True

velocity_dir = "velocity"+'/'
if not os.path.exists(velocity_dir):
    os.makedirs(velocity_dir)
velocity_dir = velocity_dir + '/' + experiment_id_v2 + '/'
if not os.path.exists(velocity_dir):
    os.makedirs(velocity_dir)
velocity_dir = velocity_dir + '/' + env_name  + '/'
if not os.path.exists(velocity_dir):
    os.makedirs(velocity_dir)
velocity_dir = velocity_dir + '/' + str(args.seed) + '/'
if not os.path.exists(velocity_dir):
    os.makedirs(velocity_dir)	

embedding_space_csv_path = velocity_dir+ "/embedding_velocity_record.csv"
embedding_space_file = open(embedding_space_csv_path,"r")
embedding_space_list = np.loadtxt(embedding_space_file, delimiter = ",")

print(embedding_space_list)

x = embedding_space_list[:, 0]
y = embedding_space_list[:, 1]
z = embedding_space_list[:, 2]
label = embedding_space_list[:, 5]


ax = plt.gca(projection='3d') 


ax.scatter(x, y, z, c=label)


ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()