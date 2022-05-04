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
env = gym.make(params['env_name'])
env_name = params['env_name']
representation_shape = params['representation_shape']
embedding_shape = params['embedding_shape']
params['p_state_net']['base_type'] = networks.MLPBase
params['p_task_net']['base_type'] = networks.MLPBase
params['p_action_net']['base_type'] = networks.MLPBase

pf_state = networks.Net(
	input_shape=env.observation_space.shape[0], 
	output_shape=representation_shape,
	**params['p_state_net']
)

pf_action=policies.ActionRepresentationGuassianContPolicy(
	input_shape = representation_shape + embedding_shape,
	output_shape = 2 * env.action_space.shape[0],
	**params['p_action_net'] 
)
experiment_id = str(args.id)
experiment_id_v2 = experiment_id + "_embedding_space_detail_1"
model_dir="log/"+experiment_id+"/"+params['env_name']+"/"+str(args.seed)+"/model/"

pf_state.load_state_dict(torch.load(model_dir + "model_pf_state_finish.pth", map_location='cpu'))
pf_action.load_state_dict(torch.load(model_dir + "model_pf_action_finish.pth", map_location='cpu'))



device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

env.reset(seed=args.seed)
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

embedding_space_csv_path = velocity_dir+ "/embedding_velocity_record_detail_1.csv"
embedding_space_file = open(embedding_space_csv_path,"w")
embedding_space_writer = csv.writer(embedding_space_file)
# embedding_space_writer.writerow(["embedding_x", "embedding_y", "embedding_z", "v_mean", "v_std", "label"])

embeddings = []


for theta in range(0, 180, 4):
    for fi in range(0, 360, 8):
        x = 5 * sin(theta * pi/ 180) * cos(fi * pi/ 180)
        y = 5 * sin(theta * pi/ 180) * sin(fi * pi/ 180)
        z = 5 * cos(theta * pi/ 180)
        embeddings.append([x, y ,z])
	


for embedding in embeddings:

    embedding=torch.Tensor(embedding).unsqueeze(0)	
    velocity_csv_path = velocity_dir+ '/velocity_record_detail_1.csv'
    velocity_file = open(velocity_csv_path,'w')
    velocity_writer = csv.writer(velocity_file)

    ob=env.reset()
    with torch.no_grad():
        for t in range(20000):
            representation = pf_state.forward(torch.Tensor( ob ).to("cpu").unsqueeze(0))
            out = pf_action.explore(representation,embedding)
            act = out["action"]
            act = act.detach().cpu().numpy()
            next_ob, _, done, info = env.step(act)
            
            x_velocity = info['x_velocity']
            velocity_writer.writerow([x_velocity])

            ob=next_ob
            if done:
                break


    velocity_file.close()
    velocity_file = open(velocity_csv_path,'r')
    velocity_list = np.loadtxt(velocity_file)
    velocity_list = velocity_list[100:]
    v_mean = np.mean(velocity_list)
    v_std = np.std(velocity_list)
    if (v_std < v_mean/4) and (v_mean>1 and v_mean<12) :
        embedding = embedding.squeeze(0).numpy()
        embedding_space_writer.writerow([embedding[0], embedding[1], embedding[2], v_mean])


env.close()




	


