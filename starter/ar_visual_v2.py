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
from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)  # Create a window to init GLFW.




args = get_args()
params = get_params(args.config)
env=gym.make(params['env_name'])
# task_list=["forward_5","forward_6","forward_7","forward_8","forward_9","forward_10"]
task_list=["forward_1.5_mixed","forward_2.5_mixed","forward_3.5_mixed","forward_4.5_mixed","forward_5.5_mixed","forward_6.5_mixed","forward_7.5_mixed","forward_8.5_mixed"]
task_num=len(task_list)
representation_shape= params['representation_shape']
embedding_shape=params['embedding_shape']
params['p_state_net']['base_type']=networks.MLPBase
params['p_task_net']['base_type']=networks.MLPBase
params['p_action_net']['base_type']=networks.MLPBase
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
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
model_dir="log/"+experiment_id+"/"+params['env_name']+"/"+str(args.seed)+"/model/"

pf_state.load_state_dict(torch.load(model_dir + "model_pf_state_finish.pth", map_location='cpu'))
pf_action.load_state_dict(torch.load(model_dir + "model_pf_action_finish.pth", map_location='cpu'))


############################# save images for gif ##############################


def save_gif_images(env_name, max_ep_len):

	print("============================================================================================")
	device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	if args.cuda:
		torch.backends.cudnn.deterministic=True

	# make directory for saving gif images
	gif_images_dir = "gif_images" + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	gif_images_dir = gif_images_dir + '/' + experiment_id + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	# make environment directory for saving gif images
	gif_images_dir = gif_images_dir + '/' + env_name + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	gif_images_dir_list=[]
	
	for i in range(len(task_list)):
		# gif_images_dir_list[i]=gif_images_dir+"/"+cls_list[i]+"/"
		gif_images_dir_list.append(gif_images_dir+"/"+task_list[i]+"/")
		if not os.path.exists(gif_images_dir_list[i]):
			os.makedirs(gif_images_dir_list[i])

	# make directory for gif
	gif_dir = "gifs" + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + experiment_id + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	# make environment directory for gif
	gif_dir = gif_dir + '/' + env_name  + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir_list=[]
	
	for i in range(len(task_list)):
        # gif_dir_list[i]=gif_dir+"/"+cls_list[i]+"/"
		gif_dir_list.append(gif_dir+"/"+task_list[i]+"/")
		if not os.path.exists(gif_dir_list[i]):
			os.makedirs(gif_dir_list[i])

	if params["save_embedding"]:
		embed_dir = "embedding"+'/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + experiment_id + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + env_name  + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + str(args.seed) + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)	

	if params["save_velocity"]:
		velocity_dir = "velocity"+'/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + experiment_id + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + env_name  + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + str(args.seed) + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)	

		average_v_csv_path = velocity_dir+ "/average_velocity_mixed.csv"
		average_v_file = open(average_v_csv_path,"w")
		average_v_writer = csv.writer(average_v_file)
		average_v_writer.writerow(["task","v_mean","v_std"])

	pre_embeddings = []
	pre_embedding = torch.Tensor([153.79562,280.24725,-529.27454,-256.4803,418.4877,326.17126,357.36996,835.0294,305.83417,385.8792,-336.05194,-447.83795,-525.18695,14.119919,470.81238,178.08601]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding = torch.Tensor([134.76494,19.112263,-614.83606,140.54968,-54.789043,176.52586,370.16415,882.07806,542.43384,144.7351,-43.848366,-256.3044,-469.94504,-203.53123,428.9156,14.24445]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)	
	pre_embedding = torch.Tensor([109.54987,184.96585,-528.96,171.06128,10.6771145,53.088627,332.48068,650.1679,465.79703,34.77299,-128.60837,-264.37967,-379.33554,-192.87706,184.73157,-21.901499]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)	
	pre_embedding = torch.Tensor([108.81322,128.67648,-443.9819,95.23057,-9.84914,77.29975,273.8642,488.3134,373.95648,61.89438,-101.368835,-207.6508,-320.509,-217.83714,124.40808,40.202007]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)	
	pre_embedding = torch.Tensor([116.66403,119.96243,-351.09204,75.93475,-0.3881659,47.560207,210.16183,394.06943,303.57553,45.66178,-90.592026,-167.94612,-247.4873,-163.70718,117.24094,81.90116]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding = torch.Tensor([89.052956,73.54011,-262.6336,45.055637,-3.3709183,24.53887,151.59283,298.6229,228.81876,43.02485,-88.03662,-113.893166,-180.4979,-140.0946,102.57138,32.55124]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding = torch.Tensor([69.323326,74.46447,-174.82712,28.269516,-1.2693439,10.076403,87.81177,245.18869,157.76514,50.685978,-64.21518,-68.9504,-122.94437,-119.96198,106.703285,25.989784]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding = torch.Tensor([49.46687,49.43225,-118.72213,11.566086,1.7729292,8.120008,54.789127,205.96275,102.005905,61.151184,-46.54895,-45.96882,-87.423935,-98.79142,106.20102,24.958017]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding = torch.Tensor([47.273678,56.719967,-59.401848,-16.39408,-21.8002,26.747112,23.259615,186.2252,43.807724,86.055824,-40.994904,-17.574709,-45.97895,-73.92034,110.33748,19.125895]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)

	embeddings = []
	for i in range(task_num):
		embedding = 0.5 * pre_embeddings[i] + 0.5 * pre_embeddings[i+1]
		embeddings.append(embedding)
	
	for i in range(task_num):
		if params["save_embedding"]:
			embed_csv_path = embed_dir + '/' + task_list[i] + ".csv"
			embed_file = open(embed_csv_path, "w")
			embed_writer = csv.writer(embed_file)
		if params["save_velocity"]:
			velocity_csv_path = velocity_dir+ '/' + task_list[i] + ".csv"
			velocity_file = open(velocity_csv_path,'w')
			velocity_writer = csv.writer(velocity_file)

		ob=env.reset()
		with torch.no_grad():
			for t in range(1, max_ep_len+1):
				representation = pf_state.forward(torch.Tensor( ob ).to("cpu").unsqueeze(0))
				embedding = embeddings[i]
				# embedding_5 = torch.Tensor([116.66403,119.96243,-351.09204,75.93475,-0.3881659,47.560207,210.16183,394.06943,303.57553,45.66178,-90.592026,-167.94612,-247.4873,-163.70718,117.24094,81.90116]).unsqueeze(0)
				# embedding_6 = torch.Tensor([89.052956,73.54011,-262.6336,45.055637,-3.3709183,24.53887,151.59283,298.6229,228.81876,43.02485,-88.03662,-113.893166,-180.4979,-140.0946,102.57138,32.55124]).unsqueeze(0)
				# embedding = 0.5 * embedding_5 + 0.5 * embedding_6
				out=pf_action.explore(representation,embedding)
				act=out["action"]
				act = act.detach().cpu().numpy()
				next_ob, _, done, info = env.step(act)
				if params["save_velocity"]:
					x_velocity = info['x_velocity']
					velocity_writer.writerow([x_velocity])
				# img = env.render(mode = 'rgb_array')
				# img = Image.fromarray(img)
				# img.save(gif_images_dir_list[i] + '/' + experiment_id + '_' + task_list[i] + str(t).zfill(6) + '.jpg')
				ob=next_ob
				if done:
					break

		if params["save_embedding"]:
			embedding = embedding.squeeze(0)
			embedding = embedding.detach().cpu().numpy()
			embed_writer.writerow(embedding)
			embed_file.close()
		if params["save_velocity"]:
			velocity_file.close()
			velocity_file = open(velocity_csv_path,'r')
			velocity_list = np.loadtxt(velocity_file)
			velocity_list = velocity_list[100:]
			average_v_writer.writerow([task_list[i], np.mean(velocity_list), np.std(velocity_list)])


	env.close()











######################## generate gif from saved images ########################

def save_gif(env_name):

	print("============================================================================================")

	gif_num = args.seed    
	experiment_id=str(args.id)

	# adjust following parameters to get desired duration, size (bytes) and smoothness of gif
	total_timesteps = 30000
	step = 3
	frame_duration = 200

	# input images
	gif_images_dir = "gif_images/" + experiment_id + '/' + env_name +"/"
	gif_images_dir_list=[]
	for i in range(len(task_list)):
		gif_images_dir_list.append(gif_images_dir+"/"+task_list[i]+"/*.jpg")

	# output gif path
	gif_dir = "gifs"
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + experiment_id + '/' + env_name
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)
	gif_path_list=[]
	for i in range(len(task_list)):
		gif_path_list.append(gif_dir+"/"+task_list[i]+"/"+experiment_id+'_'+task_list[i]+ '_gif_' + str(gif_num) + '.gif')
	
	img_paths_list=[]
	for i in range(len(task_list)):

		img_paths_list.append(sorted(glob.glob(gif_images_dir_list[i]))) 
		img_paths_list[i] = img_paths_list[i][:total_timesteps]
		img_paths_list[i] = img_paths_list[i][::step]

		img, *imgs = [Image.open(f) for f in img_paths_list[i]]
		img.save(fp=gif_path_list[i], format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration, loop=0)
		print("saved gif at : ", gif_path_list[i])



if __name__ == '__main__':
	env_name = params["env_name"]
	max_ep_len = 20000           
	save_gif_images(env_name,  max_ep_len)
	# save_gif(env_name)


