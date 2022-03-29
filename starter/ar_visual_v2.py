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
task_list=["forward_1.5_mixed","forward_2.5_mixed","forward_3.5_mixed","forward_4.5_mixed","forward_5.5_mixed","forward_6.5_mixed","forward_7.5_mixed","forward_8.5_mixed","forward_9.5_mixed"]
task_list=["forward_5.5_mixed"]
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
	# env.seed(args.seed)
	env.reset(seed=args.seed)
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
		average_v_file = open(average_v_csv_path,"a")
		average_v_writer = csv.writer(average_v_file)
		average_v_writer.writerow(["task","v_mean","v_std"])

	pre_embeddings = []
	pre_embedding = torch.Tensor([14.38648,-9.580958,-16.041727,-1.164166,12.3197365,3.672143,-3.1414335,-4.0125556,-6.67028,0.63735306,-60.31507,-26.34062,-53.79265,16.057978,2.2778113,3.4116797]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding = torch.Tensor([4.8981857,-2.1659336,3.6557002,-2.8433738,14.906114,5.482699,-2.4826305,5.823681,1.2792983,-2.8405387,-25.147417,-27.718727,-52.27161,-6.88948,-0.24129571,8.554256]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)	
	pre_embedding = torch.Tensor([4.5458994,-2.4300458,1.1819551,0.26374474,10.243903,-1.2598388,0.9122858,6.1226835,-5.1163387,2.9107726,-13.963078,-36.49345,-43.173885,0.9039712,-0.21359862,-4.856464]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)	
	pre_embedding = torch.Tensor([0.750622,-0.7132483,-1.2707405,-0.19877918,14.4466,1.3527036,-2.9070518,3.2331638,-3.613355,0.078160346,-5.9566245,-22.410223,-27.49646,13.948006,0.009050498,6.310787]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)	
	pre_embedding = torch.Tensor([2.0344,-0.18211967,-1.016011,-0.58528906,9.130649,0.6399375,3.0131373,6.2298174,-0.51527363,-0.891654,-5.2474966,-22.75474,-25.72084,13.634716,0.80226827,11.896404]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding = torch.Tensor([-0.9266835,-0.6330202,-1.6333201,0.63965774,6.352248,0.66435474,0.8719162,2.4231133,1.841162,1.8156424,-7.528089,-24.090658,-28.71157,13.208413,-2.377902,13.784845]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding = torch.Tensor([-1.807375,-0.2934243,-0.53428394,1.122416,3.113141,0.43747795,2.2722063,2.1431649,-1.4136422,-0.55535513,-7.9582887,-23.233164,-28.072113,14.592276,-2.953101,14.686382]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding = torch.Tensor([-0.095981285,0.007204056,-0.6657414,-1.8221579,1.1506329,-0.12960353,1.6366785,0.75239086,-2.0837288,0.45380038,-6.7266593,-24.581785,-28.539667,14.750387,-2.41665,16.084827]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding = torch.Tensor([0.8919491,0.50149125,-0.11360723,0.42947873,-0.5063789,0.54685724,0.14096078,0.09657806,-0.5318501,0.5202051,-6.641752,-25.089462,-28.634327,13.858293,0.71471775,18.007915]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding = torch.Tensor([-0.93191594,-0.41936427,-0.07612151,-0.059824474,-1.2776589,1.6989226,1.0947989,-2.3466268,-1.4157791,2.0586221,-7.58253,-28.525173,-30.30743,14.7991085,2.3285377,18.076374]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)

	embeddings = []
	for i in range(task_num):
		embedding = 0.5 * pre_embeddings[i] + 0.5 * pre_embeddings[i+1]
		embeddings.append(embedding)
	# embedding = torch.Tensor([91.18837,123.0279,-117.92559,74.70717,-0.8363521,237.00586,143.59999,257.87402,585.4402,28.773743,53.124676,-471.30557,-26.860031,-317.3784,119.19396,124.50042]).unsqueeze(0)
	# embeddings=[]
	# embeddings.append(embedding)


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
				embedding=torch.Tensor([93.40412,160.58662,-150.15569,91.56476,25.814232,176.67238,228.76828,228.39816,673.3303,81.97838,200.58913,-530.2282,192.42592,-315.63486,107.21659,83.910355]).unsqueeze(0)
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


