from ronin import Ronin
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class ReducedRonin(Ronin):
	def __init__(self, data_id, data_path, output_path, sample_interval, write_flag=False):
		super(ReducedRonin, self).__init__(data_id, data_path, output_path, sample_interval, write_flag=False)
		print(output_path)
		
	def load_original_traj(self):
		mag, theta = [], []
		with open(self.output_path+"c_mag_theta.txt") as f:
			for i, line in enumerate(f):
				if i==0:
					continue
				line = line.strip()
				tokens = [float(token) for token in line.split('\t') if token.strip() != '']
				mag.append(tokens[0])
				theta.append(tokens[1])
		mag = np.array(mag)
		theta = np.array(theta)
		self.start_xy = np.asarray([0,0])
		return mag, theta


	def read_reduced_ronin(self, joint=True):
		self.mag, self.theta = self.load_original_traj()
		data = np.load(self.output_path + "rssi_reduced.npz")
		self.rssi = data["rssi"]
		self.rssi[np.isnan(self.rssi)] = -100

		self.haswifi = data["haswifi"]

		# print(self.mag.shape, self.theta.shape)
		# print(self.rssi.shape, self.haswifi.shape)

	def update_traj(self, bias):
		updated_x = np.cos(self.theta+bias)*self.mag
		updated_y = np.sin(self.theta+bias)*self.mag
		tmp = np.stack([updated_x, updated_y],-1)
		tmp = np.cumsum(tmp, axis=0)
		upt_traj = np.zeros((tmp.shape[0]+1,2))
		upt_traj[1:,:]= tmp
		upt_traj = upt_traj+self.start_xy
		return upt_traj

	def visulize_result(self, ransac=True):
		ori_traj = self.update_traj(0)
		if ransac:
			self.load_align_result("c_single_corres_align_ransac.txt")
			save_name = "./experiments/singlesearch/"+self.day_id+"-align-ransac.png"
		else:
			self.load_align_result("c_single_corres_align.txt")
			save_name = "./experiments/singlesearch/"+self.day_id+"-align.png"
		new_traj = self.aligned_ronin
		print(ori_traj.shape, new_traj.shape)
		
		plot_traj_pair(ori_traj, new_traj, save_name=save_name, third=None)



def plot_traj_pair(t1, t2, save_name=None, third=None):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(t1[:,0], t1[:,1], color=(0,1,0), s=2)
	ax.scatter(t2[:,0], t2[:,1], color=(0,0,1), s=2)
	if third is not None:
		ax.scatter(third[:,0], third[:,1], color=(1,0,0), s=2)
	ax.axis('equal')
	plt.tight_layout()
	plt.savefig(save_name)
	plt.close(fig)

def huber_func(x, k):
	abs_x = np.absolute(x)
	result = k/(abs_x+1e-6)
	result[abs_x<=k] = 1
	return result
	
def tukey_func(x, b):
	abs_x = np.absolute(x)
	result = (1-(x**2)/(b**2))**2
	result[abs_x>b] = 0
	return result

def rssi_metric(x, y):
	flag = np.logical_and(x.astype(np.int)==-100, y.astype(np.int)==-100)
	flag = np.logical_not(flag)
	valid_n = np.sum(flag)
	if valid_n==0:
		return 0
	huber_para=10
	tukey_para=160
	t1 = np.absolute(x-y)
	t2 = np.absolute((x+y)/2)
	dist = huber_func(t1, huber_para)*tukey_func(t2, tukey_para)
	
	return np.sum(dist[flag])/np.sum(flag)
	

def align_traj_single(ref, idx, output_path):
	save_dir = "./experiments/singlesearch/"
	
	# ref is fixed, src is changed
	ref_traj = ref.update_traj(0)
	ref_traj_ori = ref_traj.copy()
	ori_ids = np.arange(ref_traj_ori.shape[0])
	ref_traj = ref_traj[ref.haswifi,:]
	ori_ids = ori_ids[ref.haswifi]
	
	knn_num = 10
	nbrs = NearestNeighbors(n_neighbors=knn_num, algorithm='ball_tree', metric='manhattan').fit(ref.rssi)
	distances, indices = nbrs.kneighbors(ref.rssi)
	n = indices.shape[0]
	corres_ids = []
	for i in range(n):
		flag = np.absolute(indices[i,:]-i)>n*0.1
		matchid = indices[i,flag]
		distance_match = distances[i,flag]
		if len(matchid)>0 and distance_match[0]>0.1 and distance_match[0]<150:
			corres_ids.append([ori_ids[i], ori_ids[matchid[0]], distance_match[0]])

	if len(corres_ids)==0:
		return
	corres_ids = np.asarray(corres_ids).reshape(-1,3)
	print(np.mean(corres_ids[:,2]),np.median(corres_ids[:,2]))
	np.savetxt(output_path + "c_single_corres.txt", corres_ids, delimiter='\t', header=str(corres_ids.shape[0]))

	
	tmp_traj1 = ref_traj_ori[corres_ids[:,0].astype(np.int),:]
	tmp_traj2 = ref_traj_ori[corres_ids[:,1].astype(np.int),:]
	c_traj1, c_traj2 = tmp_traj1[:,:], tmp_traj2[:,:]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(ref_traj_ori[:,0], ref_traj_ori[:,1], color=(0,1,0), s=2)
	for i in range(c_traj1.shape[0]):
		ax.plot([c_traj1[i,0], c_traj2[i,0]], [c_traj1[i,1], c_traj2[i,1]])
	ax.axis('equal')
	plt.tight_layout()
	plt.savefig(save_dir+f"{ref.day_id}.png")
	plt.close(fig)
	return




def align_traj_pair(ref, src, idx):
	save_dir = "./experiments/search/"
	
	# ref is fixed, src is changed
	ref_traj = ref.update_traj(0)
	ref_traj = ref_traj[ref.haswifi,:]
	src_traj = src.update_traj(0)
	src_traj = src_traj[src.haswifi,:]
	knn_num = 1
	nbrs = NearestNeighbors(n_neighbors=knn_num, algorithm='ball_tree', metric='manhattan').fit(ref.rssi)
	distances, indices = nbrs.kneighbors(src.rssi)
	indices = indices.reshape(-1)
	tmp_traj = ref_traj[indices,:]
	c_traj1, c_traj2 = tmp_traj[0::10,:], src_traj[0::10,:]


	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(ref.aligned_ronin[:,0], ref.aligned_ronin[:,1], color=(0,1,0), s=2)
	ax.scatter(src.aligned_ronin[:,0], src.aligned_ronin[:,1], color=(0,0,1), s=2)
	for i in range(c_traj1.shape[0]):
		ax.plot([c_traj1[i,0], c_traj2[i,0]], [c_traj1[i,1], c_traj2[i,1]])
	ax.axis('equal')
	plt.tight_layout()
	plt.savefig(save_dir+f"{idx}.png")
	plt.close(fig)
	return


	min_error, min_bias = 1e100, 0
	for bias in np.arange(-np.pi,np.pi,0.1):
		# update src position
		src_traj = src.update_traj(bias)
		#print(np.amax(np.absolute(src_traj[1:,:]-src.aligned_ronin)))

		src_traj = src_traj[src.haswifi,:]
		distances, indices = nbrs.kneighbors(src_traj)
		indices = indices.reshape(-1)
		diff = ref.rssi[indices,:] - np.repeat(src.rssi, knn_num, 0)
		dist = np.mean(np.absolute(diff),1)
		match_error = np.mean(dist)
		# print(bias, match_error)
		if match_error<min_error:
			min_error = match_error
			min_bias = bias
	src_traj = src.update_traj(min_bias)
	plot_traj_pair(ref.aligned_ronin, src_traj, third=src.aligned_ronin, save_name=save_dir+f"{idx}.png")
		

def align_traj_pair1(ref, src, idx):
	save_dir = "./experiments/search/"
	#plot_traj_pair(ref.aligned_ronin, src.aligned_ronin, save_dir+f"{idx}_before.png")
	# ref is fixed, src is changed
	npts = ref.aligned_ronin.shape[0]+1
	ref_traj = np.zeros((npts,2))
	ref_traj[0,:] = ref.start_xy
	ref_traj[1:,:] = ref.aligned_ronin
	ref_traj = ref_traj[ref.haswifi,:]
	knn_num = 3
	nbrs = NearestNeighbors(n_neighbors=knn_num, algorithm='ball_tree').fit(ref_traj)
	#distances, indices = nbrs.kneighbors(X)

	min_error, min_bias = 1e100, 0
	for bias in np.arange(-np.pi,np.pi,0.1):
		# update src position
		src_traj = src.update_traj(bias)
		#print(np.amax(np.absolute(src_traj[1:,:]-src.aligned_ronin)))

		src_traj = src_traj[src.haswifi,:]
		distances, indices = nbrs.kneighbors(src_traj)
		indices = indices.reshape(-1)
		diff = ref.rssi[indices,:] - np.repeat(src.rssi, knn_num, 0)
		dist = np.mean(np.absolute(diff),1)
		match_error = np.mean(dist)
		# print(bias, match_error)
		if match_error<min_error:
			min_error = match_error
			min_bias = bias
	src_traj = src.update_traj(min_bias)
	plot_traj_pair(ref.aligned_ronin, src_traj, third=src.aligned_ronin, save_name=save_dir+f"{idx}.png")
		


def penality_func(x):
	pass