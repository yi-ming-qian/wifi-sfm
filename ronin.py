import io
import os
import numpy as np
import statistics
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d
from scipy import interpolate
from scipy.signal import medfilt
from scipy.ndimage.measurements import label as label_mea
import imageio

class Ronin(object):
	def __init__(self, data_id, data_path, output_path, sample_interval, write_flag=True):
		self.data_id = data_id
		self.data_path = data_path
		self.output_path = output_path
		self.ronin_interval = sample_interval
		self.write_flag = write_flag
		self.day_id = os.path.basename(data_path[:-1])

	def readFromText(self, read_gyro=True, bssid_database=None, read_wifi=True):
		# ronin
		self.ronin_times, self.ronin_locations = [], []
		with open(self.data_path+"ronin.txt") as f:
			for i, line in enumerate(f):
				if i%self.ronin_interval != 0:
					continue
				line = line.strip()
				tokens = [float(token) for token in line.split(' ') if token.strip() != '']
				self.ronin_times.append(tokens[0])
				self.ronin_locations.append(tokens[1:3])
		self.ronin_times = np.asarray(self.ronin_times)
		self.ronin_locations = np.asarray(self.ronin_locations)
		#print(self.ronin_times.shape, self.ronin_locations.shape)

		# gyro
		if read_gyro:
			self.gyro_times_full, self.gyro_rates_full = [], []
			with open(self.data_path+"gyro.txt") as f:
				for i, line in enumerate(f):
					if i==0:
						continue
					line = line.strip()
					tokens = [float(token) for token in line.split(' ') if token.strip() != '']
					self.gyro_times_full.append(tokens[0])
					self.gyro_rates_full.append(tokens[1:4])
			self.gyro_times_full = np.asarray(self.gyro_times_full)*1e-9
			self.gyro_rates_full = np.asarray(self.gyro_rates_full)
			#print(self.gyro_times_full.shape, self.gyro_rates_full.shape)

		# wifi
		if read_wifi:
			self.read_wifi_txt(bssid_database)

		# print(f'ronin interval = {np.mean(np.absolute(self.ronin_times[1:]-self.ronin_times[0:-1]))}')
		# print(f'gyro interval = {np.mean(np.absolute(self.gyro_times_full[1:]-self.gyro_times_full[0:-1]))}')
		# print(f'wifi interval = {np.mean(np.absolute(self.wifi_times_full[1:]-self.wifi_times_full[0:-1]))}')

	def read_wifi_txt(self, bssid_database):
		num_bssids = len(bssid_database)
		self.wifi_times_full, self.wifi_rssis_full = [], []
		stamp_segments = []
		rssi_segments = np.full(num_bssids, np.nan)
		# wifi (BSSID, RSSI)
		with open(self.data_path+"wifi.txt") as f:
			for i, line in enumerate(f):
				if i==0:
					continue
				line = line.strip()
				tokens = [token for token in line.split() if token.strip() != '']
				if len(tokens)==1:
					if len(stamp_segments)!=0:
						self.wifi_times_full.append(statistics.mean(stamp_segments))
						self.wifi_rssis_full.append(rssi_segments)
						#print(np.sum(np.isnan(rssi_segments)==False))
						stamp_segments = []
						rssi_segments = np.full(num_bssids, np.nan)
				else:
					stamp_segments.append(float(tokens[0])*1e-9)
					if tokens[1] in bssid_database:
						rssi_segments[bssid_database[tokens[1]]] = int(tokens[2])
		self.wifi_times_full = np.asarray(self.wifi_times_full) # wifi_reading x ?
		self.wifi_rssis_full = np.asarray(self.wifi_rssis_full) # wifi_reading x ?
		print(self.wifi_times_full.shape, self.wifi_rssis_full.shape)
		
	def read_flp_txt(self):
		# read all flp results, no connection to ronin
		latlon = []
		with open(self.data_path+"FLP.txt") as f:
			for i, line in enumerate(f):
				if i==0:
					continue
				line = line.strip()
				tokens = [float(token) for token in line.split(' ') if token.strip() != '']
				latlon.append(tokens[1:3])
		latlon = np.asarray(latlon)
		self.flp_locations = np.zeros_like(latlon)
		scale = np.cos(latlon[0,0] * np.pi / 180.0)
		er = 6378137
		self.flp_locations[:,0] = scale * latlon[:,1] * np.pi * er / 180
		self.flp_locations[:,1] = scale * er * np.log( np.tan((90+latlon[:,0]) * np.pi / 360) )

	def visualize_flp(self):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(self.flp_locations[:,0], self.flp_locations[:,1], color=(1,0,0), s=2)
		ax.axis('equal')
		plt.tight_layout()
		save_name = "./experiments/singlesearch/"+self.day_id+"-flp.png"
		plt.savefig(save_name)
		plt.close(fig)

	def sample_wifi(self):
		# note that not all ronin position get a valid wifi data
		offset = 0
		self.wifi_corres_id = np.full(self.ronin_times.shape, -1)
		for w, wt in enumerate(self.wifi_times_full):
			offset = np.argmin(np.absolute(self.ronin_times[offset:]-wt))+offset
			self.wifi_corres_id[offset] = w
		self.wifi_has_flag = self.wifi_corres_id>=0


	def sample_gyro(self, read_flag=True):
		# smoothing
		# self.gyro_rates_full = gaussian_filter1d(self.gyro_rates_full, 1., axis=0)
		# self.gyro_rates_full = maximum_filter1d(self.gyro_rates_full, 400, axis=0)
		if read_flag:
			closest_gyro_ids = np.load(self.data_path+"closest_gyro_ids.npy")
		else:
			closest_gyro_ids = np.zeros_like(self.ronin_times, dtype=np.int)
		time_diff = np.zeros_like(self.ronin_times) 
		self.gyro_rates = np.zeros((self.ronin_locations.shape[0],3))
		offset = 0
		gyro_full_len = self.gyro_times_full.shape[0]
		half_win = 1
		for r, rt in enumerate(self.ronin_times):
			if read_flag:
				offset = closest_gyro_ids[r]
			else:
				offset = np.argmin(np.absolute(self.gyro_times_full[offset:]-rt))+offset
				closest_gyro_ids[r] = offset
			time_diff[r] = abs(self.gyro_times_full[offset]-rt)
			self.gyro_rates[r,:] = self.gyro_rates_full[offset,:]
			continue
			if offset-half_win<0:
				local_range = np.arange(0, half_win*2+1)
			elif offset+half_win>=gyro_full_len:
				local_range = np.arange(gyro_full_len-half_win*2-1, gyro_full_len)
			else:
				local_range = np.arange(offset-half_win, offset+half_win+1)
			local_gyro = self.gyro_rates_full[local_range,:]
			local_gyro_time = self.gyro_times_full[local_range]
			# polynomial fitting
			for i in range(3):
				fit_func = np.poly1d(np.polyfit(local_gyro_time-local_gyro_time[0], local_gyro[:,i], 2))
				self.gyro_rates[r,i] = fit_func(rt-local_gyro_time[0])
			# spline fitting
			# for i in range(3):
			#	 fit_func = interpolate.splrep(local_gyro_time-local_gyro_time[0], local_gyro[:,i], s=0)
			#	 self.gyro_rates[r,i] = interpolate.splev(rt-local_gyro_time[0], fit_func, der=0)
		print(np.mean(time_diff), np.median(time_diff))
		if read_flag==False:
			print(self.data_path)
			np.save(self.data_path+"closest_gyro_ids.npy", closest_gyro_ids)

		self.gyro_mag = np.linalg.norm(self.gyro_rates, axis=-1)
		print(np.median(self.gyro_mag))
		#self.gyro_mag = maximum_filter1d(self.gyro_mag, 10)
		self.moving_flag = self.gyro_mag>0.1

	def visualize_gyro(self, full=False):
		if full:
			gyrox = self.gyro_times_full
			gyroy = self.gyro_rates_full
		else:
			gyrox = self.ronin_times
			gyroy = self.gyro_rates
		fig, axs = plt.subplots(3)
		axs[0].plot(gyrox, gyroy[:,0])
		axs[0].set_title('x direction gyro')
		axs[1].plot(gyrox, gyroy[:,1])
		axs[1].set_title('y direction gyro')
		axs[2].plot(gyrox, gyroy[:,2])
		axs[2].set_title('z direction gyro')
		plt.tight_layout()
		fig.savefig(f'./experiments/{self.data_id}_gyro.png')
		plt.close('all')

		fig, axs = plt.subplots(2)
		axs[0].plot(gyrox, np.linalg.norm(gyroy, axis=-1))
		axs[0].set_title('L2 norm gyro')
		moving = np.ma.masked_where(np.logical_not(self.moving_flag), self.gyro_mag)
		static = np.ma.masked_where(self.moving_flag, self.gyro_mag)
		axs[1].plot(gyrox, static, color=(1,0,0))
		axs[1].plot(gyrox, moving, color=(0,1,0))
		axs[1].set_title('L2 norm gyro filtered (green moving, red static)')
		plt.tight_layout()
		fig.savefig(f'./experiments/{self.data_id}_gyro_mag.png')
		plt.close('all')
		

	def visualize_locations(self):
		fig, axs = plt.subplots(2)
		axtmp = axs[0].scatter(self.ronin_locations[:,0], self.ronin_locations[:,1], c=self.gyro_mag, edgecolor='none', cmap='jet', s=2)
		# plt.colorbar(axtmp,ax=axs[0])
		#axs[0].colorbar()
		axs[0].axis('equal')
		axs[0].set_title('ronin traj (colored by L2 norm gyro)')
		tmp_color = np.array([1,0,0,0,1,0]).reshape(2,-1)
		#axs[1].scatter(self.ronin_locations[:,0], self.ronin_locations[:,1], c=tmp_color[self.moving_flag.astype(np.int),:], s=2)
		axs[1].scatter(self.ronin_locations[self.moving_flag,0], self.ronin_locations[self.moving_flag,1], color=(0,1,0), s=2)
		axs[1].scatter(self.ronin_locations[np.logical_not(self.moving_flag),0], self.ronin_locations[np.logical_not(self.moving_flag),1], color=(1,0,0), s=2)
		axs[1].axis('equal')
		axs[1].set_title('ronin traj (green moving, red static)')
		plt.tight_layout()
		fig.savefig(f'./experiments/{self.data_id}_ronin.png')
		plt.close('all')

		for idx in self.moving_indices:
			plt.scatter(self.ronin_locations[idx,0], self.ronin_locations[idx,1], s=2)
		plt.axis('equal')
		plt.tight_layout()
		plt.savefig(f'./experiments/{self.data_id}_ronin_split.png')
		plt.close('all')

	def visualize_cluster(self, ronin_xy, ronin_clusters, cluster_colors, postfix="cluster"):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(ronin_xy[:,0], ronin_xy[:,1], color=(0.5,0.5,0.5), s=2)
		ax.scatter(ronin_clusters[:,0], ronin_clusters[:,1], color=cluster_colors, s=100, marker='d')
		ax.axis('equal')
		plt.tight_layout()
		plt.savefig(f'./experiments/{self.data_id}_{postfix}.png')
		plt.close(fig)
		return ronin_xy, ronin_clusters, cluster_colors
			

	def visualize_dynamic(self):
		valid_wifi_flag = self.wifi_corres_id>=0
		gif_duration = 1.5
		temp_gifs = []
		for j, idx in enumerate(self.moving_indices):
			# plot base
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.scatter(self.ronin_locations[:,0], self.ronin_locations[:,1], color=(0.5,0.5,0.5), s=2)
			ax.scatter(self.ronin_locations[idx,0], self.ronin_locations[idx,1], color=(0,1,0), s=4)
			ax.scatter(self.ronin_locations[valid_wifi_flag,0], self.ronin_locations[valid_wifi_flag,1], color=(0,0,1), s=4)
			ax.axis('equal')
			plt.tight_layout()
			# fig.savefig(f'./experiments/{self.data_id}_{j}_ronin_split.png')
			img = fig_to_numpy(fig)
			temp_gifs.append(img)
			# cv2.imwrite(f'./experiments/{self.data_id}_{j}_ronin_split.png', img)
			plt.close('all')
		imageio.mimsave(f'./experiments/{self.data_id}_ronin_split.gif', temp_gifs, duration=gif_duration)

		temp_gifs = []
		for j, idx in enumerate(self.moving_indices):
			# plot base
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.plot(self.ronin_times, self.gyro_mag, color=(0.5,0.5,0.5), linewidth=2)
			ax.plot(self.ronin_times[idx], self.gyro_mag[idx], color=(0,1,0), linewidth=4)
			plt.tight_layout()
			img = fig_to_numpy(fig)
			temp_gifs.append(img)
			plt.close('all')
		imageio.mimsave(f'./experiments/{self.data_id}_gyro_split.gif', temp_gifs, duration=gif_duration)

	def load_align_result(self, name):
		mag, theta, aligned_ronin = [], [], []
		file_name = self.output_path+name
		joint = name == "c_multialign.txt"
		start_xy = [0,0]
		with open(file_name, "r") as f:
			for i, line in enumerate(f):
				if i==0:
					continue
				line = line.strip()
				tokens = [float(token) for token in line.split('\t') if token.strip() != '']
				if len(tokens)>2:
					mag.append(tokens[0])
					theta.append(tokens[1]+tokens[2])
					aligned_ronin.append(tokens[3:5])
				else:
					start_xy = tokens[0:2]
		self.aligned_ronin = np.asarray(aligned_ronin)
		self.start_xy = np.asarray(start_xy)
		self.aligned_ronin = np.concatenate([self.start_xy.reshape(-1,2), self.aligned_ronin])
		return np.asarray(mag), np.asarray(theta)


		# mag = np.asarray(mag)
		# theta = np.asarray(theta)
		# tmp = np.stack([mag*np.cos(theta), mag*np.sin(theta)], -1)
		# self.aligned_ronin = np.cumsum(tmp, axis=0)
		# return

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(aligned_ronin[:,0], aligned_ronin[:,1], color=(0.5,0.5,0.5), s=2)
		ax.axis('equal')
		plt.tight_layout()
		plt.savefig(f'./experiments/align_result.png')
		plt.close('all')


	def comp_velocity(self):
		time_diff = self.ronin_times[1:] - self.ronin_times[0:-1]
		location_diff = self.ronin_locations[1:,:] - self.ronin_locations[0:-1,:]
		self.velocity = np.zeros_like(self.ronin_locations)
		self.velocity[1:,:] = location_diff/np.stack([time_diff,time_diff],axis=1)
		self.velocity_mag = np.linalg.norm(self.velocity, axis=-1)

	def split_trajectory(self):
		labelled, numfeats = label_mea(1-self.moving_flag.astype(np.int))
		indices = [np.nonzero(labelled == k)[0] for k in np.unique(labelled)[1:]]
		for idx in indices:
			if idx[-1]-idx[0]+1<=180: # 3 minutes
				self.moving_flag[idx] = True
		labelled, numfeats = label_mea(self.moving_flag.astype(np.int))
		indices = [np.nonzero(labelled == k)[0] for k in np.unique(labelled)[1:]]
		for idx in indices:
			if idx[-1]-idx[0]+1<=10: # 10 seconds
				self.moving_flag[idx] = False

		labelled, numfeats = label_mea(self.moving_flag.astype(np.int))
		self.moving_indices = [np.nonzero(labelled == k)[0] for k in np.unique(labelled)[1:]]
		moving_len = 0
		for idx in self.moving_indices:
			moving_len += (idx[-1]-idx[0]+1)
		print(f"{self.moving_flag.shape[0]} vs. {np.sum(self.moving_flag)} vs. {moving_len}")

		labelled, numfeats = label_mea(1-self.moving_flag.astype(np.int))
		self.static_indices = [np.nonzero(labelled == k)[0] for k in np.unique(labelled)[1:]]

		self.averaging_static_wifi()
		self.extract_optim_traj()

	def extract_optim_traj(self):
		# extract trajectory for optimization
		cluster_labels = np.load(self.data_path+"static_cluster_label.npy")
		u_cluster_labels = np.unique(cluster_labels)
		print(f"label num: {len(u_cluster_labels)}")
		m_id, s_id = 0, 0
		m_len, s_len = len(self.moving_indices), len(self.static_indices)
		#print(m_len, s_len, cluster_labels.shape)

		optim_ronin, label_map, optim_rssi, optim_haswifi = [], [], [], [] # ronin traj for optimization, corresponding clsuter label
		while m_id<m_len or s_id<s_len:
			if self.moving_indices[0][0] == 0:
				if m_id<m_len:
					midx = self.moving_indices[m_id]
					optim_ronin.append(self.ronin_locations[midx,:])
					label_map.append(np.full(optim_ronin[-1].shape[0], -1))
					optim_haswifi.append(self.wifi_has_flag[midx])
					temp_id = midx[self.wifi_has_flag[midx]]
					temp_rssi = self.wifi_rssis_full[self.wifi_corres_id[temp_id],:]
					optim_rssi.append(temp_rssi)
				if s_id<s_len:
					optim_ronin.append(self.static_mean_ronin[s_id:s_id+1,:])
					label_map.append(cluster_labels[s_id:s_id+1])
					optim_rssi.append(self.static_mean_wifi[s_id:s_id+1,:])
					optim_haswifi.append(np.array([True]))
			elif self.static_indices[0][0] == 0:
				if s_id<s_len:
					optim_ronin.append(self.static_mean_ronin[s_id:s_id+1,:])
					label_map.append(cluster_labels[s_id:s_id+1])
					optim_rssi.append(self.static_mean_wifi[s_id:s_id+1,:])
					optim_haswifi.append(np.array([True]))
				if m_id<m_len:
					midx = self.moving_indices[m_id]
					optim_ronin.append(self.ronin_locations[midx,:])
					label_map.append(np.full(optim_ronin[-1].shape[0], -1))
					optim_haswifi.append(self.wifi_has_flag[midx])
					temp_id = midx[self.wifi_has_flag[midx]]
					temp_rssi = self.wifi_rssis_full[self.wifi_corres_id[temp_id],:]
					optim_rssi.append(temp_rssi)
			else:
				raise NotImplementedError
			m_id += 1
			s_id += 1
		optim_ronin = np.concatenate(optim_ronin)
		label_map = np.concatenate(label_map)
		optim_haswifi = np.concatenate(optim_haswifi)
		optim_rssi = np.concatenate(optim_rssi)
		
		cluster_indices = [np.nonzero(label_map==i)[0] for i in u_cluster_labels]
		if self.write_flag:
			np.savez(self.data_path + "c_cluster_info.npz", label_map=label_map, optim_ronin=optim_ronin)
		np.savez(self.output_path + "rssi_reduced.npz", rssi=optim_rssi, haswifi=optim_haswifi)
		# convert to mag-angle model
		v = optim_ronin[1:] - optim_ronin[:-1]
		mag = np.linalg.norm(v, axis=-1)
		theta = np.arctan2(v[:,1], v[:,0])

		# save mag, theta, cluster_indices
		if self.write_flag:
			np.savetxt(self.output_path + "c_mag_theta.txt", np.stack((mag, theta),-1), delimiter='\t', header=str(mag.shape[0]))
			print(cluster_indices)
			with open(self.output_path + "c_cluster_indices.txt", "w") as f:
				f.write(str(len(cluster_indices))+"\n")
				for j, ci in enumerate(cluster_indices):
					f.write(str(u_cluster_labels[j])+"\t"+str(ci.shape[0])+"\t")
					for i in ci:
						f.write(str(i)+"\t")
					f.write("\n")
		return

		# complex_v = v[:,0]+1j*v[:,1]
		# mag1, theta1 = np.abs(complex_v), np.angle(complex_v)
		# print(np.mean(np.absolute(mag-mag1)), np.mean(np.absolute(theta-theta1)))
		tmp = np.stack([mag*np.cos(theta), mag*np.sin(theta)], -1)
		tmp = np.cumsum(tmp, axis=0)
		recover_ronin = np.zeros_like(optim_ronin)
		recover_ronin[1:,:] = tmp
		recover_ronin = recover_ronin + optim_ronin[0:1,:]

		print(np.mean(np.absolute(recover_ronin-optim_ronin)))

				

	def averaging_static_wifi(self):
		static_mean_ronin, static_mean_wifi = [], []
		for idx in self.static_indices:
			temp_id = idx[self.wifi_has_flag[idx]]
			temp_rssi = self.wifi_rssis_full[self.wifi_corres_id[temp_id],:]
			temp_mean_rssi = np.nanmean(temp_rssi, axis=0)
			static_mean_wifi.append(temp_mean_rssi)
			static_mean_ronin.append(np.mean(self.ronin_locations[idx,:],axis=0))
		self.static_mean_wifi = np.asarray(static_mean_wifi)
		self.static_mean_ronin = np.asarray(static_mean_ronin)
		#print(static_mean_wifi.shape, len(self.static_indices), static_mean_ronin.shape)
		if self.write_flag:
			np.save(self.data_path+"static_mean_wifi.npy", self.static_mean_wifi)
			np.save(self.data_path+"static_mean_ronin.npy", self.static_mean_ronin)


		

def fig_to_numpy(fig):
	io_buf = io.BytesIO()
	fig.savefig(io_buf, format='raw')
	io_buf.seek(0)
	img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
						 newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
	io_buf.close()
	return img_arr

