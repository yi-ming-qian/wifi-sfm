import numpy as np
from ronin import Ronin
from reducedronin import ReducedRonin, plot_traj_pair, align_traj_pair
from reducedronin import align_traj_single
import glob
import pickle
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from utils import cluster_colors
import os

write_flag = False

def main():
    base_path = "../data/yasu_data/"#pyojin_Asus_Tango_SFU_Multiple_Buildings_withVIO/"
    output_path = "./outputs/"
    # read bssid database
    folders = glob.glob(base_path+"*_WiFi_SfM/")
    with open(base_path+"bssids_database.pkl", "rb") as f:
        bssid_database = pickle.load(f)
    sample_interval = 200
    for i, p in enumerate(folders):
        print(f"Day {i}---------------")
        # if i>2:
        #     exit()
        print(p)
        folder_name = os.path.basename(p[:-1])
        out_path = output_path+folder_name+"/"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        ronin_data = Ronin(i, p, out_path, sample_interval, write_flag=write_flag)
        ronin_data.readFromText(bssid_database = bssid_database)
        ronin_data.sample_wifi()
        ronin_data.sample_gyro(read_flag=True)
        # ronin_data.comp_velocity()
        ronin_data.split_trajectory()
        #ronin_data.visualize_gyro()
        #ronin_data.visualize_locations()
        ronin_data.visualize_dynamic()
        
def load_singlealign_result():
    base_path = "../data/yasu_data/"
    output_path = "./outputs/"
    folders = glob.glob(base_path+"*_WiFi_SfM/")
    for i, p in enumerate(folders):
        print(i, p)
        data = np.load(p+"c_cluster_info.npz")
        label_map = data["label_map"]
        optim_ronin = data["optim_ronin"]
        folder_name = os.path.basename(p[:-1])
        out_path = output_path+folder_name+"/"

        ronin_data = Ronin(i, p, out_path, 200)
        ronin_data.load_align_result()
        aligned_ronin = np.zeros_like(optim_ronin)
        aligned_ronin[1:,:] = ronin_data.aligned_ronin
        aligned_ronin = aligned_ronin + optim_ronin[0:1,:]
        
        idx = np.nonzero(label_map>=0)[0]

        ronin_data.visualize_cluster(optim_ronin, optim_ronin[idx,:], cluster_colors[label_map[idx],:], postfix="before")
        ronin_data.visualize_cluster(aligned_ronin, aligned_ronin[idx,:], cluster_colors[label_map[idx],:], postfix="after")
        #exit()

def load_multi_corres_align():
    base_path = "../data/yasu_data/"
    output_path = "./outputs/"
    folder_names = []
    with open(output_path+"folder_list_manual.txt") as f:
        for line in f:
            line = line.strip()
            if len(line)>0:
                folder_names.append(line)
    fig = plt.figure()
    ax=fig.add_subplot(111)
    for i, name in enumerate(folder_names):
        p = base_path+name+"/"
        out_path = output_path+name+"/"
        ronin_data = Ronin(i, p, out_path, 200)
        ronin_data.load_align_result("c_multi_corres_align.txt")
        ax.scatter(ronin_data.aligned_ronin[:,0], ronin_data.aligned_ronin[:,1], color=np.random.random(3), s=0.01)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(f'./experiments/multialign/joint.png')
    plt.close('all')

def load_multialign_result():
    base_path = "../data/yasu_data/"
    output_path = "./outputs/"
    folders = glob.glob(base_path+"*_WiFi_SfM/")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, p in enumerate(folders):
        print(i, p)
        data = np.load(p+"c_cluster_info.npz")
        label_map = data["label_map"]
        optim_ronin = data["optim_ronin"]
        folder_name = os.path.basename(p[:-1])
        out_path = output_path+folder_name+"/"

        ronin_data = Ronin(i, p, out_path, 200)
        ronin_data.load_align_result(joint=True)
        aligned_ronin = np.zeros_like(optim_ronin)
        aligned_ronin[1:,:] = ronin_data.aligned_ronin
        aligned_ronin[0,:] = ronin_data.start_xy
        #aligned_ronin = aligned_ronin + optim_ronin[0:1,:]
        
        idx = np.nonzero(label_map>=0)[0]
        #ronin_data.visualize_cluster(optim_ronin, optim_ronin[idx,:], cluster_colors[label_map[idx],:], postfix="before")
        
        
        ronin_xy, ronin_clusters, c_colors = ronin_data.visualize_cluster(aligned_ronin, aligned_ronin[idx,:], cluster_colors[label_map[idx],:], postfix="multi")
        
        ax.scatter(ronin_xy[:,0], ronin_xy[:,1], color=np.random.random(3), s=0.01)
        ax.scatter(ronin_clusters[:,0], ronin_clusters[:,1], color=c_colors, s=100, marker='d')
        #plt.show()
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(f'./experiments/joint.png')
    plt.close('all')

def cluster_wifi():        
    base_path = "/local-scratch/yimingq/wifi/data/yasu_data/"
    folders = glob.glob(base_path+"*_WiFi_SfM/")
    mean_ronins, mean_wifis = [], []
    day_idx = []
    for i, p in enumerate(folders):
        tmp_ronin = np.load(p+"static_mean_ronin.npy")
        tmp_wifi = np.load(p+"static_mean_wifi.npy")
        mean_ronins.append(tmp_ronin)
        mean_wifis.append(tmp_wifi)
        day_idx.append(np.full(tmp_ronin.shape[0],i))
        #print(tmp_wifi.shape, tmp_ronin.shape)
    mean_ronins = np.concatenate(mean_ronins)
    mean_wifis = np.concatenate(mean_wifis)
    day_idx = np.concatenate(day_idx)
    # print(mean_ronins.shape, mean_wifis.shape, day_idx.shape, day_idx)
    # print(np.amax(mean_wifis[np.isnan(mean_wifis)==False]))
    # exit()

    mean_wifis[np.isnan(mean_wifis)] = -100
    bandwidth = estimate_bandwidth(mean_wifis, quantile=0.5)
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(mean_wifis)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters)

    print(labels_unique)
    plot_colors = cluster_colors[labels,:]
    plt.scatter(mean_ronins[:,0], mean_ronins[:,1], s=5, color=plot_colors)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    sample_interval = 200
    for i, p in enumerate(folders):
        print(i)
        ronin_data = Ronin(i, p, sample_interval)
        ronin_data.readFromText(read_gyro=False, read_wifi=False)
        mask= day_idx==i
        np.save(ronin_data.data_path+"static_cluster_label.npy", labels[mask])
        
        start_point = ronin_data.ronin_locations[0:1,:]
        ronin_data.visualize_cluster(ronin_data.ronin_locations, mean_ronins[mask,:], plot_colors[mask,:])
        # ronin_data.load_c_result()
        # ronin_data.visualize_cluster(ronin_data.aligned_ronin, mean_ronins[mask,:]-start_point, plot_colors[mask,:], postfix="align")

def align_by_search():
    base_path = "../data/yasu_data/"#pyojin_Asus_Tango_SFU_Multiple_Buildings_withVIO/"
    output_path = "./outputs/"
    folders = glob.glob(base_path+"*_WiFi_SfM/")
    all_ronins = []
    for i, p in enumerate(folders):
        print(f"Day {i}---------------")
        print(p)
        folder_name = os.path.basename(p[:-1])
        # if folder_name != "20191219100730R_WiFi_SfM":
        #     continue
        out_path = output_path+folder_name+"/"
        rro = ReducedRonin(i, p, out_path, 200)
        rro.read_reduced_ronin()
        rro.visualize_result(ransac=False)
        #rro.visualize_result(ransac=True)
        #rro.visualize_ransac()
        all_ronins.append(rro)
    exit()
    
    for i, r1 in enumerate(all_ronins):
        print(i)
        align_traj_single(all_ronins[i], f"{i}", r1.output_path)
            
def matching_pair():
    base_path = "../data/yasu_data/"#pyojin_Asus_Tango_SFU_Multiple_Buildings_withVIO/"
    output_path = "./outputs/"
    folders = glob.glob(base_path+"*_WiFi_SfM/")
    all_ronins = []
    for i, p in enumerate(folders):
        print(f"Day {i}---------------")
        folder_name = os.path.basename(p[:-1])
        out_path = output_path+folder_name+"/"
        rro = ReducedRonin(i, p, out_path, 200)
        rro.read_reduced_ronin()
        all_ronins.append(rro)
    for i, r1 in enumerate(all_ronins):
        for j, r2 in enumerate(all_ronins):
            if i>=j:
                continue
            align_traj_pair(r1, r2, 0)
def process_flp():
    base_path = "../data/yasu_data/"#pyojin_Asus_Tango_SFU_Multiple_Buildings_withVIO/"
    output_path = "./outputs/"
    folders = glob.glob(base_path+"*_WiFi_SfM/")
    all_ronins = []
    for i, p in enumerate(folders):
        print(f"Day {i}---------------")
        print(p)
        folder_name = os.path.basename(p[:-1])
        out_path = output_path+folder_name+"/"
        ronin_data = Ronin(i, p, out_path, 200, write_flag=False)
        ronin_data.read_flp_txt()
        ronin_data.visualize_flp()        


if __name__ == "__main__":
    #main()
    #cluster_wifi()
    #load_singlealign_result()
    #load_multialign_result()
    #align_by_search()
    #process_flp()
    #matching_pair()
    load_multi_corres_align()
