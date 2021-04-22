import os
import glob
fp = open("./run.sh", "w")

fp.write("#!/bin/bash\n")

base_path = "../data/yasu_data/"
folders = glob.glob(base_path+"*_WiFi_SfM/")

for i, p in enumerate(folders):
	name = os.path.basename(os.path.normpath(p))
	fp.write(f"./build/singleAlignCorre {name}\n")

fp.close()

# fp = open("./folder_list.txt", "w")

# base_path = "../data/yasu_data/"
# folders = glob.glob(base_path+"*_WiFi_SfM/")

# for i, p in enumerate(folders):
# 	name = os.path.basename(os.path.normpath(p))
# 	fp.write(name+"\n")

# fp.close()