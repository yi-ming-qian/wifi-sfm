import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

base_path = "/local-scratch/stationary/wifi/data/yasu_data/"#pyojin_Asus_Tango_SFU_Multiple_Buildings_withVIO/"
folders = glob.glob(base_path+"*_WiFi_SfM/")

bssids = {}
for p in folders:
	print(p)
	with open(p+"wifi.txt") as f:
		for i, line in enumerate(f):
			if i==0:
				continue
			line = line.strip()
			tokens = [token for token in line.split() if token.strip() != '']
			if len(tokens)==1:
				continue
			if tokens[1] not in bssids:
				bssids[tokens[1]] = 0.
			else:
				bssids[tokens[1]] += 1.
numbers = np.sort(np.fromiter(bssids.values(), dtype=float))
print(type(numbers), numbers.shape)
x = np.arange(len(numbers))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x, numbers)
plt.show()

bssids_database = {}
offset = 0
for k, v in bssids.items():
	if v<2e2:
		continue
	bssids_database[k] = offset
	offset += 1

print(len(bssids_database))

# save to file
with open(base_path+"bssids_database.pkl","wb") as f:
	pickle.dump(bssids_database,f)

# import glob
# import pickle

# base_path = "/local-scratch/yimingq/wifi/data/yasu_data/"#pyojin_Asus_Tango_SFU_Multiple_Buildings_withVIO/"
# folders = glob.glob(base_path+"*_WiFi_SfM/")

# bssids = set()
# for p in folders:
# 	print(p)
# 	with open(p+"wifi.txt") as f:
# 		for i, line in enumerate(f):
# 			if i==0:
# 				continue
# 			line = line.strip()
# 			tokens = [token for token in line.split() if token.strip() != '']
# 			if len(tokens)==1:
# 				continue
# 			bssids.add(tokens[1])

# bssids_database = {x: i for i, x in enumerate(bssids)}
# print(len(bssids_database))

# # save to file
# with open(base_path+"bssids_database.pkl","wb") as f:
# 	pickle.dump(bssids_database,f)