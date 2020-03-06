#this script is work in RobotCar Dataset
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import os
import pickle
import shutil

PC_PATH = "/data/lyh/benchmark_datasets/oxford/"
IMG_PATH = "/data/lyh/RobotCar/mono_left_color/"
TRAJ_PATH = "/data/lyh/RobotCar/gps_ins/"


def load_sequence():
	#For Oxford
	seq_dirs=np.empty(0,dtype=object)
	runs_folder = "oxford/"
	all_folders=sorted(os.listdir(PC_PATH))
	index_list=[5,6,7,9,10,11,12,13,14,15,16,17,18,19,22,24,31,32,33,38,39,43,44]
	print(len(index_list))
	for index in index_list:
		seq_dirs = np.append(seq_dirs,all_folders[index])
		
		'''
		#verify test sequence
		if os.path.exists(os.path.join(PC_PATH,all_folders[index],"pointcloud_locations_20m.csv")):
			print("exist",os.path.join(PC_PATH,all_folders[index],"pointcloud_locations_20m.csv"))
		else:
			print("Error",os.path.join(PC_PATH,all_folders[index],"pointcloud_locations_20m.csv"))
		'''
		
	return seq_dirs

	
def main():
	seq_dirs = sorted(load_sequence())
	seq_pc_dict = {}
	for i,cur_dir in enumerate(seq_dirs):
		#if i>2:
		#	break;
		print("\n\n",cur_dir)
		#load position gps/ins data for cur_dir
		POS_FILE_PATH = os.path.join(TRAJ_PATH,cur_dir,"gps/ins.csv")
		pos_data = pd.read_csv(POS_FILE_PATH)
		print(pos_data.shape)
		col_n = ['timestamp','northing','easting']
		pos_data = pd.DataFrame(pos_data,columns = col_n)
		pos_data = np.array(pos_data)
		time_tree = KDTree(pos_data[:,0:1])
		
		#load image time and give its time
		IMG_TIME_FILE_PATH = os.path.join(IMG_PATH,cur_dir,"mono_left.timestamps")
		image_time = pd.read_csv(IMG_TIME_FILE_PATH,sep=" ", header=None)
		image_time = image_time.dropna(axis=0,how='any')
		image_time = np.array(image_time)
		
		#remove image that is out of ins time range
		start_index = 0
		end_index = image_time.shape[0]
		for i in range(image_time.shape[0]):
			early_end = True
			if image_time[i,0]<pos_data[0,0]:
				start_index = i+1
				early_end = False
			if image_time[image_time.shape[0]-i-1,0] > pos_data[-1,0]:
				end_index = image_time.shape[0]-i-1
				early_end = False
			if early_end:
				break
		
		print(start_index,end_index)
		image_time = image_time[start_index:end_index,:]
			
		#give time & pos to image saved in image pos
		nearest_time_dis,nearest_time_index = time_tree.query(image_time[:,0:1],k=1)
		nearest_time_index = np.array(nearest_time_index).flatten()
		image_pos = pos_data[nearest_time_index,:]
		image_pos_tree = KDTree(image_pos[:,1:3])
		image_time_tree = KDTree(image_pos[:,0:1])
		
		#load pointcloud time and pos
		PC_TIME_POS_PATH = os.path.join(PC_PATH,cur_dir,"pointcloud_locations_20m.csv")
		pc_time_pos = pd.read_csv(PC_TIME_POS_PATH,header=0)
		pc_time_pos = np.array(pc_time_pos).astype(np.double)
		
		#match between pointcloud and image according to pos&timestamp
		positive_img_pos_tmp = image_pos_tree.query_radius(pc_time_pos[:,1:3],r=10)
		time_gap,positive_img_ind = image_time_tree.query(pc_time_pos[:,0:1],k=1)
		positive_img_pos_tmp = np.array(positive_img_pos_tmp)
		positive_img_ind = np.array(positive_img_ind)
		
		for i in range(time_gap.shape[0]):
			if time_gap[i] > 2e6 :
				positive_img_ind[i] = -1
		
		#max_len for near neighbor
		max_len = 0
		for i in range(positive_img_pos_tmp.shape[0]):
			if max_len<positive_img_pos_tmp[i].shape[0]:
				max_len = positive_img_pos_tmp[i].shape[0]
		positive_img_pos = np.zeros([positive_img_pos_tmp.shape[0],max_len],dtype = np.float)
		
		for i in range(positive_img_pos_tmp.shape[0]):
			for j in range(max_len):
				if j<positive_img_pos_tmp[i].shape[0]:
					positive_img_pos[i,j] = positive_img_pos_tmp[i][j]
				else:
					positive_img_pos[i,j] = -1
		
		#only keep the neighbor that distance small than 10m & time interval small than 2s
		for i in range(positive_img_pos_tmp.shape[0]):
			for j in range(positive_img_pos_tmp[i].shape[0]):
				if abs(image_pos[int(positive_img_pos[i,j]),0] - pc_time_pos[i,0])>2000000:
					positive_img_pos[i][j] = -1
		
		#combine both time pos and position pos
		positive_img = np.concatenate((positive_img_pos,positive_img_ind),axis = 1)
		positive_img = (positive_img.astype(np.int)).tolist()

		
		#ignore the duplicate and invalid item
		for i in range(len(positive_img)):
			positive_img[i] = np.unique(positive_img[i])
			positive_img[i] = [x for x in positive_img[i] if x >=0]
			
		#construct point cloud image match dict
		cur_pc_img_pair_dict = {}
		
		for i in range(pc_time_pos.shape[0]):
			cur_pc_img_pair_dict[str(pc_time_pos[i,0].astype(np.int))] = image_time[positive_img[i],0].astype(np.int)
		
		#print(cur_pc_img_pair_dict.keys())
		seq_pc_dict[cur_dir] = cur_pc_img_pair_dict
	
	#verify the correctness of the file
	print(seq_pc_dict.keys())
	print("--------------------------split--------------------------------")
	
	#save the dictionary into file
	filename = "mono_left_pointcloud_image_match_test.pickle"
	with open(filename, 'wb') as handle:
		pickle.dump(seq_pc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	#reload the file to varify
	with open(filename, 'rb') as handle:
		verify_dict = pickle.load(handle)
		print(verify_dict.keys())			
	

if __name__ == '__main__':
	main()
