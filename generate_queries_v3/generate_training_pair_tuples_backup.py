#load position gps/ins data

#load pointcloud pos

#load image time and give it time

#for seq in all seqs

#for pcl in pointclouds
#determine train/test
#search positive pointcloud&img
#search negative pointcloud&img

#save to file

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import os
import pickle
import shutil


def load_pc_file(filename):
	#returns Nx3 matrix
	pc=np.fromfile(filename, dtype=np.float64)

	if(pc.shape[0]!= 4096*3):
		print("Error in pointcloud shape")
		return np.array([])

	pc=np.reshape(pc,(pc.shape[0]//3,3))
	return pc
	
def main():
	#load position gps/ins data
	POS_FILE_PATH = "/home/lyh/lab/benchmark_datasets/oxford_img/2014-05-19-13-20-57/2014-05-19-13-20-57/gps/ins.csv"
	pos_data = pd.read_csv(POS_FILE_PATH)
	print(pos_data.shape)
	col_n = ['timestamp','northing','easting']
	pos_data = pd.DataFrame(pos_data,columns = col_n)
	print(pos_data)
	pos_data = np.array(pos_data)
	print(pos_data[:,0:1])
	time_tree = KDTree(pos_data[:,0:1])
		
		
	#load image time and give it time
	IMG_TIME_FILE_PATH = "/home/lyh/lab/benchmark_datasets/oxford_img/2014-05-19-13-20-57/2014-05-19-13-20-57/stereo.timestamps"
	image_time = np.loadtxt(IMG_TIME_FILE_PATH)

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
	print(nearest_time_index)
	print(nearest_time_index.shape)
	image_pos = pos_data[nearest_time_index,:]
	print(image_pos.shape)
	image_pos_tree = KDTree(image_pos[:,1:3])
	image_time_tree = KDTree(image_pos[:,0:1])
		


	#load pointcloud time and pos
	PC_TIME_POS_PATH = "/home/lyh/lab/benchmark_datasets/oxford/2014-05-19-13-20-57/pointcloud_locations_20m_10overlap.csv"
	pc_time_pos = pd.read_csv(PC_TIME_POS_PATH)
	pc_time_pos = np.array(pc_time_pos)
	print(pc_time_pos)
	print(pc_time_pos.shape)
	
	#match between pointcloud and image according to pos&timestamp
	positive_img_pos_tmp = image_pos_tree.query_radius(pc_time_pos[:,1:3],r=10)
	_,positive_img_ind = image_time_tree.query(pc_time_pos[:,0:1],k=1)
	positive_img_pos_tmp = np.array(positive_img_pos_tmp)
	positive_img_ind = np.array(positive_img_ind)
	print(positive_img_pos_tmp)
	print(positive_img_ind)
	
	#max_len for near neighbor
	max_len = 0
	for i in range(positive_img_pos_tmp.shape[0]):
		if max_len<positive_img_pos_tmp[i].shape[0]:
			max_len = positive_img_pos_tmp[i].shape[0]
	positive_img_pos = np.zeros([positive_img_pos_tmp.shape[0],max_len],dtype = np.float64)
	print(positive_img_pos.shape)
	
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
				
	#print(positive_img_pos.shape)
	#print(positive_img_ind.shape)
				
	positive_img = (np.concatenate((positive_img_pos,positive_img_ind),axis = 1)).tolist()
	
	#ignore the duplicate and invalid item
	for i in range(len(positive_img)):
		positive_img[i] = np.unique(positive_img[i])
		positive_img[i] = [x for x in positive_img[i] if x >=0]
		print(positive_img[i])
		
	#print(positive_img[0][0])
	#print(len(positive_img[0]))
	
	
	pc_img_match = {}
	for i in range(len(positive_img)):
		os.mkdir(os.path.join("./showcase","%d"%(i)))
		#print("point cloud path = ",pc_time_pos[i,0])
		if os.path.exists(os.path.join("/home/lyh/lab/benchmark_datasets/oxford/2014-05-19-13-20-57/pointcloud_20m_10overlap/","%d.bin"%(pc_time_pos[i,0]))):
			#print("exist",os.path.join("/home/lyh/lab/benchmark_datasets/oxford/2014-05-19-13-20-57/pointcloud_20m_10overlap/","%d.bin"%(pc_time_pos[i,0])))
			pc = os.path.join("/home/lyh/lab/benchmark_datasets/oxford/2014-05-19-13-20-57/pointcloud_20m_10overlap/","%d.bin"%(pc_time_pos[i,0]))
			pcl = load_pc_file(pc)
			np.savetxt(os.path.join("./showcase","%d"%(i),"%d.txt"%(pc_time_pos[i,0])),pcl,fmt = "%.3f")
		img = []
		img_num = len(positive_img[i])
		for j in range(len(positive_img[i])):
			#print("i = ",i," j = ",j," ind = ",int(positive_img[i][j])," len = ",len(positive_img[i]))
			#print("	positive image path",image_pos[int(positive_img[i][j]),0])
			if os.path.exists(os.path.join("/home/lyh/lab/benchmark_datasets/oxford_img/2014-05-19-13-20-57/2014-05-19-13-20-57/stereo/centre/","%d.png"%(image_time[int(positive_img[i][j]),0]))):
				#print("	exist",os.path.join("/home/lyh/lab/benchmark_datasets/oxford_img/2014-05-19-13-20-57/2014-05-19-13-20-57/stereo/centre/","%d.png"%(image_time[int(positive_img[i][j]),0])))
				img.append(os.path.join("/home/lyh/lab/benchmark_datasets/oxford_img/2014-05-19-13-20-57/2014-05-19-13-20-57/stereo/centre/","%d.png"%(image_time[int(positive_img[i][j]),0])))
				shutil.copy(os.path.join("/home/lyh/lab/benchmark_datasets/oxford_img/2014-05-19-13-20-57/2014-05-19-13-20-57/stereo/centre/","%d.png"%(image_time[int(positive_img[i][j]),0])),os.path.join("./showcase","%d"%(i)))
				#print("	img pos time diff",abs(image_time[int(positive_img[i][j]),0]-image_pos[int(positive_img[i][j]),0])/1000000)
				#print("	pc img time diff",(image_time[int(positive_img[i][j]),0]-pc_time_pos[i,0])/1000000)
		pc_img_match[i] = {"pc":pc,"img":img,"img_num":img_num}
	
	print(pc_img_match)
	with open("pc_img_match.pickle", 'wb') as handle:
		pickle.dump(pc_img_match, handle, protocol=pickle.HIGHEST_PROTOCOL)
				
	

if __name__ == '__main__':
	main()
