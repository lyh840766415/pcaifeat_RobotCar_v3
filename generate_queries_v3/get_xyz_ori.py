#coding=utf-8
#get xyz ori

#list all dieectories in the root directory

#find the specific file directory

#list all the files in the specific directories

#get the required data from each file and save to a vector

#examine whether this trajectories is valid

#make directories for the valid sequence

#save the trejectory to file

import os
import utm
import math

BASE_PATH = "/home/luyuheng/lab/dataset/KITTI_RAW"

root_dirs = os.listdir(BASE_PATH)

errors = [];
avg_error_cnt = 0;
error_cnt = 0;

for dirs in root_dirs:
	if os.path.isdir(os.path.join(BASE_PATH,dirs)):
		#print(os.path.join(BASE_PATH,dirs))
		cur1_dirs = os.listdir(os.path.join(BASE_PATH,dirs))
		for dirs1 in cur1_dirs:
			if os.path.isdir(os.path.join(BASE_PATH,dirs,dirs1)):
				#print(os.path.join(BASE_PATH,dirs,dirs1))
				cur2_dirs = os.listdir(os.path.join(BASE_PATH,dirs,dirs1))
				for dirs2 in cur2_dirs:
					if os.path.isdir(os.path.join(BASE_PATH,dirs,dirs1,dirs2)):
						#print(os.path.join(BASE_PATH,dirs,dirs1,dirs2))
						if os.path.isdir(os.path.join(BASE_PATH,dirs,dirs1,dirs2,"oxts/data")):
							#如果序列长度小于500
							if len(os.listdir(os.path.join(BASE_PATH,dirs,dirs1,dirs2,"oxts/data")))<300:
								continue
							#新建存储文件夹
							#os.mkdir(dirs)
							print(os.path.join(BASE_PATH,dirs,dirs1,dirs2))
							print(len(os.listdir(os.path.join(BASE_PATH,dirs,dirs1,dirs2,"oxts/data"))))
							easts = []
							norths = []
							alts = []
							yaws = []
							print(easts)
							print(norths)
							print(alts)
							print(yaws)
							for i in range(len(os.listdir(os.path.join(BASE_PATH,dirs,dirs1,dirs2,"oxts/data")))):
								
								with open(os.path.join(BASE_PATH,dirs,dirs1,dirs2,"oxts/data","%010d.txt"%(i))) as file:
									line = file.readline()
									data = line.split()

									#print("lat = %.12f, long = %.12f ,alt = %.12f, yaw = %.12f "%(float(data[0]),float(data[1]),float(data[2]),float(data[5])))
									#print(type(float(data[0])))
									east,north,_,_ = utm.from_latlon(float(data[0]),float(data[1]))
									alt = float(data[2])
									yaw = float(data[5])
									#print("lat = %.12f, long = %.12f ,alt = %.12f, yaw = %.12f "%(east,north,alt,yaw))  
									easts.append(east)
									norths.append(north)
									alts.append(alt)
									yaws.append(yaw) 
							
							'''	
							print(easts)
							print(norths)
							print(alts)
							print(yaws)
							'''
							max_east = max(easts)
							min_east = min(easts)
							max_north = max(norths)
							min_north = min(norths)
							
							east_dif = max_east-min_east
							north_dif = max_north-min_north
							tot_dis = math.sqrt(east_dif*east_dif+north_dif*north_dif)
							if tot_dis < 50:
								errors.append(len(os.listdir(os.path.join(BASE_PATH,dirs,dirs1,dirs2,"oxts/data"))))
								avg_error_cnt = avg_error_cnt+len(os.listdir(os.path.join(BASE_PATH,dirs,dirs1,dirs2,"oxts/data")))
								error_cnt = error_cnt+1;
								print("error_seqence, avg_error_cnt = %f"%(avg_error_cnt/error_cnt))
								continue
							#新建存储文件夹
							os.mkdir(dirs)
							with open("%s/%s.csv"%(dirs,dirs),"w+") as file:
								for i in range(len(easts)):
									file.write("%.3f,%.3f,%.3f,%.6f\n"%(easts[i],norths[i],alts[i],yaws[i]))
							print("tot_dis = %f"%(tot_dis))
							#exit()
							
							
                #print(line)                             
                
print("error_seqence, avg_error_cnt = %f , max_error_cnt = %d,len_error_cnt = %d"%(avg_error_cnt/error_cnt,max(errors),error_cnt))

				