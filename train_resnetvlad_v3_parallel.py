import tensorflow as tf
import numpy as np
from loading_input_v3 import *
from pointnetvlad_v3.pointnetvlad_cls import *
import nets_v3.resnetvlad_v1_50 as resnet

import shutil
from multiprocessing.dummy import Pool as ThreadPool
import threading
import time
import cv2

#thread pool
pool = ThreadPool(40)

# is rand init 
RAND_INIT = False
# model path
MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v2/model_v3/pcai_model/model_00015005.ckpt"
PC_MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v2/model/pc_model/pc_model_00525175.ckpt"
IMG_MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3/log/train_save_v3_resnetvlad/img_model_00156052.ckpt"
# log path
LOG_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3/log/train_save_v3_resnetvlad/"
# 1 for point cloud only, 2 for image only, 3 for pc&img&fc
TRAINING_MODE = 2
#TRAIN_ALL = True
ONLY_TRAIN_FUSION = False



#Loss
quadruplet = True

#image path
#IMAGE_PATH = "/media/lyh/shared_space/lyh/dataset/ROBOTCAR/mono_left_color"
IMAGE_PATH = "/data/lyh/RobotCar/mono_left_color"

SENSOR = "mono_left"


# Epoch & Batch size &FINAL EMBBED SIZE & learning rate
EPOCH = 5
LOAD_BATCH_SIZE = 100
FEAT_BATCH_SIZE = 2
LOAD_FEAT_RATIO = LOAD_BATCH_SIZE//FEAT_BATCH_SIZE
EMBBED_SIZE = 256
BASE_LEARNING_RATE = 3.6e-5

#pos num,neg num,other neg num,all_num
POS_NUM = 2
NEG_NUM = 5
OTH_NUM = 1
BATCH_DATA_SIZE = 1 + POS_NUM + NEG_NUM + OTH_NUM

# Hard example mining start
HARD_MINING_START = 5

# Margin
MARGIN1 = 0.5
MARGIN2 = 0.2

#Train file index & pc img matching
TRAIN_FILE = 'generate_queries_v3/training_queries_RobotCar.pickle'
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
PC_IMG_MATCH_FILE = 'generate_queries_v3/mono_left_pointcloud_image_match.pickle'
PC_IMG_MATCH_DICT = get_pc_img_match_dict(PC_IMG_MATCH_FILE)

#cur_load for get_batch_keys
CUR_LOAD = 0

#multi threading share global variable
TRAINING_DATA = []
TRAINING_DATA_LOCK = threading.Lock()
#for each load batch
BATCH_REACH_END = False
cnt = 0
LOAD_QUENE_SIZE = 20
EP = 0


def get_learning_rate(epoch):
	learning_rate = BASE_LEARNING_RATE*((0.9)**(epoch//5))
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
	return learning_rate

def get_bn_decay(step):
	#batch norm parameter
	DECAY_STEP = 200000
	BN_INIT_DECAY = 0.5
	BN_DECAY_DECAY_RATE = 0.5
	BN_DECAY_DECAY_STEP = float(DECAY_STEP)
	BN_DECAY_CLIP = 0.99
	bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,step*FEAT_BATCH_SIZE,BN_DECAY_DECAY_STEP,BN_DECAY_DECAY_RATE,staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay

def init_imgnetwork():
	with tf.variable_scope("img_var"):
		img_placeholder = tf.placeholder(tf.float32,shape=[FEAT_BATCH_SIZE*BATCH_DATA_SIZE,256,256,3])
		endpoints,body_prefix = resnet.endpoints(img_placeholder,is_training=True)
		img_feat = tf.layers.dense(endpoints,1000,activation=tf.nn.relu)
	return img_placeholder,img_feat
	
def init_pcnetwork(step):
	with tf.variable_scope("pc_var"):
		pc_placeholder = tf.placeholder(tf.float32,shape=[FEAT_BATCH_SIZE*BATCH_DATA_SIZE,4096,3])
		is_training_pl = tf.placeholder(tf.bool, shape=())
		bn_decay = get_bn_decay(step)
		endpoints = pointnetvlad(pc_placeholder,is_training_pl,bn_decay)
		pc_feat = endpoints
	return pc_placeholder,is_training_pl,pc_feat
	
def init_fusion_network(pc_feat,img_feat):
	with tf.variable_scope("fusion_var"):
		img_pc_concat_feat = tf.concat((pc_feat,img_feat),axis=3)
		endpoints,body_prefix = resnet_fusion.endpoints(img_pc_concat_feat,is_training=True)
	
	pcai_feat = endpoints['model_output']
	print(pcai_feat)
				
	return pcai_feat

def init_pcainetwork():
	#training step
	step = tf.Variable(0)
	
	#init sub-network
	img_placeholder, img_feat = init_imgnetwork()

	#prepare data and loss	
	img_feat = tf.reshape(img_feat,[FEAT_BATCH_SIZE,BATCH_DATA_SIZE,img_feat.shape[1]])
	q_img_vec, pos_img_vec, neg_img_vec, oth_img_vec = tf.split(img_feat, [1,POS_NUM,NEG_NUM,OTH_NUM],1)
	img_loss = lazy_quadruplet_loss(q_img_vec, pos_img_vec, neg_img_vec, oth_img_vec, MARGIN1, MARGIN2)
	tf.summary.scalar('img_loss', img_loss)
		
	#learning rate strategy, all in one?
	epoch_num_placeholder = tf.placeholder(tf.float32, shape=())
	learning_rate = get_learning_rate(epoch_num_placeholder)
	tf.summary.scalar('learning_rate', learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	
	#variable update
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)	
	#training operation
	with tf.control_dependencies(update_ops):
		img_train_op = optimizer.minimize(img_loss, global_step=step)
		
	#merged all log variable
	merged = tf.summary.merge_all()
	
	#output of pcainetwork init
	ops = {
		"img_placeholder":img_placeholder,
		"epoch_num_placeholder":epoch_num_placeholder,
		"img_loss":img_loss,
		"img_train_op":img_train_op,
		"merged":merged,
		"step":step}
	return ops

		

def init_network_variable(sess,train_saver):
	sess.run(tf.global_variables_initializer())
	print("random init")
	
	if RAND_INIT:
		return
	
	train_saver['all_saver'].restore(sess,IMG_MODEL_PATH)
	print("img_model restored")
	return
	

def init_train_saver():
	all_saver = tf.train.Saver()
	
	train_saver = {
		'all_saver':all_saver}
	
	return train_saver
	
def prepare_batch_data(pc_data,img_data,feat_batch,ops,ep):
	is_training = True
	if TRAINING_MODE != 2:
		feat_batch_pc = pc_data[feat_batch*BATCH_DATA_SIZE*FEAT_BATCH_SIZE:(feat_batch+1)*BATCH_DATA_SIZE*FEAT_BATCH_SIZE]
	if TRAINING_MODE != 1:
		feat_batch_img = img_data[feat_batch*BATCH_DATA_SIZE*FEAT_BATCH_SIZE:(feat_batch+1)*BATCH_DATA_SIZE*FEAT_BATCH_SIZE]
	

	if TRAINING_MODE == 1:
		train_feed_dict = {
		  ops["is_training_pl"]:is_training,
			ops["pc_placeholder"]:feat_batch_pc,
			ops["epoch_num_placeholder"]:ep}
		return train_feed_dict
		
	if TRAINING_MODE == 2:
		train_feed_dict = {
			ops["img_placeholder"]:feat_batch_img,
			ops["epoch_num_placeholder"]:ep}
		return train_feed_dict
		
	if TRAINING_MODE == 3:
		train_feed_dict = {
			ops["is_training_pl"]:is_training,
			ops["img_placeholder"]:feat_batch_img,
			ops["pc_placeholder"]:feat_batch_pc,
			ops["epoch_num_placeholder"]:ep}
		return train_feed_dict
		
	print("prepare_batch_data_error,no_such train mode.")
	exit()

def train_one_step(sess,ops,train_feed_dict,train_writer):
	summary,step,img_loss,_,= sess.run([ops["merged"],ops["step"],ops["img_loss"],ops["img_train_op"]],feed_dict = train_feed_dict)
	print("batch num = %d , img_loss = %f"%(step, img_loss))

			
	#other training strategy
	train_writer.add_summary(summary, step)
	return step
	
def evaluate():
	return
	
def model_save(sess,step,train_saver):
	save_path = train_saver['all_saver'].save(sess,os.path.join(LOG_PATH, "img_model_%08d.ckpt"%(step)))
	print("resnetvlad Model saved in file: %s" % save_path)
	return

def get_correspond_img(pc_filename):
	timestamp = pc_filename[-20:-4]
	seq_name = pc_filename[-65:-46]
	image_ind = PC_IMG_MATCH_DICT[seq_name][timestamp]
	if len(image_ind) <= 0:
		return None
	
	#print("len(image_ind) = ",len(image_ind))
	image_timestamp = image_ind[random.randint(0,len(image_ind)-1)]
	image_filename = os.path.join(IMAGE_PATH,seq_name,SENSOR,"%s.png"%(image_timestamp))
	#print(image_filename)
	#if os.path.exists(image_filename):
	#	print("exist")
	return image_filename

def is_negative(query,not_negative):
	return not query in not_negative
	
def get_load_batch_filename(load_batch_keys,quadruplet):		
	pc_files = []
	img_files = []
	for key_cnt ,key in enumerate(load_batch_keys):
		pc_files.append(TRAINING_QUERIES[key]["query"])
		img_files.append(get_correspond_img(TRAINING_QUERIES[key]["query"]))
		random.shuffle(TRAINING_QUERIES[key]["positives"])
		
		#print(TRAINING_QUERIES[key])
		cur_pos = 0;
		for i in range(POS_NUM):
			while True:
				filename = get_correspond_img(TRAINING_QUERIES[TRAINING_QUERIES[key]["positives"][cur_pos]]["query"])

				if filename in img_files[9*(key_cnt)+1:9*(key_cnt)+1+i]:
					cur_pos = cur_pos+1
					continue
				if (not filename is None) and os.path.exists(filename):
					break
				cur_pos = cur_pos+1
				if cur_pos>len(TRAINING_QUERIES[key]["positives"]):
					print("line 259, error in positive number")
					exit()
			
			pc_files.append(TRAINING_QUERIES[TRAINING_QUERIES[key]["positives"][cur_pos]]["query"])
			img_files.append(filename)		
		
		neg_indices = []
		for i in range(NEG_NUM):
			while True:
				while True:
					neg_ind = random.randint(0,len(TRAINING_QUERIES.keys())-1)
					if is_negative(neg_ind,TRAINING_QUERIES[key]["not_negative"]):
						break
				

				filename = get_correspond_img(TRAINING_QUERIES[neg_ind]["query"])
				if filename in img_files[9*(key_cnt)+1+POS_NUM:9*(key_cnt)+1+POS_NUM+i]:
					continue
				if (not filename is None) and os.path.exists(filename):
					break
					
			neg_indices.append(neg_ind)
			pc_files.append(TRAINING_QUERIES[neg_ind]["query"])
			img_files.append(filename)
		
		'''
		tmp_list = img_files[9*(key_cnt)+1+POS_NUM:9*(key_cnt)+1+POS_NUM+NEG_NUM]
		if len(tmp_list)!=len(set(tmp_list)):
			print("neg_duplicate")
			input()
		'''
		
		if quadruplet:
			neighbors=[]
			for pos in TRAINING_QUERIES[key]["positives"]:
				neighbors.append(pos)
			for neg in neg_indices:
				for pos in TRAINING_QUERIES[neg]["positives"]:
					neighbors.append(pos)
					
			#print("neighbors size = ",len(neighbors))
			while True:
				neg_ind = random.randint(0,len(TRAINING_QUERIES.keys())-1)
				if is_negative(neg_ind,neighbors):
					filename = get_correspond_img(TRAINING_QUERIES[neg_ind]["query"])
					if (not filename is None) and os.path.exists(filename):
						break						
									
			pc_files.append(TRAINING_QUERIES[neg_ind]["query"])
			img_files.append(get_correspond_img(TRAINING_QUERIES[neg_ind]["query"]))
	
	if TRAINING_MODE == 1:
		return pc_files,None
	
	if TRAINING_MODE == 2:
		return None,img_files
		
	if TRAINING_MODE == 3:
		return pc_files,img_files
	

def get_batch_keys(train_file_idxs,train_file_num):
	global CUR_LOAD
	load_batch_keys = []
	
	while len(load_batch_keys) < LOAD_BATCH_SIZE:
		skip_num = 0
		#make sure cur_load is valid
		if CUR_LOAD >= train_file_num:
			return True,None
			
		cur_key = train_file_idxs[CUR_LOAD]
		if len(TRAINING_QUERIES[cur_key]["positives"]) < POS_NUM:
			CUR_LOAD = CUR_LOAD + 1
			skip_num = skip_num + 1
			continue
		
		filename = get_correspond_img(TRAINING_QUERIES[cur_key]["query"])
		if filename is None or (not os.path.exists(filename)):
			#print(TRAINING_QUERIES[cur_key]["query"])
			CUR_LOAD = CUR_LOAD + 1
			skip_num = skip_num + 1
			continue
		
		valid_pos = 0
		for i in range(len(TRAINING_QUERIES[cur_key]["positives"])):
			filename = get_correspond_img(TRAINING_QUERIES[TRAINING_QUERIES[cur_key]["positives"][i]]["query"])
			if (not filename is None) and os.path.exists(filename):
					valid_pos = valid_pos + 1
				
		if valid_pos < POS_NUM:
			skip_num = skip_num + 1
			CUR_LOAD = CUR_LOAD + 1
			continue
						
		load_batch_keys.append(train_file_idxs[CUR_LOAD])
		CUR_LOAD = CUR_LOAD + 1
		
	return False,load_batch_keys
	
def load_data(train_file_idxs):
	global BATCH_REACH_END
	global TRAINING_DATA
	global cnt
	
	while True:
		TRAINING_DATA_LOCK.acquire()
		list_len = len(TRAINING_DATA)
		TRAINING_DATA_LOCK.release()
		if list_len > LOAD_QUENE_SIZE:
			print("reach maximum")
			time.sleep(1)
			continue
		
		BATCH_REACH_END,load_batch_keys = get_batch_keys(train_file_idxs,train_file_idxs.shape[0])
		if BATCH_REACH_END:
			print("load thread ended---------------------------------------------------------------------------------------------------")
			break
		
		
		#select load_batch tuple
		
		load_pc_filenames,load_img_filenames = get_load_batch_filename(load_batch_keys,quadruplet)
		'''
		for i in range(LOAD_BATCH_SIZE):
			validate_base = "/data/lyh/lab/pcaifeat_RobotCar_v3/validate"
			if not os.path.exists(os.path.join(validate_base,"%04d"%(i))):
				os.makedirs(os.path.join(validate_base,"%04d"%(i)))
			
			shutil.copy(load_img_filenames[i*9],os.path.join(validate_base,"%04d"%(i)))
			
			if not os.path.exists(os.path.join(validate_base,"%04d"%(i),"positive")):
				os.makedirs(os.path.join(validate_base,"%04d"%(i),"positive"))
			for j in range(POS_NUM):
				shutil.copy(load_img_filenames[i*9+j+1],os.path.join(validate_base,"%04d"%(i),"positive"))			
			
			if not os.path.exists(os.path.join(validate_base,"%04d"%(i),"negative")):
				os.makedirs(os.path.join(validate_base,"%04d"%(i),"negative"))
			for j in range(NEG_NUM):
				shutil.copy(load_img_filenames[i*9+j+1+POS_NUM],os.path.join(validate_base,"%04d"%(i),"negative"))
				
			if not os.path.exists(os.path.join(validate_base,"%04d"%(i),"oth_neg")):
				os.makedirs(os.path.join(validate_base,"%04d"%(i),"oth_neg"))
			for j in range(OTH_NUM):
				shutil.copy(load_img_filenames[i*9+j+1+POS_NUM+NEG_NUM],os.path.join(validate_base,"%04d"%(i),"oth_neg"))
		input()
		'''	
		
		#load pc&img data from file
		pc_data,img_data = load_img_pc_from_net(load_pc_filenames,load_img_filenames,pool)
		#pc_data,img_data = load_img_pc(load_pc_filenames,load_img_filenames,pool)
		TRAINING_DATA_LOCK.acquire()
		TRAINING_DATA.append([pc_data,img_data])
		TRAINING_DATA_LOCK.release()
		
		cnt = cnt + 1
		print("load one batch",cnt)
		
	return

def training(sess,train_saver,train_writer,ops):
	global BATCH_REACH_END
	
	first_loop = True
	consume_all = False
	while True:			
		TRAINING_DATA_LOCK.acquire()
		list_len = len(TRAINING_DATA)
		TRAINING_DATA_LOCK.release()
		#determine whether the first loop
		if not first_loop:
			#determine whether the consume all
			if not consume_all:
				if list_len <= 0:
					consume_all = True
					#end training
					if BATCH_REACH_END:
						print("training thread ended")
						break
					continue
			else:
				if list_len < LOAD_QUENE_SIZE:
					print("list_len = %d, wait for list_len >= %d"%(list_len,LOAD_QUENE_SIZE))
					time.sleep(20)
					continue
				else:
					consume_all = False
					continue					
		else:
			if list_len <= 0:
				time.sleep(1)
				continue
			first_loop = False
		
		TRAINING_DATA_LOCK.acquire()
		cur_batch_data = TRAINING_DATA[0]
		del(TRAINING_DATA[0])
		TRAINING_DATA_LOCK.release()
		pc_data = cur_batch_data[0]
		img_data = cur_batch_data[1]
		
		print("consume one batch")
	
		for feat_batch in range(LOAD_FEAT_RATIO):
			#prepare this batch data
			train_feed_dict = prepare_batch_data(pc_data,img_data,feat_batch,ops,EP)
											
			#training
			step = train_one_step(sess,ops,train_feed_dict,train_writer)
						
			#evaluate TODO
			if step%201 == 0:
				evaluate()
						
			if step%3001 == 0:
				model_save(sess,step,train_saver)
	
	return
	
	
def main():
	global CUR_LOAD
	global BATCH_REACH_END
	global EP
	
	#init network pipeline
	ops = init_pcainetwork()
	
	#init train saver
	train_saver = init_train_saver()

	#init GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	
	
	#init tensorflow Session
	with tf.Session(config=config) as sess:
		#init all the variable
		init_network_variable(sess,train_saver)
		train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
		
		#init_training thread
		training_thread = threading.Thread(target=training, args=(sess,train_saver,train_writer,ops,))
		training_thread.start()

		#start training
		for ep in range(EPOCH):
			train_file_num = len(TRAINING_QUERIES.keys())
			train_file_idxs = np.arange(0,train_file_num)
			np.random.shuffle(train_file_idxs)
			print('Eppch = %d, train_file_num = %f , FEAT_BATCH_SIZE = %f , iteration per batch = %f' %(ep,len(train_file_idxs), FEAT_BATCH_SIZE,len(train_file_idxs)//FEAT_BATCH_SIZE))
			EP = ep
			BATCH_REACH_END = False
			CUR_LOAD = 0
			#load data thread
			load_data_thread = threading.Thread(target=load_data, args=(train_file_idxs,))
			load_data_thread.start()
				
			load_data_thread.join()
		training_thread.join()
						
					
if __name__ == "__main__":
	main()