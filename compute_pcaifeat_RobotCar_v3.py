import numpy as np
from loading_input_v3 import *
from pointnetvlad_v3.pointnetvlad_cls import *
import nets_v3.resnet_v1_50 as resnet
import nets_v3.resnet_v1_fusion as resnet_fusion
import tensorflow as tf
from time import *
import pickle
from multiprocessing.dummy import Pool as ThreadPool

#thread pool
pool = ThreadPool(40)

# 1 for point cloud only, 2 for image only, 3 for pc&img&fc
TRAINING_MODE = 3
BATCH_SIZE = 100
EMBBED_SIZE = 256

DATABASE_FILE= 'generate_queries/RobotCar_oxford_evaluation_database.pickle'
QUERY_FILE= 'generate_queries/RobotCar_oxford_evaluation_query.pickle'
PC_IMG_MATCH_FILE = 'generate_queries/pcai_pointcloud_image_match_test.pickle'
DATABASE_SETS= get_sets_dict(DATABASE_FILE)
QUERY_SETS= get_sets_dict(QUERY_FILE)
PC_IMG_MATCH_DICT = get_pc_img_match_dict(PC_IMG_MATCH_FILE)

#model_path & image path
IMAGE_PATH = '/data/lyh/RobotCar/stereo_centre/'
PC_MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v2/log/train_save_pc_bn/pc_model_00516172.ckpt"
IMG_MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v2/model/img_model/img_model_00291097.ckpt"
MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v2/model_v3/pcai_model/model_00135045.ckpt"

def get_correspond_img(pc_filename):
	timestamp = pc_filename[-20:-4]
	seq_name = pc_filename[-55:-36]
	image_ind = PC_IMG_MATCH_DICT[seq_name][timestamp]
	image_timestamp = image_ind[random.randint(0,len(image_ind)-1)]
	image_filename = os.path.join(IMAGE_PATH,seq_name,"stereo/centre","%s.png"%(image_timestamp))

	return image_filename

def output_to_file(output, filename):
	with open(filename, 'wb') as handle:
		pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Done ", filename)

def save_feat_to_file(database_feat,query_feat):
	if TRAINING_MODE == 1:
		output_to_file(database_feat["pc_feat"],"database_pc_feat_"+PC_MODEL_PATH[-13:-5]+".pickle")
		output_to_file(query_feat["pc_feat"],"query_pc_feat_"+PC_MODEL_PATH[-13:-5]+".pickle")
	
	if TRAINING_MODE == 2:
		output_to_file(database_feat["img_feat"],"database_img_feat_"+IMG_MODEL_PATH[-13:-5]+".pickle")
		output_to_file(query_feat["img_feat"],"query_img_feat_"+IMG_MODEL_PATH[-13:-5]+".pickle")
	
	if TRAINING_MODE == 3:
		#output_to_file(database_feat["pc_feat"],"database_pc_feat_"+MODEL_PATH[-13:-5]+".pickle")
		#output_to_file(query_feat["pc_feat"],"query_pc_feat_"+MODEL_PATH[-13:-5]+".pickle")
		#output_to_file(database_feat["img_feat"],"database_img_feat_"+MODEL_PATH[-13:-5]+".pickle")
		#output_to_file(query_feat["img_feat"],"query_img_feat_"+MODEL_PATH[-13:-5]+".pickle")
		output_to_file(database_feat["pcai_feat"],"database_pcai_feat_"+MODEL_PATH[-13:-5]+".pickle")
		output_to_file(query_feat["pcai_feat"],"query_pcai_feat_"+MODEL_PATH[-13:-5]+".pickle")
	
def get_load_batch_filename(dict_to_process,batch_keys,edge = False,remind_index = 0):
	pc_files = []
	img_files = []
	
	if edge :
		for i in range(BATCH_SIZE):
			cur_index = min(remind_index-1,i)
			pc_files.append(dict_to_process[batch_keys[cur_index]]["query"])			
			img_files.append(get_correspond_img(dict_to_process[batch_keys[cur_index]]["query"]))
	else:
		for i in range(BATCH_SIZE):
			pc_files.append(dict_to_process[batch_keys[i]]["query"])			
			img_files.append(get_correspond_img(dict_to_process[batch_keys[i]]["query"]))
		

	
	
	
	if TRAINING_MODE == 1:
		return pc_files,None
	if TRAINING_MODE == 2:
		return None,img_files
	if TRAINING_MODE == 3:
		return pc_files,img_files	

def prepare_batch_data(pc_data,img_data,ops):
	is_training=False
	if TRAINING_MODE == 1:
		train_feed_dict = {
			ops["is_training_pl"]:is_training,
			ops["pc_placeholder"]:pc_data}
		return train_feed_dict
		
	if TRAINING_MODE == 2:
		train_feed_dict = {
			ops["img_placeholder"]:img_data}
		return train_feed_dict
		
	if TRAINING_MODE == 3:
		train_feed_dict = {
			ops["is_training_pl"]:is_training,
			ops["img_placeholder"]:img_data,
			ops["pc_placeholder"]:pc_data}
		return train_feed_dict
		
	print("prepare_batch_data_error,no_such train mode.")
	exit()

def train_one_step(sess,ops,train_feed_dict):
	if TRAINING_MODE == 1:
		pc_feat= sess.run([ops["pc_feat"]],feed_dict = train_feed_dict)
		feat = {
			"pc_feat":pc_feat[0]}
		return feat

	if TRAINING_MODE == 2:
		img_feat= sess.run([ops["img_feat"]],feed_dict = train_feed_dict)
		feat = {
			"img_feat":img_feat[0]}
		return feat
		
	if TRAINING_MODE == 3:
		pcai_feat= sess.run([ops["pcai_feat"]],feed_dict = train_feed_dict)
		print(pcai_feat[0].shape)
		
		feat = {
			#"pc_feat":pc_feat,
			#"img_feat":img_feat,
			"pcai_feat":pcai_feat[0]}
		return feat
		
def init_all_feat():
	if TRAINING_MODE != 2:
		pc_feat = np.empty([0,256],dtype=np.float32)
	if TRAINING_MODE != 1:
		img_feat = np.empty([0,256],dtype=np.float32)
	if TRAINING_MODE == 3:
		pcai_feat = np.empty([0,1024],dtype=np.float32)
	
	if TRAINING_MODE == 1:
		all_feat = {"pc_feat":pc_feat}
	if TRAINING_MODE == 2:
		all_feat = {"img_feat":img_feat}
	if TRAINING_MODE == 3:
		all_feat = {
			"pc_feat":pc_feat,
			"img_feat":img_feat,
			"pcai_feat":pcai_feat}
	
	return all_feat
	
def concatnate_all_feat(all_feat,feat):
	if TRAINING_MODE == 1:
		print("all_feat ",all_feat["pc_feat"].shape)
		print("feat ",feat["pc_feat"].shape)
		all_feat["pc_feat"] = np.concatenate((all_feat["pc_feat"],feat["pc_feat"]),axis=0)
	if TRAINING_MODE == 2:
		all_feat["img_feat"] = np.concatenate((all_feat["img_feat"],feat["img_feat"]),axis=0)
	if TRAINING_MODE == 3:
		#all_feat["pc_feat"] = np.concatenate((all_feat["pc_feat"],feat["pc_feat"]),axis=0)
		#all_feat["img_feat"] = np.concatenate((all_feat["img_feat"],feat["img_feat"]),axis=0)
		all_feat["pcai_feat"] = np.concatenate((all_feat["pcai_feat"],feat["pcai_feat"]),axis=0)
	return all_feat			

def get_unique_all_feat(all_feat,dict_to_process):
	if TRAINING_MODE == 1:
		all_feat["pc_feat"] = all_feat["pc_feat"][0:len(dict_to_process.keys()),:]
	if TRAINING_MODE == 2:
		all_feat["img_feat"] = all_feat["img_feat"][0:len(dict_to_process.keys()),:]
	if TRAINING_MODE == 3:
		#all_feat["pc_feat"] = all_feat["pc_feat"][0:len(dict_to_process.keys()),:]
		#all_feat["img_feat"] = all_feat["img_feat"][0:len(dict_to_process.keys()),:]	
		all_feat["pcai_feat"] = all_feat["pcai_feat"][0:len(dict_to_process.keys()),:]			
	return all_feat
		
def get_latent_vectors(sess,ops,dict_to_process):
	print("dict_size = ",len(dict_to_process.keys()))
	train_file_idxs = np.arange(0,len(dict_to_process.keys()))
	all_feat = init_all_feat()
	for i in range(len(train_file_idxs)//BATCH_SIZE):
		batch_keys = train_file_idxs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
		pc_files=[]
		img_files=[]
		if i<0:
			print("Error, ready for delete")
			continue
		
		
		#select load_batch tuple
		load_pc_filenames,load_img_filenames = get_load_batch_filename(dict_to_process,batch_keys)
		
		begin_time = time()
		
		pc_data,img_data = load_img_pc(load_pc_filenames,load_img_filenames,pool)
		
		end_time = time()
		
		print ('load time ',end_time - begin_time)
		
		train_feed_dict = prepare_batch_data(pc_data,img_data,ops)
		
		begin_time = time()
		feat = train_one_step(sess,ops,train_feed_dict)
		end_time = time()
		print ('feature time ',end_time - begin_time)
		
		all_feat = concatnate_all_feat(all_feat,feat)
		
	#no edge case
	if len(train_file_idxs)%BATCH_SIZE == 0:
		return all_feat
	
	#hold edge case
	remind_index = len(train_file_idxs)%BATCH_SIZE
	tot_batches = len(train_file_idxs)//BATCH_SIZE		
	batch_keys = train_file_idxs[tot_batches*BATCH_SIZE:tot_batches*BATCH_SIZE+remind_index]
	
	load_pc_filenames,load_img_filenames = get_load_batch_filename(dict_to_process,batch_keys,True,remind_index)
	
	pc_data,img_data = load_img_pc(load_pc_filenames,load_img_filenames,pool)
	
	train_feed_dict = prepare_batch_data(pc_data,img_data,ops)
	
	feat = train_one_step(sess,ops,train_feed_dict)
	
	all_feat = concatnate_all_feat(all_feat,feat)
	all_feat = get_unique_all_feat(all_feat,dict_to_process)
	return all_feat
	
def	append_feat(all_feat,cur_feat):
	if TRAINING_MODE != 2:
		all_feat["pc_feat"].append(cur_feat["pc_feat"])
	if TRAINING_MODE != 1:
		all_feat["img_feat"].append(cur_feat["img_feat"])
	if TRAINING_MODE == 3:
		all_feat["pcai_feat"].append(cur_feat["pcai_feat"])
	return all_feat
	
def cal_all_features(ops,sess):
	if TRAINING_MODE != 2:
		database_pc_feat = []
		query_pc_feat = []
	if TRAINING_MODE != 1:
		database_img_feat = []
		query_img_feat = []
	if TRAINING_MODE == 3:
		database_pcai_feat = []
		query_pcai_feat = []
		
	if TRAINING_MODE == 1:
		database_feat = {
			"pc_feat":database_pc_feat}
		query_feat = {
			"pc_feat":query_pc_feat}
	if TRAINING_MODE == 2:
		database_feat = {
			"img_feat":database_img_feat}
		query_feat = {
			"img_feat":query_img_feat}
	if TRAINING_MODE == 3:
		database_feat = {
			"pc_feat":database_pc_feat,
			"img_feat":database_img_feat,
			"pcai_feat":database_pcai_feat}
		query_feat = {
			"pc_feat":query_pc_feat,
			"img_feat":query_img_feat,
			"pcai_feat":query_pcai_feat}
	
	
	for i in range(len(DATABASE_SETS)):
		cur_feat = get_latent_vectors(sess, ops, DATABASE_SETS[i])
		
		database_feat = append_feat(database_feat,cur_feat)
			
	for j in range(len(QUERY_SETS)):
		cur_feat = get_latent_vectors(sess, ops, QUERY_SETS[j])
		query_feat = append_feat(query_feat,cur_feat)
	
	save_feat_to_file(database_feat,query_feat)	

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
	bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,step*BATCH_SIZE,BN_DECAY_DECAY_STEP,BN_DECAY_DECAY_RATE,staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay

def init_imgnetwork():
	with tf.variable_scope("img_var"):
		img_placeholder = tf.placeholder(tf.float32,shape=[BATCH_SIZE,256,256,3])
		endpoints,body_prefix = resnet.endpoints(img_placeholder,is_training=True)
		img_feat = endpoints['img_var/resnet_v1_50/block4']
	return img_placeholder,img_feat
	
def init_pcnetwork(step):
	with tf.variable_scope("pc_var"):
		pc_placeholder = tf.placeholder(tf.float32,shape=[BATCH_SIZE,4096,3])
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
	if TRAINING_MODE != 2:
		pc_placeholder, is_training_pl, pc_feat = init_pcnetwork(step)	
	if TRAINING_MODE != 1:
		img_placeholder, img_feat = init_imgnetwork()
	if TRAINING_MODE == 3:
		pcai_feat = init_fusion_network(pc_feat,img_feat)

	#output of pcainetwork init
	if TRAINING_MODE == 1:
		ops = {
			"is_training_pl":is_training_pl,
			"pc_placeholder":pc_placeholder,
			"pc_feat":pc_feat}
		return ops
		
	if TRAINING_MODE == 2:
		ops = {
			"img_placeholder":img_placeholder,
			"img_feat":img_feat}
		return ops
		
	if TRAINING_MODE == 3:
		ops = {
			"is_training_pl":is_training_pl,
			"pc_placeholder":pc_placeholder,
			"img_placeholder":img_placeholder,
			#"pc_feat":pc_feat,
			#"img_feat":img_feat,
			"pcai_feat":pcai_feat}
		return ops
		
def init_train_saver():
	all_saver = tf.train.Saver()
	variables = tf.contrib.framework.get_variables_to_restore()
	pc_variable = [v for v in variables if v.name.split('/')[0]!='img_var']
	img_variable = [v for v in variables if v.name.split('/')[0]!='pc_var']
	
	pc_saver = tf.train.Saver(pc_variable)
	img_saver = tf.train.Saver(img_variable)
	
	train_saver = {
		'all_saver':all_saver,
		'pc_saver':pc_saver,
		'img_saver':img_saver}
	
	return train_saver
	
def init_network_variable(sess,train_saver):		
	if TRAINING_MODE == 1:
		train_saver['pc_saver'].restore(sess,PC_MODEL_PATH)
		print("pc_model restored")
		return
		
	if TRAINING_MODE == 2:
		train_saver['img_saver'].restore(sess,IMG_MODEL_PATH)
		print("img_model restored")
		return
	
	if TRAINING_MODE == 3:
		train_saver['all_saver'].restore(sess,MODEL_PATH)
		print("all_model restored")
		return

def main():
	#init network pipeline
	ops = init_pcainetwork()
	
	#init train saver
	train_saver = init_train_saver()

	#init GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	
	init_network_variable(sess,train_saver)
	print("model restored")
	
	cal_all_features(ops,sess)


if __name__ == "__main__":
	main()
