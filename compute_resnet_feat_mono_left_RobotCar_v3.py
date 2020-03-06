import numpy as np
from loading_input_v3 import *
import nets_v3.resnet_v1_50 as resnet
import tensorflow as tf
from time import *
import pickle
from multiprocessing.dummy import Pool as ThreadPool

#thread pool
pool = ThreadPool(40)

# 1 for point cloud only, 2 for image only, 3 for pc&img&fc
TRAINING_MODE = 2
BATCH_SIZE = 100
EMBBED_SIZE = 256

DATABASE_FILE= 'generate_queries_v3/mono_left_RobotCar_oxford_evaluation_database.pickle'
QUERY_FILE= 'generate_queries_v3/mono_left_RobotCar_oxford_evaluation_query.pickle'
PC_IMG_MATCH_FILE = 'generate_queries_v3/mono_left_pointcloud_image_match_test.pickle'
DATABASE_SETS= get_sets_dict(DATABASE_FILE)
QUERY_SETS= get_sets_dict(QUERY_FILE)
PC_IMG_MATCH_DICT = get_pc_img_match_dict(PC_IMG_MATCH_FILE)

#model_path & image path
SENSOR = "mono_left"
IMAGE_PATH = '/data/lyh/RobotCar/mono_left_color/'
IMG_MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3/log/train_save_v3_resnet/img_model_00198066.ckpt"

def get_correspond_img(pc_filename):
	splited = pc_filename.split('/')
	timestamp = splited[-1][:-4]
	seq_name = splited[-3]
	image_ind = PC_IMG_MATCH_DICT[seq_name][timestamp]
	if len(image_ind) <= 0:
		return None
	
	#print("len(image_ind) = ",len(image_ind))
	random.shuffle(image_ind)
	for i in range(len(image_ind)):
		image_filename = os.path.join(IMAGE_PATH,seq_name,SENSOR,"%s.png"%(image_ind[i]))
		if os.path.exists(image_filename):
			return image_filename	
	return None
	
def output_to_file(output, filename):
	with open(filename, 'wb') as handle:
		pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Done ", filename)

def save_feat_to_file(database_feat,query_feat):
	output_to_file(database_feat["img_feat"],"database_img_feat_"+IMG_MODEL_PATH[-13:-5]+".pickle")
	output_to_file(query_feat["img_feat"],"query_img_feat_"+IMG_MODEL_PATH[-13:-5]+".pickle")
	
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
		

	return None,img_files

def prepare_batch_data(pc_data,img_data,ops):
	is_training=False
	train_feed_dict = {
		ops["img_placeholder"]:img_data}
	return train_feed_dict
	
def train_one_step(sess,ops,train_feed_dict):
	img_feat= sess.run([ops["img_feat"]],feed_dict = train_feed_dict)
	feat = {
		"img_feat":img_feat[0]}
	return feat
		
def init_all_feat():
	img_feat = np.empty([0,1000],dtype=np.float32)
	all_feat = {"img_feat":img_feat}
	return all_feat
	
def concatnate_all_feat(all_feat,feat):
	all_feat["img_feat"] = np.concatenate((all_feat["img_feat"],feat["img_feat"]),axis=0)
	return all_feat			

def get_unique_all_feat(all_feat,dict_to_process):
	all_feat["img_feat"] = all_feat["img_feat"][0:len(dict_to_process.keys()),:]		
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
	all_feat["img_feat"].append(cur_feat["img_feat"])
	return all_feat
	
def cal_all_features(ops,sess):
	database_img_feat = []
	query_img_feat = []

	database_feat = {
		"img_feat":database_img_feat}
	query_feat = {
		"img_feat":query_img_feat}
	
	
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
		endpoints,body_prefix = resnet.endpoints(img_placeholder,is_training=False)
		img_feat = tf.layers.dense(endpoints,1000,activation=tf.nn.relu)
	return img_placeholder,img_feat

def init_pcainetwork():	
	#init sub-network
	img_placeholder, img_feat = init_imgnetwork()

	#output of pcainetwork init
	ops = {
		"img_placeholder":img_placeholder,
		"img_feat":img_feat}
	return ops
		
def init_train_saver():
	all_saver = tf.train.Saver()
	
	train_saver = {
		'all_saver':all_saver}
	
	return train_saver

def init_network_variable(sess,train_saver):
	sess.run(tf.global_variables_initializer())
	
	train_saver['all_saver'].restore(sess,IMG_MODEL_PATH)
	print("img_model restored")
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
