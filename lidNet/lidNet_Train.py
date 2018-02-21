#General Python packages
import time
import os
import os.path
import codecs
import warnings
import h5py
import boto3
import re

#Specific Python packages
import pandas as pd
import numpy as np
import cytoolz as ct
import multiprocessing as mp
from functools import partial
from random import randint
from random import shuffle 
from collections import defaultdict

#SciKit-Learn Dependencies
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from gensim.parsing import preprocessing

#Keras dependencies
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint	
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import load_model

#Language names
from get_names import get_names

#Set maximum length for training examples, in chars
MAX_LENGTH = 50

#-------------------------------------------------------------------------------------------------------------------------------------------------#

def train(feature_type = "tensor", 			#(str): "hashing" for character n-gram hashes; "tensor" for character sequence tensors
			model_type = "mlp",				#(str): "mlp" is a dense multi-layered perceptron; "cnn" uses convolutional layers
			filename_prefix = "",			#(str): A prefix for all file names; allows communication with non-local storage
			s3_storage = False,				#(boolean/str): If not False, use boto3 to connect to data on s3 bucket, with bucket name as string
			prefix = "",					#(str): If using s3 bucket, this is prefix for data folder; must end with "/"
			nickname = "Language",			#(str): The nickname for saving / loading models
			n_features = 524288,			#(int): Number of character n-gram features to allow, hashing only
			n_gram_tuple = (3,3),			#(tuple of ints): Range of n-grams to hash
			line_length = 200,				#(int): For tensor features, size of observations (i.e., number of characters; 100 or 
			max_chars = 20000, 				#(int): The maximum number of characters for tensor represetations; limits dimensionality
			divide_data = True,				#(boolean): If True, crawl for dataset; if False, just load it
			lang_threshold = 100,			#(int): Number of samples in the dataset required to include a language; can come from single domain
			domain_threshold = 2,			#(int): Number of domains in the dataset require to include a language; samples per domain doesn't matter
			n_samples = 2, 					#(int): Number of samples of each language+domain to use per epoch (currently, 18 possible domains)
			n_concat = 1,	 				#(int): Number of samples to concat per batch, to control memory use
			data_dir = "./Data",			#(str): Path to data directory; contains domain directories with language sub-directories
			load_vectorizer = True,			#(boolean): If using tensors, load a previously fit vectorizer for the character inventory
			vectorizer_file = "",			#(str): If loading a charcter inventory, this is the filename to use
			file_list = None,				#(str): If a string, file to load for list of training files (as a text file)
			write_files = False,			#(boolean): If True, write the file list to the specified file
			n_workers = 1,					#(int): Number of workers for Keras fit function
			q_size = 1,						#(int): Number of samples to queue for Keras fit function
			pickle_safe = False				#(boolean): Whether Keras multi-processing is pickle safe
			):
	
	#Get language info
	family_dict, merge_dict, remove_list = get_names()
	print("\n\tMax sample length: " + str(MAX_LENGTH) + " characters.")
	
	if divide_data == True:
		print("\n\tDividing data into testing, training, developing sets.")
		test_files, eval_files, train_files = crawl_data(data_dir, n_samples, remove_list, merge_dict, s3_storage, prefix)
		#Pruning and preparing files
		test_files, eval_files, train_files, lang_list = prep_work(test_files, eval_files, train_files, lang_threshold, domain_threshold, filename_prefix)

		write_file(os.path.join(filename_prefix, nickname + ".test_files.p"), test_files)
		write_file(os.path.join(filename_prefix, nickname + ".eval_files.p"), eval_files)
		write_file(os.path.join(filename_prefix, nickname + ".train_files.p"), train_files)
	
	else:
		print("\n\tLoading file lists.")
		test_files = read_file(os.path.join(filename_prefix, nickname + ".test_files.p"))
		eval_files = read_file(os.path.join(filename_prefix, nickname + ".eval_files.p"))
		train_files = read_file(os.path.join(filename_prefix, nickname + ".train_files.p"))
		test_files, eval_files, train_files, lang_list = prep_work(test_files, eval_files, train_files, lang_threshold, domain_threshold, filename_prefix)
	
	#Write out the languages being learned
	print("\n\tCurrent languages: " + str(len(lang_list)))
	print("\t", end="")
	counter = 0
	for lang in sorted(lang_list):
		counter += 1
		if counter > 10:
			print("")
			print("\t", end="")
			counter = 0
		print(" " + str(lang) + " ", end="")
		
	#Get language families
	family_list = [family_dict[x] for x in lang_list]
	family_list = list(set(family_list))
	
	#Print language names
	print("\n\n\tCurrent Langauge Names: " + str(len(family_list)))
	print("\t", end="")
	counter = 0
	for family in sorted(family_list):
		counter += 1
		if counter > 3:
			print("")
			print("\t", end="")
			counter = 0
		
		try:
			padding = 40 - len(family)
			padding = str(" " * padding)
			print("" + str(family) + padding, end="")
			
		except:
			family = family.encode()
			padding = 40 - len(family)
			padding = str(" " * padding)
			print("" + str(family) + padding, end="")
		
	#Display domains
	domain_list = [x[1] for x in train_files]
	domain_list = list(set(domain_list))
	print("\n\n\tCurrent domains: " + str(len(domain_list)))
	print("\t", end="")
	counter = 0
	for domain in sorted(domain_list):
		domain = domain.split("\\")
		domain = domain[-1]
		counter += 1
		if counter > 1:
			print("")
			print("\t", end="")
			counter = 0
		padding = 15 - len(domain)
		padding = str(" " * padding)
		print("" + str(domain) + padding, end="")
	
	print("\n\n")
	print("\tSize of training set: " + str(len(train_files) * 100))
	print("\tSize of test set: " + str(len(test_files) * 100))
	print("\tSize of eval set: " + str(len(eval_files) * 100))
		
	print("\n\tInitializing feature extraction and input layers: " + feature_type)
	if feature_type == "hashing":
		x_encoder = get_x_hashing(n_features, n_gram_tuple)
			
		#Define what the input shape looks like for hashing (vectors)
		inputs = Input(shape=(n_features,), name = "input", dtype = "float32")
			
		if model_type == "cnn":
			print("CNNs are not currently able to use hashing features.")
			sys.kill()
		
	elif feature_type == "tensor":

		chars, char_indices = prep_x_tensor(train_files, max_chars, filename = vectorizer_file, load = load_vectorizer)
		x_encoder = partial(get_x_tensor, line_length = line_length, chars = chars, char_indices = char_indices)

		#Define what the input shape looks like for tensors
		inputs = Input(shape=(line_length, len(chars), ), name = "input", dtype = "int32")
	
	print("\n\n\tTraining model for classifying languages.")
	
	#Get class encoder for language classification
	class_array = np.array(lang_list)
	n_classes = len(lang_list)
	y_encoder = get_y_encoder(class_array)
	n_per_epoch = get_n_epoch(train_files, n_samples)
	print("\n\tNumber of samples per epoch: " + str(n_per_epoch * 100))
	
	write_file(os.path.join(filename_prefix, "Model." + nickname + ".Callback.hdf.Langs"), class_array)

	#Try to load previous callback model
	model_filename = os.path.join(filename_prefix, "Model." + nickname + ".Callback.hdf")
	if os.path.isfile(model_filename):
		model_language = load_model(model_filename)
		
	#Initializing Keras models for language classification	
	else:
	
		if model_type == "mlp":
			model_language = build_model_mlp(inputs = inputs, cat_output = n_classes, feature_type = feature_type)
			
		elif model_type == "cnn":
			model_language = build_model_cnn(inputs = inputs, maxlen = line_length, cat_output = n_classes, feature_type = feature_type)
			
		elif model_type == "mlp_embedding":
			model_language = build_model_mlp_embedding(inputs = inputs, cat_output = n_classes, feature_type = feature_type, max_chars = max_chars, line_length = line_length)
		
	#Stop learning once reach maturity
	model_filename = os.path.join(filename_prefix, "Model." + nickname + ".Callback.hdf")
	
	callbacks = [EarlyStopping(monitor = "val_acc", mode = "max", min_delta = 0.0, patience = 25, verbose = 1),
				ModelCheckpoint(model_filename, monitor = "val_acc", mode = "max", save_best_only = True, period = 1, verbose = 1)]
	
	#Only fit the language model if a final version isn't saved
	filename = os.path.join(filename_prefix, "Model." + nickname + "." + model_type + "." + feature_type + ".p")
	if not os.path.isfile(filename):
	
		#Start fitting language
		model_language.fit_generator(generate_data(train_files, x_encoder, y_encoder, n_classes, n_features, n_samples, n_concat, feature_type), 
										steps_per_epoch = int(n_per_epoch / n_concat),
										epochs = 500000, 
										verbose = 1, 
										validation_data = generate_data(test_files, x_encoder, y_encoder, n_classes, n_features, n_samples, n_concat, feature_type),
										validation_steps = int(n_classes * n_samples),
										workers = 1,
										callbacks = callbacks,
										use_multiprocessing = False, 
										max_queue_size = 100
										)

		#Save final language model
		model_language.save(filename)
		
	#Finished language model already exists, so load it
	model_language = load_model(filename)
	
	#Final evaluation on held-out data
	model_language.evaluate_generator(generator = generate_data(eval_files, x_encoder, y_encoder, n_classes, n_features, n_samples, n_concat, feature_type),
										steps = len(eval_files), 
										max_q_size = q_size,
										workers = n_workers,
										pickle_safe = pickle_safe
										)

	return model_language
#----------------------------------------------------------------------------------#

def crawl_data(root_directory, n_samples, remove_list = [], merge_dict = {}, s3_storage = False, prefix = ""):

	#Dataset is organized in folders following the pylangid convention:
	#--- Root / Domain / Lang
	#--- This crawls that dataset to create the training / testing / validating data
	
	root_directory = os.path.join(".", root_directory)
	develop_files = []
	train_files = []
	test_files = []
	lang_list = []
	write_list = []
	file_list = []
	
	#Use boto3 to get s3 data
	if s3_storage != False:
	
		print("\tConnecting to s3 bucket: " + s3_storage)
		client = boto3.client("s3")
		print(s3_storage, prefix)
		paginator = client.get_paginator("list_objects_v2")
		operation_parameters = {"Bucket": s3_storage,
								"Prefix": prefix}
		page_iterator = paginator.paginate(**operation_parameters)
		
		for page in page_iterator:

			for key in page["Contents"]:
				filename = key["Key"]
				
				temp_list = filename.split("/")
				
				if len(temp_list) > 2:
					domain = temp_list[1]
					lang = temp_list[2]
					
					if lang in merge_dict.keys():
						lang = merge_dict[lang]
						
					if lang not in remove_list:
						file_list.append((lang, domain, filename))
				
	#If not using s3, crawl local data
	else:

		for subdir, dirs, files in os.walk(root_directory):

			if subdir == root_directory:
				domain_list = dirs
				
			else:
				if len(dirs) > 1:	
					lang_list += dirs
					
				elif len(files) > 1:
					
					shuffle(files)
					files = [os.path.join(subdir, x) for x in files]
					
					language = subdir[-3:]
					domain = os.path.split(subdir)
					domain = domain[-2]
					
					[write_list.append(x) for x in files]
					
					if language in merge_dict.keys():
						language = merge_dict[language]						
					
					if language not in remove_list:
						for filename in files:
							file_list.append((language, domain, filename))
							
	#Now assign to training / testing / eval sets
	test_dict  = defaultdict(int)
	develop_dict = defaultdict(int)		
	
	for lang, domain, filename in file_list:
		
		#Check and add test files
		if test_dict[(lang, domain)] < n_samples:
			test_files.append((lang, domain, filename))
			test_dict[(lang, domain)] += 1
			
		elif develop_dict[(lang, domain)] < n_samples:
			develop_files.append((lang, domain, filename))
			develop_dict[(lang, domain)] += 1
			
		else:
			train_files.append((lang, domain, filename))
														
	return test_files, develop_files, train_files
#----------------------------------------------------------------------------------#

def get_n_epoch(file_list, n_samples):

	#Figure out how many samples to use per epoch
	#-- The goal is to sample each lang/domain pair n times

	counter = 0
	lang_dict = {}
		
	#Iterate over language files for this loop
	for pair in file_list:
		
		lang = pair[0]
		domain = pair[1]
		filename = pair[2]
			
		#Check samples from current lang
		if lang in lang_dict:
			if domain in lang_dict[lang]:
				domain_count = lang_dict[lang].count(domain)
				domain_count = int(domain_count)
										
				#Add another language-domain sample
				if domain_count < n_samples:
					grab_it = 1
					lang_dict[lang].append(domain)
						
				#Domain and language already sufficiently sampled
				else:
					grab_it = 0
						
			#New domain, add it
			else:
				lang_dict[lang].append(domain)
				grab_it = 1
			
		#Add current lang to sampling record
		else:
			lang_dict[lang] = []
			lang_dict[lang].append(domain)
			grab_it = 1
							
		#If this language has not already been oversampled
		if grab_it == 1:
			counter += 1
			
	return counter
#----------------------------------------------------------------------------------#

def generate_data(file_list, x_encoder, y_encoder, n_classes, n_features, n_samples, n_concat, feature_type):

	#Feed training / testing data to Keras
	concat_counter = 0
	
	if feature_type == "hashing" and MAX_LENGTH < 100:
		feature_type = "hashing_reduced"

	while True:
	
		#Randomize files for this loop
		#lang_dict ensures that languages and domains are sampled equally per epoch
		#Even if using family as class, each language is still sampled
		shuffle(file_list)
		lang_dict = {}
			
		#Iterate over language files for this loop
		for pair in file_list:
			
			lang = pair[0]
			domain = pair[1]
			filename = pair[2]
			
			#Check samples from current lang
			if lang in lang_dict:
				if domain in lang_dict[lang]:
					domain_count = lang_dict[lang].count(domain)
					domain_count = int(domain_count)
										
					#Add another language-domain sample
					if domain_count < n_samples:
						grab_it = 1
						lang_dict[lang].append(domain)
						
					#Domain and language already sufficiently sampled
					else:
						grab_it = 0
						
				#New domain, add it
				else:
					lang_dict[lang].append(domain)
					grab_it = 1
			
			#Add current lang to sampling record
			else:
				lang_dict[lang] = []
				lang_dict[lang].append(domain)
				grab_it = 1
							
			#If this language has not already been oversampled
			if grab_it == 1:
			
				#Get feature vector
				try:
					with codecs.open(filename, "r", encoding = "utf-8") as fo:
						
						if feature_type == "tensor":
							X_vector = x_encoder(fo = fo)
						
						elif feature_type == "hashing":
							X_vector = x_encoder(fo)
						
						elif feature_type == "hashing_reduced":
							line_list = []
							for line in fo:
								line_list.append(line[:100])
								line_list.append(line[100:])
							X_vector = x_encoder(line_list)
					
					#Get class vector
					if feature_type == "tensor":
						instances = X_vector.shape[0]
										
					elif feature_type == "hashing" or feature_type == "hashing_reduced":
						instances = X_vector.shape[0]
						X_vector = X_vector.toarray()

					y_temp = [lang for x in range(instances)]				
					y_vector = np.array(y_temp)
					y_vector = y_encoder.transform(y_vector)
					y_vector = np_utils.to_categorical(y_vector, num_classes = n_classes)

					yield X_vector, y_vector
					
				except Exception as e:
					error_counter = 0
					print("")
					print(e)
					print(filename)
					print(pair)
					print("")
#----------------------------------------------------------------------------------#

def prep_work(test_files, eval_files, train_files, lang_threshold, domain_threshold, filename_prefix):

	#Filter by number of samples
	lang_list = [x[0] for x in train_files]
	starting = len(set(lang_list))
	lang_dict = ct.frequencies(lang_list)
	threshold = lambda x: x  >= lang_threshold
	lang_dict = ct.valfilter(threshold, lang_dict)
	lang_list = list(lang_dict.keys())
	print("\t\tReducing initial set of " + str(starting) + " languages to " + str(len(lang_list)) + " after frequency threshold.")
	
	#Filter by number of domains
	if domain_threshold > 1:
		domain_dict = {}
		for lang, domain, file in train_files:
			try:
				domain_dict[lang].append(domain)
				
			except:
				domain_dict[lang] = []
				domain_dict[lang].append(domain)
		
		domain_threshold_list = []		
		
		for lang in lang_list:
			temp_list = list(set(domain_dict[lang]))
			if len(temp_list) >= domain_threshold:
				domain_threshold_list.append(lang)
		lang_list = domain_threshold_list		
	
	#Prune and shuffle file lists
	test_files = [x for x in test_files if x[0] in lang_list]
	test_files = [(x[0], x[1], os.path.join(str(filename_prefix), str(x[2]))) for x in test_files]
	shuffle(test_files)
	
	train_files = [x for x in train_files if x[0] in lang_list]
	train_files = [(x[0], x[1], os.path.join(str(filename_prefix), str(x[2]))) for x in train_files]
	shuffle(train_files)
	
	eval_files = [x for x in eval_files if x[0] in lang_list]
	eval_files = [(x[0], x[1], os.path.join(str(filename_prefix), str(x[2]))) for x in eval_files]
	shuffle(eval_files)
	
	return test_files, eval_files, train_files, lang_list
#--------------------------------------------------------------------------------------#

def get_y_encoder(class_array):
	
	#Initialize encoder
	encoder = LabelEncoder()
	encoder.fit(class_array)
	
	return encoder
#--------------------------------------------------------------------------------------#

def x_hashing_pre(line, myre):

	#Remove links, hashtags, at-mentions, mark-up, and "RT"
	line = re.sub(r"http\S+", "", line)
	line = re.sub(r"@\S+", "", line)
	line = re.sub(r"#\S+", "", line)
	line = re.sub("<[^>]*>", "", line)
	line = line.replace(" RT", "").replace("RT ", "")
	
	#Remove emojis
	line = re.sub(myre, "", line)
	
	#Remove punctuation and extra spaces
	line = ct.pipe(line, preprocessing.strip_tags, preprocessing.strip_punctuation, preprocessing.strip_numeric, preprocessing.strip_non_alphanum, preprocessing.strip_multiple_whitespaces)
	
	#Strip and reduce to max training length
	line = line.lower().strip().lstrip()
	line = line[0:MAX_LENGTH]
	
	return line
#--------------------------------------------------------------------------------------#

def get_x_hashing(n_features, n_gram_tuple):

	try:
	# Wide UCS-4 build
		myre = re.compile(u'['
			u'\U0001F300-\U0001F64F'
			u'\U0001F680-\U0001F6FF'
			u'\u2600-\u26FF\u2700-\u27BF]+', 
			re.UNICODE)
	except re.error:
		# Narrow UCS-2 build
			myre = re.compile(u'('
			u'\ud83c[\udf00-\udfff]|'
			u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
			u'[\u2600-\u26FF\u2700-\u27BF])+', 
			re.UNICODE)
			
	hashing_partial = partial(x_hashing_pre, myre = myre)
	
	#Initialize vectorizer
	vectorizer = HashingVectorizer(encoding = "utf-8",
								decode_error = "strict",
								analyzer = "char",
								preprocessor = hashing_partial,
								ngram_range = n_gram_tuple,
								stop_words = None,
								lowercase = False,
								norm = None,
								n_features = n_features
								)
	
	#Partial so that vectorizer only takes file object								
	vectorizer = vectorizer.transform
									
	return vectorizer
#----------------------------------------------------------------------------------------#

def get_x_tensor(line_length, chars, char_indices, fo):
		
	sentences = []
		
	#Get feature vector
	for line in fo:
							
		line = line.strip().lower()
							
		if len(line) > line_length:
			line = line[0:line_length]
								
		elif len(line) < line_length:
			extra = line_length - len(line)
			line += str(" " * extra)
									
		sentences.append(line)
		
	#Prepare a one-hot X encoding of this file
	X = np.zeros((len(sentences), line_length, len(chars)), dtype = np.bool)

	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			if char in chars:
				X[i, t, char_indices[char]] = 1
				
	return X
#----------------------------------------------------------------------------------------#

def prep_x_tensor(train_files, max_chars, filename = "CountVectorizer.p", load = False):
	
	starting = time.time()
	
	if load == False:
		
		#Get set of characters
		vectorizer = CountVectorizer(input = "filename", 
									encoding = "utf-8",
									decode_error = "replace",
									strip_accents = None,
									analyzer = "char",
									lowercase = True,
									preprocessor = None,
									tokenizer = None,
									stop_words = None,
									ngram_range = (1, 1), 
									max_df = 1.0, 
									min_df = 0.0, 
									max_features = max_chars,
									vocabulary = None, 
									binary = False
									)
		
		vectorizer.fit([x[2] for x in train_files])							
		write_file(filename, vectorizer)
		
	else:
		vectorizer = read_file(filename)
				
	chars = vectorizer.get_feature_names()

	#Done with character inventory
	n_chars = len(chars)
	print("\n\n\tTotal characters: " + str(n_chars) + " in " + str(time.time() - starting))

	#Get char indices
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))
	
	return chars, char_indices
#----------------------------------------------------------------------------------------#

def pop_layer(model):
	if not model.outputs:
		raise Exception('Sequential model cannot be popped: model is empty.')

	model.layers.pop()
	if not model.layers:
		model.outputs = []
		model.inbound_nodes = []
		model.outbound_nodes = []
	else:
		model.layers[-1].outbound_nodes = []
		model.outputs = [model.layers[-1].output]
	model.built = False
#--------------------------------------------------------------------------------------#

def prep_existing_model(model_file, feature_type):

	#Take a pre-trained model, reload it, and prepare it for adding to a new model
	
	if isinstance(model_file, str):
		print("\tLoading " + str(model_file) + " and removing prediction layers.")
		loaded_model = load_model(model_file)
	
	else:
		loaded_model = model_file

	print("Current state of loaded model: ")
	loaded_model.summary()
	
	if feature_type == "tensor":
	
		#Remove last layer until the flattened layer is gone
		while True:
			
			name_list = [x.name for x in loaded_model.layers]
			print("\t\t\tCurrent layers: " + str(name_list))
			
			if "flatten" not in name_list:
				pop_layer(loaded_model)
				
			else:
				pop_layer(loaded_model)
				break

	#Rename dropout layers and set all as untrainable
	for i in range(len(loaded_model.layers)):

		#Rename layers
		loaded_model.layers[i].name = "pretrained_" + str(loaded_model.layers[i].name)
		
		#Set to untrainable
		loaded_model.layers[i].trainable = False	

	print("\n\n\tDone removing prediction layers on previous model. Current state:")
	loaded_model.summary()
	
	return loaded_model
#----------------------------------------------------------------------------------------#

def build_model_mlp(inputs, cat_output, feature_type):

	print("\n\tInitializing Keras Model, MLP")

	#Set sizes for each mlp layer
	mlp_sizes = [500, 500, 500, 500]
	
	layer1 = Dense(mlp_sizes[0], activation = "relu")(inputs)
	layer1 = Dropout(0.25)(layer1)
	
	layer2 = Dense(mlp_sizes[1], activation = "relu")(layer1)
	layer2 = Dropout(0.25)(layer2)
	
	layer3 = Dense(mlp_sizes[2], activation = "relu")(layer2)
	layer3 = Dropout(0.25)(layer3)
	
	layer4 = Dense(mlp_sizes[3], activation = "relu")(layer3)
	layer4 = Dropout(0.25)(layer4)
	
	#Tensor inputs need to be flattened before prediction
	if feature_type == "tensor":
		layer4 = Flatten()(layer4)
	
	#Output dense layer with softmax activation
	pred = Dense(cat_output, activation = "softmax", name = "output")(layer4)

	#Inputs depend on whether we are adding to pre-trained model
	model = Model(inputs = inputs, outputs = pred)
	
	#Now compile
	model.compile(loss = "categorical_crossentropy", 
					optimizer = "sgd",
					metrics = ["accuracy"]
					)
					
	print("\n\n\tFinal state:")
	model.summary()

	return model
#--------------------------------------------------------------------------------------------------#

def write_file(file, candidate_list):
	
	import pickle
	import os.path
	import os
	
	if os.path.isfile(file):
		os.remove(file)
	
	with open(file,'wb') as f:
		pickle.dump(candidate_list,f)
	
	return
#--------------------------------------------------------------------------------------------------#

def read_file(file):
	
	import pickle
	
	with open(file,'rb') as f:
		candidate_list = pickle.load(f)
		
	return candidate_list
#--------------------------------------------------------------------------------------------------#