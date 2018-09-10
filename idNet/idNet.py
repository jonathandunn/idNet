#General Python packages
import time
import os
import os.path
import codecs
import warnings
import h5py
import boto3
import re
import pickle

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
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import vstack

#Keras dependencies
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint	
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import LambdaCallback
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Flatten
from keras.models import load_model

#idNet modules, fix if necessary
try:
	from idNet.modules.Loader import Loader
	from idNet.modules.Crawler import Crawler
	from idNet.modules.Names import Names
	from idNet.modules.Features import Features
	
except Exception as e:
	if e.__class__ == ModuleNotFoundError:
		from idNet.idNet.modules.Loader import Loader
		from idNet.idNet.modules.Crawler import Crawler
		from idNet.idNet.modules.Names import Names
		from idNet.idNet.modules.Features import Features

#Keras callback class to upload model checkpoints to s3 -----------------------------#
class SaveCheckpoint(Callback):

	def __init__(self,  s3, filename1, filename2, Load):
		self.s3 = s3
		self.filename1 = filename1
		self.filename2 = filename2
		self.Load = Load

	def on_epoch_begin(self, two, three):
		if os.path.exists(self.filename1):
			if self.s3 == True:
				self.Load.upload_to_s3(self.filename1, self.Load.output_dir + "/" + self.filename1, delete_flag = False)
				self.Load.upload_to_s3(self.filename2, self.Load.output_dir + "/" + self.filename2, delete_flag = False)

#-------------------------------------------------------------------------------------------------------------------------------------------------#

class idNet_Enrich(object):

	def __init__(self, model_name, language = "", s3_bucket = None):
	
		self.model_name = model_name
		self.s3_bucket = s3_bucket
		self.language = language
		
		#Get type
		if ".DID" in model_name:
			self.type = "DID"
		else:
			self.type = "LID"
			
		#Get architecture
		if ".SVM" in model_name:
			self.architecture = "SVM"
		else:
			self.architecture = "MLP"
			
		#Load idNet_Train for support functions
		self.Support = idNet_Train(type = self.type, input = "", output = "", support = True)
		self.Features = Features(Loader = False, type = self.type, sample_size = 10000, language = self.language)
			
		#Save model from s3 to current working directory
		if self.s3_bucket != None:
			
			#Initialize boto3 client
			import boto3
			client = boto3.client("s3")
			
			#Dowload to local location
			client.download_file(self.s3_bucket, model_name, "temp_model.p")
			
			#Download additional MLP file
			if self.architecture == "MLP":
				client.download_file(self.s3_bucket, model_name + ".Langs", "temp_model.p" + ".Langs")
				
			#Model file is now local; rename it
			model_name = "temp_model.p"
		
		#Now load local models
		if self.architecture == "MLP":
			self.model = load_model(model_name)
			with open(model_name + ".Langs", "rb") as handle:
				self.class_array = pickle.load(handle)
				self.class_array = sorted(list(self.class_array))
				
		elif self.architecture == "SVM":
			with open(model_name, "rb") as handle:
				svm_model = pickle.load(handle)
				
			with open(model_name + ".Classes", "rb") as handle:
				self.class_array = pickle.load(handle)
				
		#Prep for LID with MLP
		if self.architecture == "MLP":
			self.n_features = self.model.layers[0].input_shape[1]
			self.n_classes = self.model.layers[-1].output_shape[1]
			
			print("\tNumber of features: " + str(self.n_features))
			print("\tNumber of classes: " + str(self.n_classes))
			
			if ".1-3grams" in model_name:
				n_gram_tuple = (1,3)
			elif ".1-4grams" in model_name:
				n_gram_tuple = (1,4)
			
			#Check compatibility
			if len(self.class_array) != self.n_classes:
				print("\t\tModel is not compatible with provided classes.")
				sys.kill()
				
			self.x_encoder = self.Features.get_x_lid(self.n_features, n_gram_tuple)
			self.y_encoder = self.Support.get_y_encoder(self.class_array)
			
			#Get dictionary of class labels
			self.class_dict = {}
			for i in range(len(self.class_array)):
				self.class_dict[i] = self.class_array[i]
			self.class_dict[self.n_classes] = "und"
			
		elif self.architecture == "SVM":
			self.language = svm_model["language"]
			self.model = svm_model["cls"]
			self.normalize_check = svm_model["normalize_check"]
			self.chi_square_check = svm_model["chi_square_check"]
			self.y_encoder = svm_model["y_encoder"]
			self.grammar = svm_model["grammar"]
			self.constructions = svm_model["constructions"]
			self.weights = svm_model["weights"]
				
			#Save feature mask
			if self.chi_square_check == True:
				self.significant_features = svm_model["significant_features"]
	
	#---------------------------------------------------------------------------------------------#

	def predict(self, input, workers = 1):
			
		#Accepts a single text as a str or a list of texts
		if isinstance(input, str):
			input = [input]
			
		#For LID predict
		if self.type == "LID" and self.architecture == "MLP":
		
			#Get features
			X_vector = self.x_encoder(input)
			X_vector = X_vector.todense()

			#Get model output
			y_predict = self.model.predict(X_vector)
				
			#Get highest value classes
			y_classes = np.argmax(y_predict, axis = 1)

			#Allows for uncertainty in predictions
			#y_strength = np.amax(y_predict, axis = 1)
			#y_classes[y_strength < prediction_threshold] = n_classes + 1

			#Convert to str language labels
			y_classes = [self.class_array[x] for x in y_classes]
				
			return y_classes

		#For DID predict		
		elif self.type == "DID" and self.architecture == "SVM":
		
			#Get features
			X_vector = self.Features.CxG.parse_return(input, mode = "idNet", workers = workers)
			X_vector = np.vstack(X_vector)
			
			#Feature pruning if necessary
			if self.chi_square_check == True:
				X_vector = X_vector[:, self.significant_features]
			
			y_classes = self.model.predict(X_vector)
			y_classes = [self.class_array[x] for x in y_classes]
			
			return y_classes
			
#-------------------------------------------------------------------------------------------------------------------------------------------------#

class idNet_Train(object):

	def __init__(self, 
				type,							#(str): Whether to work with language or dialect identification
				input,							#(str): Path to input folder
				output,							#(str): Path to output folder
				s3 = False,						#(boolean): If True, use boto3 to interact with s3 bucket
				s3_bucket = "",					#(str): s3 bucket name as string
				nickname = "Language",			#(str): The nickname for saving / loading models
				divide_data = True,				#(boolean): If True, crawl for dataset; if False, just load it
				test_samples = 20,				#(int): The number of files for each class to use for testing; note that LID and DID have different samples per file
				threshold = 100,				#(int): Number of files required before language or country is included in the model
				samples_per_epoch = 5,			#(int): Number of samples to use per training epoch for language-domain or language-country pairs
				language = "",					#(str): For DID, specifies the language of the current model
				lid_sample_size = 200,			#(int): For LID, the number of characters to allow per sample
				did_sample_size	= 1,			#(int): For DID, the number of 100 word samples to combine
				preannotate_cxg = False,		#(boolean): For DID, if True enrich and save all CxG vectors
				preannotated_cxg = False,		#(boolean): For DID, if True just load pre-enriched CxG vectors
				cxg_workers = 1,				#(int):	For DID, if pre-enriching dataset, number of workers to use
				class_constraints = [],			#(list of strs): Option to constrain the number of classes
				merge_dict = {},				#(dict): Original:New name keys
				support = False					#(boolean): For support functions only, don't do file prep
				):
		
		#Initiate loading class for dealing with files
		self.type = type
		self.Names = Names()
		if input != "" and output != "":
			self.Load = Loader(input, output, s3, s3_bucket)
			self.Crawler = Crawler(self.Load, self.type, language)
			self.test_samples = test_samples
			self.samples_per_epoch = samples_per_epoch
			self.preannotate_cxg = preannotate_cxg
			self.preannotated_cxg = preannotated_cxg
			self.s3 = s3
			self.output = output
			self.nickname = nickname
		
		#For support functions, no prep is required
		if support == False:
		
			#If pre-enrching, force divide data
			if preannotate_cxg == True:
				divide_data = True
			
			#Load existing files if necessary
			if divide_data == False:
				print("\n\tLoading file lists.")
				self.test_files = self.Load.load_file(nickname + ".test_files.p")
				self.eval_files = self.Load.load_file(nickname + ".eval_files.p")
				self.train_files = self.Load.load_file(nickname + ".train_files.p")
				self.class_list = self.Load.load_file("Model." + nickname + ".Callback.hdf.Classes")
			
			#Prepared for LID
			if type == "LID":
				if divide_data == True:
					
					print("\n\tDividing data into testing, training, developing sets.")
					self.test_files, self.eval_files, self.train_files, self.class_list = self.Crawler.crawl(test_samples, threshold, self.Names.remove_list, self.Names.merge_dict)
					
			#Prepare for DID
			elif type == "DID":
				if divide_data == True:
					
					print("\n\tDividing data into testing, training, developing sets.")
					self.test_files, self.eval_files, self.train_files, self.class_list = self.Crawler.crawl(test_samples, threshold, self.Names.remove_list, merge_dict)
					
			#Save files if necessary
			if divide_data == True:
				self.Load.save_file(self.test_files, nickname + ".test_files.p")
				self.Load.save_file(self.eval_files, nickname + ".eval_files.p")
				self.Load.save_file(self.train_files, nickname + ".train_files.p")

			#Reduce classes if necessary
			if class_constraints != []:
				self.class_list = [x for x in self.class_list if x in class_constraints]
				self.test_files = [x for x in self.test_files if x[2] in self.class_list]
				self.eval_files = [x for x in self.eval_files if x[2] in self.class_list]
				self.train_files = [x for x in self.train_files if x[2] in self.class_list]
				
			#Define samples per file
			if type == "LID":
				self.sample_size = lid_sample_size
				print("\n\tLID Sample Size in characters: " + str(lid_sample_size))
				if lid_sample_size <= 100:
					print_per_file = 200
				else:
					print_per_file = 100
			
			#or, Define DID sample size
			elif type == "DID":
				self.sample_size = did_sample_size
				print_per_file = int(2000 / did_sample_size)
				print("\n\tDID Sample Size in words: " + str(100 * did_sample_size))
				
			print("\tSize of training set: " + str(len(self.train_files) * print_per_file))
			print("\tSize of test set: " + str(len(self.test_files) * print_per_file))
			print("\tSize of eval set: " + str(len(self.eval_files) * print_per_file))	
			
			print("\n\tInitializing number of classes and epoch size.")
			self.class_array = np.array(self.class_list)
			self.n_classes = len(self.class_list)
			self.y_encoder = self.get_y_encoder(self.class_array)
			self.n_per_epoch = self.Crawler.get_n_epoch(self.train_files, self.samples_per_epoch, self.type)
			print("\tNumber of samples per epoch: " + str(self.n_per_epoch * print_per_file))
			print("\tNumber of lang/domain or lang/country pairs to use per epoch: " + str(self.samples_per_epoch))
			
			print("\n\tCurrent classes (" + str(len(self.class_list)) + "):")
			for thing in sorted(self.class_list):
				print(thing, end = ", ")
			print("\n")
			
			#Set language to number of languages for LID
			if type == "LID":
				self.language = str(self.n_classes)
			elif type == "DID":
				self.language = language
			
			#Save class file
			if divide_data == True:
				self.Load.save_file(self.class_array, "Model." + nickname + ".Callback.hdf.Classes")
			
			#If desired, pre-enrich samples with CxG vectors
			if type == "DID":
				if preannotate_cxg == True:
				
					self.Features = Features(self.Load, self.type, self.sample_size, language = self.language)
					self.n_features = self.Features.n_features
					file_list = self.train_files + self.test_files + self.eval_files
					
					print("\tPreannotating features.")
					#Multi-process
					pool_instance = mp.Pool(processes = cxg_workers, maxtasksperchild = None)
					pool_instance.map(partial(self.Features.save_cxg_vectors, 
												sample_size = did_sample_size, 
												workers = None, 
												language = self.language
												), file_list, chunksize = 1)
					pool_instance.close()
					pool_instance.join()

	#----------------------------------------------------------------------------------#
	
	def train(self,
				model_type = "MLP",				#(str): MLP or SVM
				lid_features = 524288,			#(int): Number of character n-gram features to allow, hashing only
				lid_ngrams = (1,3),				#(tuple of ints): Range of n-grams to hash
				did_grammar = ".Grammar.p",		#(str): Name of C2xG grammar to use for annotation; allows comparison of different grammars
				c2xg_workers = 1,				#(int): For DID, number of workers for c2xg enrichments
				mlp_sizes = (300, 300, 300),	#(tuple of ints): Size and number of layers; e.g., three layers with 300 neurons each
				cross_val = False,				#(boolean): Whether to evaluate using a CV approach
				dropout = 0.25,					#(float): The amount of dropout to apply to each layer
				activation = "relu",			#(str): The type of activation to use; just passes name to Keras
				optimizer = "sgd"				#(str): The type of optimization algorithm to use; just passes the name to Keras
				):
		
		self.workers = c2xg_workers
		
		print("\n\tInitializing feature extraction and input layers.\n")
		if self.type == "LID":
		
			#Initialize Features
			self.Features = Features(self.Load, self.type, self.sample_size)
		
			#Load character n-gram hashing features
			self.x_encoder = self.Features.get_x_lid(lid_features, lid_ngrams)
				
			#Define what the input shape looks like for hashing (vectors)
			inputs = Input(shape=(lid_features,), name = "input", dtype = "float32")
			
			#Set LID queue size
			q_size = 5
			self.epoch_size = self.n_per_epoch
			
		elif self.type == "DID":
				
			#Initialize Features
			self.Features = Features(self.Load, self.type, self.sample_size, language = self.language)
			self.n_features = self.Features.n_features
			
			#Load c2xg grammar vectors
			self.x_encoder = self.Features.get_x_did(c2xg_workers)
					
			#Define what the input shape looks like for hashing (vectors)
			inputs = Input(shape=(self.n_features,), name = "input", dtype = "float32")
			self.epoch_size = len(self.train_files)
			self.n_per_epoch = len(self.train_files)
			
			if self.preannotated_cxg == False:
				
				#Set DID queue size
				q_size = 5	
				self.n_per_epoch = 2
				
			else:
				print("\n\tUsing pre-enriched CxG vectors.")
				q_size = 50
	
		if model_type == "MLP":
			print("\n\tStarting MLP work.")
			
			#Try to load previous callback model
			model_filename = "Model." + self.nickname + ".Callback.hdf"
			if self.Load.check_file(model_filename):
				print("\n\tTrying to load existing callback model: " + model_filename)
				if self.s3 == True:
					self.Load.download_from_s3("./" + model_filename, self.output + "/" + model_filename)
				model = load_model(model_filename)
				if self.s3 == True:
					os.remove(model_filename)
				
			#Initializing Keras models for classification	
			else:
				print("\n\tInitializing MLP model in Keras.")
				model = self.build_mlp(inputs = inputs, 
										cat_output = self.n_classes, 
										mlp_sizes = mlp_sizes, 
										dropout = dropout, 
										activation = activation, 
										optimizer = optimizer
										)
									
			#Only fit the language model if a final version isn't saved
			final_filename = "Model." + self.nickname + "." + self.type + "." + model_type + "." + self.language + ".p"
			print("\tFinal Filename: " + final_filename + "\t\t")		
			
			#Define Keras callbacks
			callbacks = [EarlyStopping(monitor = "val_acc", mode = "max", min_delta = 0.0, patience = 50, verbose = 1),
						ModelCheckpoint(model_filename, monitor = "val_acc", mode = "max", save_best_only = True, period = 1, verbose = 1),
						CSVLogger(final_filename + ".log"),
						SaveCheckpoint(self.s3, model_filename, final_filename + ".log", self.Load)]
			
			if not self.Load.check_file(final_filename):
			
				model.fit_generator(generator = self.Features.generate_data(self.type, 
																				self.train_files, 
																				self.x_encoder, 
																				self.y_encoder, 
																				self.n_classes, 
																				self.sample_size, 
																				self.epoch_size, 
																				self.samples_per_epoch,
																				self.language, 
																				workers = c2xg_workers,
																				preannotated = self.preannotated_cxg), 
									steps_per_epoch = self.n_per_epoch,
									epochs = 500000,
									verbose = 1,
									callbacks = callbacks,
									validation_data = self.Features.generate_data(self.type, 
																					self.test_files, 
																					self.x_encoder, 
																					self.y_encoder, 
																					self.n_classes, 
																					self.sample_size, 
																					self.epoch_size, 
																					self.samples_per_epoch, 
																					self.language, 
																					workers = c2xg_workers,
																					preannotated = self.preannotated_cxg),
									validation_steps = int(len(self.test_files)),
									class_weight = None,
									max_queue_size = q_size,
									workers = 1,
									use_multiprocessing = False,
									initial_epoch = 0
									)
									
				#Save final language model
				model.save(final_filename)
				
				if self.s3 == True:
					self.Load.upload_to_s3(final_filename, self.output + "/" + final_filename)				
				
			#Finished language model already exists, so load it
			else:
				if self.s3 == True:
					self.Load.download_from_s3("./" + final_filename, self.output + "/" + final_filename)
				model = load_model(final_filename)

			#Final evaluation on held-out data
			self.evaluate_model(self.eval_files, model, self.y_encoder, n_classes = len(self.class_list))
	
		elif model_type == "SVM":
			print("\n\tStarting SVM work.")
			
			#Load development data
			X_dev, y_dev = self.Features.load_vectors(self.type, self.test_files, self.y_encoder, self.language, self.workers, self.preannotated_cxg, self.sample_size)
			print("\n\tBreakdown of development data:")
			self.data_description(y_dev)
			
			#Parameter search on test data
			cls = LinearSVC(max_iter = 100000000, class_weight = "balanced")
			tuned_parameters = [{"C": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10, 50, 100, 500, 1000, 10000]}]
			scores = ["f1"]
			chosen_parameters = self.evaluate_parameters(cls, tuned_parameters, scores, X_dev, y_dev)
	
			#Create instance of classifier with best parameters
			cls = LinearSVC(C = chosen_parameters["C"], max_iter = 100000000, class_weight = "balanced")
	
			#Evaluate feature normalization and selection
			normalize_check, chi_square_check = self.evaluate_features(cls, X_dev, y_dev)
			
			#Clear dev data and load training data
			print("\tNow loading training data")
			del X_dev
			del y_dev
			
			#Check if doing cross-validation or if training final model
			if cross_val == False:
						
				X_train, y_train = self.Features.load_vectors(self.type, self.train_files, self.y_encoder, self.language, self.workers, self.preannotated_cxg, self.sample_size)
				print("\n\tBreakdown of training data:")
				self.data_description(y_train)
				
				#Get best versions of corpus for final testing
				X_train = self.feature_prep(X_train, y_train, normalize_check, chi_square_check)
					
				print("\tFitting SVM classifier")
				cls.fit(X_train, y_train)
				print("\tFinished fitting SVM classifier.")
				
				#Clear training data and load evaluation data
				del X_train
				del y_train
				X_eval, y_eval = self.Features.load_vectors(self.type, self.eval_files, self.y_encoder, self.language, self.workers, self.preannotated_cxg, self.sample_size)
				print("\n\tBreakdown of evaluation data:")
				self.data_description(y_eval)
				
				#Normalize if required
				if normalize_check == "L1" or normalize_check == "L2":
					print("\tNormalizing eval data.")
					X_eval = self.feature_prep(X_eval, y_eval, normalize_check, chi_square_check = False)
				
				print("\tNormalization and Chi Square Feature Selection: ", end = " ")
				print(normalize_check, chi_square_check)	
				
				#Prune features is required
				if chi_square_check == True:
					print("\tChi square feature selection on eval data.")
					X_eval = csc_matrix(X_eval)
					X_eval = X_eval[:, self.significant_features]
					X_eval = coo_matrix(X_eval)
				
				#Evaluate on held-out testing data
				predicted = cls.predict(X_eval)
				
				#Generate report
				classifier_report = classification_report(y_eval, predicted)
				error_matrix = confusion_matrix(y_eval, predicted)
				self.report_results(classifier_report, error_matrix, self.class_list, domain_flag = False)
				
				print("Classes:")
				print(self.y_encoder.classes_)
				
				#Save trained classifier
				model = cls
				
				#Now save the model and all necessary support materials
				svm_model = {}
				svm_model["language"] = self.language
				svm_model["cls"] = model
				svm_model["normalize_check"] = normalize_check
				svm_model["chi_square_check"] = chi_square_check
				svm_model["y_encoder"] = self.y_encoder
				svm_model["grammar"] = self.Features.CxG.model
				svm_model["constructions"] = []
				svm_model["weights"] = model.coef_
				
				#Make readable constructions
				for x in self.Features.CxG.model:
					x = [y for y in x if y[0] != 0]
					x_readable = self.Features.CxG.Encode.decode_construction(x)
					svm_model["constructions"].append(x_readable)
				
				#Save feature mask
				if chi_square_check == True:
					svm_model["significant_features"] = self.significant_features
				
				#Save model
				final_filename = "Model." + self.nickname + "." + self.type + "." + model_type + "." + self.language + ".p"
				self.Load.save_file(svm_model, final_filename)			
			
			#Do alternate cross-validation tests
			elif cross_val == True:
				print("\n\tStarting cross-validation experiments; this does not save a final model.\n")
				cv_files = self.train_files + self.eval_files
				X_cv, y_cv = self.Features.load_vectors(self.type, cv_files, self.y_encoder, self.language, self.workers, self.preannotated_cxg, self.sample_size)
				
				print("\n\tBreakdown of training data:")
				self.data_description(y_cv)
				
				#Normalize if required
				print("\tNormalizing / features selection.")
				X_cv = self.feature_prep(X_cv, y_cv, normalize_check, chi_square_check)

				print("\tNormalization and Chi Square Feature Selection: ", end = " ")
				print(normalize_check, chi_square_check)	
				
				#Now do cross-eval
				cv_results = cross_validate(cls, X_cv, y_cv, scoring = "f1_weighted", return_train_score = False, cv = 5, n_jobs = c2xg_workers)
				for key in sorted(cv_results.keys()):
					print(key)
					print(cv_results[key])
					
				y_pred = cross_val_predict(cls, X_cv, y_cv, cv = 5, n_jobs = c2xg_workers)
				
				#Generate report
				classifier_report = classification_report(y_cv, y_pred)
				error_matrix = confusion_matrix(y_cv, y_pred)
				self.report_results(classifier_report, error_matrix, self.class_list, domain_flag = False)
				
				print("Classes:")
				print(self.y_encoder.classes_)
				model = None
		
		if model != None:
			return model
			s
	#--------------------------------------------------------------------------------------#
	
	def data_description(self, y_dev):
	
		freqs = ct.frequencies(y_dev)
		
		for i in range(len(self.y_encoder.classes_)):
			print("\t", end = "")
			print(self.y_encoder.classes_[i], freqs[i])

	#--------------------------------------------------------------------------------------#
		
	def get_y_encoder(self, class_array):
		
		#Initialize encoder
		encoder = LabelEncoder()
		encoder.fit(class_array)
		
		return encoder
		
	#--------------------------------------------------------------------------------------#

	def build_mlp(self, inputs, cat_output, mlp_sizes, dropout, activation, optimizer):

		layer_list = []
		
		for i in range(len(mlp_sizes)):
		
			#First layer takes inputs of dimension defined by feature space
			if i == 0:
				temp_layer = Dense(mlp_sizes[i], activation = activation)(inputs)
				temp_layer = Dropout(dropout)(temp_layer)
				layer_list.append(temp_layer)
			
			#All other layers take inputs from previous layer, which is the last in the list
			else:
				temp_layer = Dense(mlp_sizes[i], activation = activation)(layer_list[-1])
				temp_layer = Dropout(dropout)(temp_layer)
				layer_list.append(temp_layer)
			
		#Output dense layer with softmax activation
		pred = Dense(cat_output, activation = "softmax", name = "output")(layer_list[-1])

		#Inputs depend on whether we are adding to pre-trained model
		model = Model(inputs = inputs, outputs = pred)
		
		#Now compile
		model.compile(loss = "categorical_crossentropy", 
						optimizer = optimizer,
						metrics = ["accuracy"]
						)
						
		print("\n\n\tFinal state:")
		model.summary()

		return model

	#--------------------------------------------------------------------------------------------------#
	
	def evaluate_model(self, eval_files, model, y_encoder, n_classes):
	
		#Initialize holders for real / predicted classes
		all_true = []
		all_predict = []
	
		#Process each file
		for filename in eval_files:
			
			filename = filename[-1]

			try:
				#Load vectors
				filename = filename.split("/")
				filename[0] = filename[0] + "-vectors"
				filename[-1] = filename[-1].replace(".txt", ".p")
				filename = filename[0] + "/" + filename[3] + "/" + filename[1] + "/" + filename[2] + "/" + filename[-1]
				load_tuple = self.Load.load_file(filename, fix = True)
				X_vector = load_tuple[0]
				y_vector = load_tuple[1]
							
				#Make dense
				X_vector = X_vector.todense()
				instances = X_vector.shape[0]
						
				#Tranform y
				y_true = y_vector.tolist()				
				
				#Get prediction layer
				y_predict = model.predict(X_vector, batch_size = instances)

				#Convert to discrete output
				y_classes = np.argmax(y_predict, axis = 1)
				y_classes = [self.class_list[x] for x in y_classes]
								
				#print(y_classes)
				#print(y_true)
					
				#Add to total prediction record
				all_true += y_true
				all_predict += y_classes
					
			except Exception as e:
				print(e)
				print("Error: " + filename)
				continue
		
		#Done going through files
		classifier_report = classification_report(all_true, all_predict)
		error_matrix = confusion_matrix(all_true, all_predict)
		self.report_results(classifier_report, error_matrix, self.class_list, domain_flag = False)
		
	#----------------------------------------------------------------------------------------#
	
	def report_results(self, test_report, test_matrix, class_dict, domain_flag):

		print("")
		print("\t-----------------------")
		print("\t---CLASSIFIER REPORT---")
		print("\t-----------------------")
		print(test_report)
		print("")	
		
		# Print confusion matrix in readable format#
		if self.language != "eng":
			print("")
			print("\t----------------------")
			print("\t---CONFUSION MATRIX---")
			print("\t----------------------")
			print("")
			
			#Print values from confusion matrix#
			for i in range(test_matrix.shape[0]):
				
				row_total = 0
				lang1 = class_dict[i]
				
				for j in range(test_matrix.shape[0]):
				
					lang2 = class_dict[j]
					
					if lang1 != lang2:
						errors = test_matrix[i,j]
						row_total += errors

						if errors != 0:
							print(lang1 + ":" + lang2 + "," + str(errors))
	
		return

	#---------------------------------------------------------------------------------#
	
	def evaluate_parameters(self, cls, tuned_parameters, scores, X_train, y_train):

		for score in scores:
			
			print("\n\t# Tuning hyper-parameters for %s" % score)

			clf = GridSearchCV(cls, tuned_parameters, cv = 5, scoring = "%s_weighted" % score, n_jobs = self.workers)
			clf.fit(X_train, y_train)

			print("\tBest parameters set found on development set:")
			print("\t" + str(clf.best_params_))
			chosen_parameters = clf.best_params_
					  
		return chosen_parameters
	#---------------------------------------------------------------------------------#

	def evaluate_features(self, cls, X_test, y_test):

		result_dict = {}
		
		predicted = model_selection.cross_val_predict(cls, X_test, y_test, cv = 10, n_jobs = self.workers)
		result_dict["no_normalization"] = f1_score(y_test, predicted, average = "weighted")
		
		print("\n\tF1 without normalization: " + str(result_dict["no_normalization"]))
		
		for norm_type in ["L1", "L2"]:
			
			test_vector_array = self.feature_prep(X_test, y_test, normalize_check = norm_type, chi_square_check = False)
			predicted = model_selection.cross_val_predict(cls, test_vector_array, y_test, cv = 10, n_jobs = self.workers)
			result_dict[norm_type] = f1_score(y_test, predicted, average = "weighted")
			print("\tF1 with " + str(norm_type) + " normalization: " + str(result_dict[norm_type]))
		
		print("")
		#Now get best normalized set for feature selection test#
		top = max(list(result_dict.values()))
		
		if result_dict["no_normalization"] == top:
			normalize_check = None
			print("\tBest without normalization.")
		
		elif result_dict["L1"] == top:
			normalize_check = "L1"
			print("\tBest with L1 normalization.")
		
		elif result_dict["L2"] == top:
			normalize_check = "L2"
			print("\tBest with L2 normalization.")
			
		#Get best feature vector type with chi square pruning#
		test_vector_array = self.feature_prep(X_test, y_test, normalize_check = normalize_check, chi_square_check = True)
		predicted = model_selection.cross_val_predict(cls, test_vector_array, y_test, cv = 10, n_jobs = self.workers)
		pruning_result = f1_score(y_test, predicted, average = "weighted")
		
		print("\tF1 with " + str(normalize_check) + " normalization and chi square feature selection: " + str(pruning_result))
		
		if pruning_result > top:
			chi_square_check = True
		else:
			chi_square_check = False
			
		return normalize_check, chi_square_check

	#------------------------------------------------------------------------------------#

	def feature_prep(self, vector_array, class_array, normalize_check, chi_square_check, pearson_r_check = False):

		warnings.filterwarnings("ignore")
		significant_features = None
		
		#Normalization first, if doing it#
		if normalize_check == "L1":
			#print("\tPerforming L1 feature normalization.")
			normalizer = Normalizer(norm = "l1", copy = False)
			normalize_pipeline = make_pipeline(normalizer)
			vector_array = normalize_pipeline.fit_transform(vector_array)
		
		elif normalize_check == "L2":
			#print("\tPerforming L2 feature normalization.")
			normalizer = Normalizer(norm = "l2", copy = False)
			normalize_pipeline = make_pipeline(normalizer)
			vector_array = normalize_pipeline.fit_transform(vector_array)
		
		if chi_square_check == True:
		
			print("")
			#print("\tBeginning chi-square feature selection.")
			vector_array, significant_features = self.run_chi_square(vector_array,
														class_array,
														significance_level = 0.05
														)

			self.significant_features = significant_features

		if pearson_r_check == True:
			print("")
			print("\tBeginning pearson R feature pruning.")
			vector_array = self.run_pearson_prune(vector_array, 
							class_array,
							self.workers,
							significance_level = 0.05,
							cor_level = 0.90)
		
	
		
		return vector_array
	
	#--------------------------------------------------------------------------------------------#

	def run_chi_square(self, vector_array, class_array, significance_level):

		chi_stat, pval = chi2(vector_array, class_array)
		significant_features = pval[np.where( pval < significance_level )]
		
		print("")
		print("\tNumber of features significantly different (" + str(significance_level) + ") across classes: " + str(significant_features.shape[0]) + " out of " + str(vector_array.shape[1]))
		print("")
		
		column_list = [x for x in range(0,len(pval)) if pval[x] < significance_level]
		
		vector_array = csc_matrix(vector_array)
		vector_array = vector_array[:, column_list]
		vector_array = coo_matrix(vector_array)
			
		return vector_array, column_list
		
	#--------------------------------------------------------------------------------------------#
	
	def run_pearson_prune(self, vector_array, class_array, significance_level, cor_level):

		time_start = time.time()
		
		print("Multi-processing Pearson's R feature pruning:")
		
		#Multi-process Pearson pruning#
		pool_instance=mp.Pool(processes = self.workers, maxtasksperchild = 1)
		remove_list = pool_instance.map(partial(self.process_pearson_prune, 
													vector_array = vector_array, 
													significance_level = significance_level,
													cor_level = cor_level,
													max = vector_array.shape[1]
													), [i for i in range(0, vector_array.shape[1])], chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		
		remove_list = list(ct.concat(remove_list))
		remove_list = list(set(remove_list))

		vector_array = vector_array[:, [x for x in range(0, vector_array.shape[1]) if x not in remove_list]]	
		
		print("")
		print("Features above correlation threshold (" + str(cor_level) + "): " + str(len(remove_list)))
		print("Time for completion: " + str((float(time.time())) - time_start))
		print("Finished with Pearson R Feature Pruning.")
		print("")
		
		return vector_array
		
	#--------------------------------------------------------------------------------------------#
	
	def process_pearson_prune(self, i, vector_array, significance_level, cor_level, max):

		remove_list = []
		first_column = vector_array[:, i].todense()
		
		if i % 100 == 0:
			print(i, end="")
			print(" out of ", end="")
			print(max)
		
		#For each column, get pearson R with all later columns#
		for j in range(i, vector_array.shape[1]):
				
			if i < j:
				
				second_column = vector_array[:, j].todense()
					
				result = pearsonr(first_column, second_column)
		
				current_r = result[0]
				current_p = result[1]
					
				#If correlation above threshold and significance above threshold#
				if current_r > cor_level and current_p < significance_level:

					if np.count_nonzero(first_column) >= np.count_nonzero(second_column):
						remove_list.append(j)
							
					elif np.count_nonzero(second_column) > np.count_nonzero(first_column):
						remove_list.append(i)

		return remove_list
		
	#--------------------------------------------------------------------------------------------#