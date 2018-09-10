import re
import os
import cytoolz as ct
import numpy as np
import multiprocessing as mp
from random import shuffle
from functools import partial
from gensim.parsing import preprocessing
from sklearn.feature_extraction.text import HashingVectorizer
from keras.utils import np_utils
from scipy.sparse import coo_matrix
from scipy.sparse import vstack

#----------------------------------------------------------------------------------#
#Outside of class to aid distribution across processes ----------------------------#
#----------------------------------------------------------------------------------#

def process_did(meta_tuple, sample_split, sample_size, Load):
	
	#Unpack class and filename
	country = meta_tuple[0]
	filename = meta_tuple[1]
		
	#Initiate lists
	line_list = []
	country_list = []
	
	fo = [line for line in Load.read_file(filename)]
	shuffle(fo)
								
	#If not joining samples, save lines and meta-data
	if sample_split == False:
		line_list += fo
		[country_list.append(country) for x in fo]
															
	#Join lines to make longer samples
	elif sample_split == True:
									
		#Initialize resources
		join_counter = 0
		line_str = ""
									
		#Iterate over lines
		for line in fo:
			if len(line) > 1:
					
				#Join lines
				join_counter += 1
				line = line.strip()
				line_str += line
											
				#Check if have had enough
				if join_counter >= sample_size:
					line_list.append(line_str)
					country_list.append(country)
					line_str = ""
					join_counter = 0
						
	return line_list, country_list
		
#----------------------------------------------------------------------------------#

def process_svm_load(pair, Load, y_encoder, sample_size):

	#print("\t\tLoading " + str(pair))
	
	lang = pair[0]
	region = pair[1]
	country = pair[2]
	filename = pair[3]
	class_name = pair[-1]

	try:
		#Load vectors
		filename = filename.split("/")
		filename[0] = filename[0] + "_vectors_" + str(sample_size)
		filename[-1] = filename[-1].replace(".txt", ".p")
		filename = filename[0] + "/" + filename[3] + "/" + filename[1] + "/" + filename[2] + "/" + filename[4]

		load_tuple = Load.load_file(filename, fix = True)
		X_vector = load_tuple[0]
		y_vector = load_tuple[1]
		y_vector = np.array([class_name for x in y_vector])
								
		#Tranform y
		y_vector = y_encoder.transform(y_vector)
	
		return X_vector, y_vector					
	
	except Exception as e:
		print(e)
		print(filename)
		
#----------------------------------------------------------------------------------#

class Features(object):

	def __init__(self, Loader, type, sample_size, language = ""):
	
		self.Load = Loader
		self.type = type
		self.sample_size = sample_size
		
		try:
		# Wide UCS-4 build
			self.myre = re.compile(u'['
				u'\U0001F300-\U0001F64F'
				u'\U0001F680-\U0001F6FF'
				u'\u2600-\u26FF\u2700-\u27BF]+', 
				re.UNICODE)
		except re.error:
		# Narrow UCS-2 build
				self.myre = re.compile(u'('
				u'\ud83c[\udf00-\udfff]|'
				u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
				u'[\u2600-\u26FF\u2700-\u27BF])+', 
				re.UNICODE)
				
		if type == "DID":
			
			#Import grammar vectorizing package
			import os
			import sys
			
			#Navigate to c2xg
			os.chdir(os.path.join("..", "c2xg"))
			sys.path.append(os.path.abspath("."))
			import c2xg
			from c2xg.c2xg import C2xG

			#Return to idNet directory
			os.chdir(os.path.join("..", "idNet"))
			
			#Initialize parser and set number of dimensions
			if self.Load != False:
				self.CxG = C2xG(data_dir = self.Load.input_dir, language = language, s3 = self.Load.s3, s3_bucket = self.Load.s3_bucket, zho_split = True)
				self.n_features = self.CxG.n_features
			
			else:
				self.CxG = C2xG(data_dir = "", language = language, s3 = "", s3_bucket = "", zho_split = True)

	#---------------------------------------------------------------#

	def x_hashing_pre(self, line):

		#Remove links, hashtags, at-mentions, mark-up, and "RT"
		line = re.sub(r"http\S+", "", line)
		line = re.sub(r"@\S+", "", line)
		line = re.sub(r"#\S+", "", line)
		line = re.sub("<[^>]*>", "", line)
		line = line.replace(" RT", "").replace("RT ", "")
		
		#Remove emojis
		line = re.sub(self.myre, "", line)
		
		#Remove punctuation and extra spaces
		line = ct.pipe(line, preprocessing.strip_tags, preprocessing.strip_punctuation, preprocessing.strip_numeric, preprocessing.strip_non_alphanum, preprocessing.strip_multiple_whitespaces)
		
		#Strip and reduce to max training length
		line = line.lower().strip().lstrip()
		
		#Truncate sampels for LID
		if self.type == "LID":
			line = line[0:self.sample_size]
		
		return line
	#--------------------------------------------------------------------------------------#

	def get_x_lid(self, n_features, n_gram_tuple):

		hashing_partial = partial(self.x_hashing_pre)
		
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
	
	#--------------------------------------------------------------------------------------#
	
	def get_x_did(self, workers):
	
		#Just a place holder
		
		return ""
	
	#--------------------------------------------------------------------------------------#
	
	def generate_data(self, type, file_list, x_encoder, y_encoder, n_classes, sample_size, epoch_size = 1, samples_per_epoch = 5, language = "", workers = 1, preannotated = False):
		
		n_samples = samples_per_epoch #How many files to use per country/language or domain/language pair per epoch
				
		#Feed training / testing data to Keras
		if type == "LID":
			
			#Determine whether to split LID samples to make more of them
			sample_split = False
			if sample_size < 100:
				sample_split = True
			
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
							fo = [line for line in self.Load.read_file(filename)]
								
							if sample_split == False:
								X_vector = x_encoder(fo)
								
							elif sample_split == True:
								line_list = []
								for line in fo:
									line_list.append(line[:100])
									line_list.append(line[100:])
								X_vector = x_encoder(line_list)
							
							#Get class vector
							instances = X_vector.shape[0]
							X_vector = X_vector.toarray()
							y_temp = [lang for x in range(instances)]				
							y_vector = np.array(y_temp)
							y_vector = y_encoder.transform(y_vector)
							y_vector = np_utils.to_categorical(y_vector, num_classes = n_classes)
							
							yield X_vector, y_vector
							
						except Exception as e:
							print("Error: ", end = " ")
							print(e)
							print(filename)
							print(pair)
							print("")
							
		elif type == "DID" and preannotated == False:
			
			#Determine whether to split LID samples to make more of them
			sample_split = False
			if sample_size > 1:
				sample_split = True
							
			process_list = []
			
			while True:
		
				#Iterate over files for this loop
				for pair in file_list:
					
					lang = pair[0]
					region = pair[1]
					country = pair[2]
					filename = pair[3]
					
					#Check if sample is the right language
					if lang == language:
						
						process_list.append((country, filename))
					
				#If accumulated enough samples, analyze all at once
							
				#Multi-process file loading
				pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
				result_list = pool_instance.map(partial(process_did, sample_split = sample_split, sample_size = sample_size, Load = self.Load), process_list, chunksize = 10)
				pool_instance.close()
				pool_instance.join()
							
				#Unpack results
				line_list = []
				country_list = []
				for line, country in result_list:
					line_list.append(line)
					country_list.append(country)
							
				#Flatten
				line_list = list(ct.concat(line_list))
				country_list = list(ct.concat(country_list))
								
				#Clean
				del result_list

				#Now analyze the joined samples
				X_vector = self.CxG.parse_return(line_list, mode = "idNet", workers = workers)
				X_vector = np.vstack(X_vector)
					
				#Get class vector
				y_vector = np.array(country_list)
				y_vector = y_encoder.transform(y_vector)
				y_vector = np_utils.to_categorical(y_vector, num_classes = n_classes)
					
				yield X_vector, y_vector
							
		elif type == "DID" and preannotated == True:
			
			while True:
		
				#Randomize files for this loop
				shuffle(file_list)
				
				#Reset line_list and country_list
				line_list = []	
				country_list = []
				process_list = []
					
				#Iterate over files for this loop
				for pair in file_list:
					
					lang = pair[0]
					region = pair[1]
					country = pair[2]
					filename = pair[3]
					class_name = pair[-1]
					
					#Check if sample is the right language
					if lang == language:
						
						try:
							#Load vectors
							filename = filename.split("/")
							filename[0] = filename[0] + "_vectors_" + str(sample_size)
							filename[-1] = filename[-1].replace(".txt", ".p")
							filename = filename[0] + "/" + filename[3] + "/" + filename[1] + "/" + filename[2] + "/" + filename[4]
							load_tuple = self.Load.load_file(filename, fix = True)
							X_vector = load_tuple[0]
							y_vector = load_tuple[1]
							y_vector = np.array([class_name for x in y_vector])
							
							#Make dense
							X_vector = X_vector.todense()
						
							#Tranform y
							y_vector = y_encoder.transform(y_vector)
							y_vector = np_utils.to_categorical(y_vector, num_classes = n_classes)
							
							yield X_vector, y_vector
							
						except Exception as e:
							print(e)
							print(filename)
							
	#----------------------------------------------------------------------------------#
	
	def load_vectors(self, type, file_list, y_encoder, language = "", workers = 1, preannotated = False, sample_size = 1):
		
		if type != "DID" and preannotated != True:
			print("\n\nError: Loading vectors into memory only works for preannotated CxG data.")
		
		elif type == "DID" and preannotated == True:
		
			X_list = []
			y_list = []
			
			#Randomize files for this loop
			shuffle(file_list)
			
			#Reset line_list and country_list
			line_list = []	
			country_list = []
			process_list = []
			
			#Reduce to right language
			file_list = [x for x in file_list if x[0] == language]

			#Multi-process file loading
			pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
			result_list = pool_instance.map(partial(process_svm_load, 
														Load = self.Load, 
														y_encoder = y_encoder,
														sample_size = sample_size
														), file_list, chunksize = 5)
			pool_instance.close()
			pool_instance.join()	

			for thing in result_list:
				try:
					X = thing[0]
					y = thing[1]
					X_list.append(X)
					y_list.append(y)
				except Exception as e:
					print(e)
			
			del result_list
			
			#Now merge into single array
			X_vector = vstack(X_list)
			y_vector = np.hstack(y_list)

			return X_vector, y_vector					

	#----------------------------------------------------------------------------------#
	
	def save_cxg_vectors(self, pair, sample_size, workers, language):
	
		#Determine whether to split LID samples to make more of them
		sample_split = False
		if sample_size > 1:
			sample_split = True
		
		#Get meta data
		lang = pair[0]
		region = pair[1]
		country = pair[2]
		filename = pair[3]
		actual_filename = filename.split("/")
		actual_filename = actual_filename[-1]
		actual_filename = actual_filename.replace(".txt", ".p")
		
		#Get s3 path
		vector_dir = "_vectors_" + str(sample_size)
		different_out = self.Load.input_dir + vector_dir
		different_out = os.path.join(different_out, lang, region, country)
					
		#Check if sample is the right language
		if lang == language:

			#Check if file has already been saved
			if self.Load.check_file(actual_filename, prefix = different_out) == True:
				print("\tFile already exists!")
			
			else:

				try:
					print("\t Starting " + str(pair))
					meta_tuple = ((country, filename))
					line_list, country_list = process_did(meta_tuple, sample_split, sample_size, self.Load)
										
					#Now analyze the joined samples
					X_vector = self.CxG.parse_return(line_list, mode = "idNet", workers = workers)
					X_vector = np.vstack(X_vector)
					X_vector = coo_matrix(X_vector)

					#Get class vector
					y_vector = np.array(country_list)
							
					#Save to file
					save_tuple = (X_vector, y_vector)
					self.Load.save_file(save_tuple, actual_filename, different_out = different_out)
					
				except Exception as e:
					print(e)
					print(filename)
			