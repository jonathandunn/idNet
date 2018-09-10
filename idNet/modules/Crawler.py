from collections import defaultdict
from random import shuffle
import cytoolz as ct

class Crawler(object):

	#------------------------------------------------------#
	
	def __init__(self, Loader, type, language = ""):
	
		self.Load = Loader
		self.type = type
		self.language = language
		
	#------------------------------------------------------#
		
	def crawl(self, n_samples, threshold, remove_list = [], merge_dict = {}):

		#Dataset is organized in folders:
		#--- For LID: Root / Domain / Language
		#--- For DID: Root / Region / Country / Language
		#--- This crawls that dataset to create the training / testing / validating data
		test_files = []
		train_files = []
		develop_files = []

		file_list = self.Load.list_input()
		print("\tProcessing " + str(len(file_list)) + " files.")
		
		#Now assign to training / testing / eval sets
		test_dict  = defaultdict(int)
		develop_dict = defaultdict(int)		
		
		for file in file_list:
		
			if "/" in file:
				meta = file.replace(self.Load.input_dir, "").split("/")
			else:
				meta = file.replace(self.Load.input_dir, "").split("\\")
				
			#Process LID data
			if self.type == "LID":
			
				
				domain = meta[1]
				lang = meta[2]
				
				#Check and add test files
				if test_dict[(lang, domain)] < n_samples:
					test_files.append((lang, domain, file))
					test_dict[(lang, domain)] += 1
					
				elif develop_dict[(lang, domain)] < n_samples:
					develop_files.append((lang, domain, file))
					develop_dict[(lang, domain)] += 1
					
				else:
					train_files.append((lang, domain, file))
				
			#Process DID data
			elif self.type == "DID":
				
				region = meta[1]
				country = meta[2]
				lang = meta[3]
				
				if country in merge_dict:
					class_name = merge_dict[country]
				else:
					class_name = country
				
				#Check and add test files
				if test_dict[(lang, region, country)] < n_samples:
					test_files.append((lang, region, country, file, class_name))
					test_dict[(lang, region, country)] += 1
					
				elif develop_dict[(lang, region, country)] < n_samples:
					develop_files.append((lang, region, country, file, class_name))
					develop_dict[(lang, region, country)] += 1
					
				else:
					train_files.append((lang, region, country, file, class_name))
			
		#Reduce files by threshold
		test_files, develop_files, train_files, class_list = self.prep_work(test_files, develop_files, train_files, n_samples)
		
		return test_files, develop_files, train_files, class_list
	
	#------------------------------------------------------------#

	def prep_work(self, test_files, develop_files, train_files, threshold):
	
		#Reduce DID to language-specific samples
		if self.type == "DID": 
			test_files = [x for x in test_files if x[0] == self.language]
			develop_files = [x for x in develop_files if x[0] == self.language]
			train_files = [x for x in train_files if x[0] == self.language]
			
			#Filter by number of samples
			country_list = [x[-1] for x in train_files]
			starting = len(set(country_list))
			country_dict = ct.frequencies(country_list)
			country_threshold = lambda x: x  >= threshold
			country_dict = ct.valfilter(country_threshold, country_dict)
			country_list = list(country_dict.keys())
			print("\t\tReducing initial set of " + str(starting) + " countries to " + str(len(country_list)) + " after frequency threshold.")
			
			#Prune and shuffle file lists
			test_files = [x for x in test_files if x[-1] in country_list]
			shuffle(test_files)
			
			train_files = [x for x in train_files if x[-1] in country_list]
			shuffle(train_files)
			
			develop_files = [x for x in develop_files if x[-1] in country_list]
			shuffle(develop_files)
			
			return test_files, develop_files, train_files, country_list
		
		elif self.type == "LID":
			#Filter by number of samples
			lang_list = [x[0] for x in train_files]
			starting = len(set(lang_list))
			lang_dict = ct.frequencies(lang_list)
			lang_threshold = lambda x: x  >= threshold
			lang_dict = ct.valfilter(lang_threshold, lang_dict)
			lang_list = list(lang_dict.keys())
			print("\t\tReducing initial set of " + str(starting) + " languages to " + str(len(lang_list)) + " after frequency threshold.")
			
			#Prune and shuffle file lists
			test_files = [x for x in test_files if x[0] in lang_list]
			shuffle(test_files)
			
			train_files = [x for x in train_files if x[0] in lang_list]
			shuffle(train_files)
			
			develop_files = [x for x in develop_files if x[0] in lang_list]
			shuffle(develop_files)
		
			return test_files, develop_files, train_files, lang_list
	
	#----------------------------------------------------------#

	def get_n_epoch(self, file_list, n_samples, type):
				
		#Figure out how many samples to use per epoch
		#-- The goal is to sample each lang/domain pair n times
		if type == "LID":
			
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
					
		#Figure out how many samples to use per epoch
		#-- The goal is to sample each lang/domain pair n times
		elif type == "DID":
			
			counter = 0
			country_dict = {}
				
			#Iterate over language files for this loop
			for pair in file_list:
				
				lang = pair[0]
				region = pair[1]
				country = pair[2]
				filename = pair[3]
					
				#Check samples from current lang
				if country in country_dict:
					country_count = country_dict[country]
												
					#Add another language-domain sample
					if country_count < n_samples:
						grab_it = 1
						country_dict[country] += 1
								
					#Domain and language already sufficiently sampled
					else:
						grab_it = 0
								
					#New domain, add it
				else:
					country_dict[country] = 1
					grab_it = 1
					
				#If this language has not already been oversampled
				if grab_it == 1:
					counter += 1			
				
		return counter