import os
import pickle
import codecs
import boto3
import cytoolz as ct
from random import randint

#The loader object handles all file access to enable local or S3 bucket support
class Loader(object):

	def __init__(self, input, output, s3 = False, s3_bucket = ""):
	
		#if using S3, input and output dirs are prefixes
		self.input_dir = input
		self.output_dir = output
		self.s3 = s3
		self.s3_bucket = s3_bucket
		
		#Check that directories exist
		if s3 == False:
			
			if os.path.isdir(self.input_dir) == False:
				os.makedirs(self.input_dir)
				print("Creating input folder")
			
			if os.path.isdir(self.output_dir) == False:
				os.makedirs(self.output_dir)
				print("Creating output folder")
			
	#---------------------------------------------------------------#
	
	def upload_to_s3(self, local_location, s3_location, delete_flag = True):
	
		#Initialize boto3 client
		import boto3
		client = boto3.client("s3")
		
		#Upload and delete
		client.upload_file(local_location, self.s3_bucket, s3_location)
		
		if delete_flag == True:
			os.remove(local_location)
	
	#---------------------------------------------------------------#
	
	def download_from_s3(self, local_location, s3_location):
	
		#Initialize boto3 client
		import boto3
		client = boto3.client("s3")
		
		#Dowload to local location
		client.download_file(self.s3_bucket, s3_location, local_location)
	
	#---------------------------------------------------------------#
		
	def save_file(self, file, filename, different_out = ""):
	
		#Redirect save location if necessary
		if different_out != "":
			output_dir = different_out
		elif different_out == "":
			output_dir = self.output_dir
		#---------------------------------
		
		if self.s3 == True:
			print("\t\tSaving " + filename + " to S3 as " + str(output_dir + "/" + filename))
			
			#Initialize boto3 client
			import boto3
			client = boto3.client("s3")
		
			#Write file to disk
			temp_name = "temp." + str(randint(1,10000000)) + ".p"	
			with open(os.path.join(temp_name), "wb") as handle:
				pickle.dump(file, handle, protocol = pickle.HIGHEST_PROTOCOL)
				
			#Upload and delete
			client.upload_file(temp_name, self.s3_bucket, output_dir + "/" + filename)
			os.remove(temp_name)
		
		else:
		
			#Write file to disk
			with open(os.path.join(output_dir, filename), "wb") as handle:
				pickle.dump(file, handle, protocol = pickle.HIGHEST_PROTOCOL)
				
	#---------------------------------------------------------------#
	
	def list_input(self):
	
		files = []							#Initiate list of files
		prefixes = [self.input_dir + "/"]	#Initiate list of sub-prefixes to check for s3
		ignore_list = []
		
		#If listing an S3 bucket
		if self.s3 == True:
		
			#Initialize boto3 client
			import boto3
			client = boto3.client("s3")
			
			#Recursive over folder hierarchy
			while True:
			
				#Assign current addition
				current_prefix = prefixes.pop()

				if current_prefix not in ignore_list:
					ignore_list.append(current_prefix)
					con_token = ""
					
					#Recursive over truncated pages
					while True:
												
						#Find all files in directory
						if con_token != "":
							
							response = client.list_objects_v2(
											Bucket = self.s3_bucket,
											Delimiter = "/",
											Prefix = current_prefix,
											ContinuationToken = con_token
											)
											
						else:
							response = client.list_objects_v2(
											Bucket = self.s3_bucket,
											Delimiter = "/",
											Prefix = current_prefix
											)
						
						if "Contents" in response:
							if len(response["Contents"]) >= 1:
								for key in response["Contents"]:
									files.append(key["Key"])
					
						if "CommonPrefixes" in response:
							if len(response["CommonPrefixes"]) >= 1:
								for key in response["CommonPrefixes"]:
									if key["Prefix"] not in ignore_list:
										prefixes.append(key["Prefix"])
										
						if response["IsTruncated"] == True:
							con_token = response["NextContinuationToken"]
							
						elif response["IsTruncated"] == False:
							break

				#Test for breaking folder recursion
				if len(prefixes) == 0:
					break
			
		#If reading local file	
		else:
			for root, dirs, filenames in os.walk(self.input_dir):
				for filename in filenames:
					files.append(os.path.join(root, filename))
				
		return [x for x in files if ".txt" in x]
			
	#---------------------------------------------------------------#
	
	def list_output(self, prefix = "", type = ""):
	
		files = []	#Initiate list of files
		
		if prefix == "":
			prefix = self.output_dir
		
		#If listing an S3 bucket
		if self.s3 == True:
		
			#Initialize boto3 client
			client = boto3.client("s3")
				
			#Find all files in directory
			response = client.list_objects_v2(
							Bucket = self.s3_bucket,
							Delimiter = "/",
							Prefix = prefix + "/"
							)
			try:
				for key in response["Contents"]:
					files.append(key["Key"])
				
				new_files = []
				for file in files:
					file = file.split("/")
					file = file[-1]
					new_files.append(file)
				
				files = new_files
				
				if type != "":
					files = [file for file in files if type in file]
			except:
				files = []
				
		#If reading local file	
		else:
		
			for filename in os.listdir(self.output_dir):
				if type in filename:
					files.append(filename)
				
		return files
			
	#---------------------------------------------------------------#
	
	def check_file(self, filename, prefix = ""):

		file_list = self.list_output(prefix = prefix)

		if filename in file_list:
			return True
			
		else:
			return False
	#--------------------------------------------------------------#
	
	def load_file(self, filename, fix = False):
	
		if fix == False:
			output_dir = self.output_dir
		
		else:
			holder = filename.split("/")
			filename = holder.pop()
			output_dir = "/".join(holder)
	
		if self.s3 == True:
			
			filename = output_dir + "/" + filename
			#Initialize boto3 client
			import boto3
			client = boto3.client("s3")

			#Find all files in directory
			response = client.list_objects_v2(
							Bucket = self.s3_bucket,
							Delimiter = "/",
							Prefix = output_dir + "/"
							)
		
			files = []
			for key in response["Contents"]:
				files.append(key["Key"])
			
			#Check for file specified
			if filename in files:	
			
				#Download, load and return
				temp_name = "temp." + str(randint(1,10000000)) + ".p"	#Have to avoid conflicts across cores
				client.download_file(self.s3_bucket, filename, temp_name)
				
				with open(temp_name, "rb") as handle:
					return_file = pickle.load(handle)
					
				os.remove(temp_name)
				
				return return_file
			
			#If file isn't found in the S3 bucket, return error
			else:
				print(filename + " not found")

		#If reading local file	
		else:
		
			with open(os.path.join(output_dir, filename), "rb") as handle:
					return_file = pickle.load(handle)
				
			return return_file
			
	#---------------------------------------------------------------#
	
	def read_file(self, file):
	
		#Read from S3 bucket
		if self.s3 == True:
			
			#Initialize boto3 client
			import boto3
			client = boto3.client("s3")
			
			temp_name = "temp." + str(randint(1,10000000)) + ".txt"	#Have to avoid conflicts across cores
			client.download_file(self.s3_bucket, file, temp_name)
				
			with codecs.open(temp_name, "rb") as fo:
				lines = fo.readlines()
					
			os.remove(temp_name)
				
			for line in lines:
				line = line.decode("utf-8")
				yield line
				
		#Read local directory
		else:

			with codecs.open(file, "rb") as fo:
				lines = fo.readlines()
					
			for line in lines:
				line = line.decode("utf-8", errors = "replace")
				yield line
	
	#---------------------------------------------------------------#