import time
import os
import codecs
import warnings
import zipfile
import pandas as pd
import numpy as np
import boto3
import lzma
import shutil
import multiprocessing as mp
from functools import partial
from random import randint

def format_corpus(s3_bucket, read_prefix, write_prefix, lines_per_file):
	
	print("\nSplitting and formatting dataset.")
	
	#Get zip files
	client = boto3.client("s3")
	paginator = client.get_paginator("list_objects_v2")
	operation_parameters = {"Bucket": s3_bucket,
							"Prefix": read_prefix}

	page_iterator = paginator.paginate(**operation_parameters)
	zip_list = []
		
	for page in page_iterator:

		for key in page["Contents"]:
			filename = key["Key"]
			zip_list.append(filename)

	#Iterate, unzip, upload
	for zip_file in zip_list:
			
		print("\tStarting: " + zip_file)
		
		#Make sure this is the right type of file
		if zip_file.endswith(".7z"):
			client.download_file(s3_bucket, zip_file, "temp.7z")
			os.system("7za x temp.7z -o'./read'")
			os.remove("temp.7z")
			go_flag = 1
		
		else:
			print("\tError: Expected .7z files.")
			go_flag = 0
			
		if go_flag == 1:
		
			#Open zip file and iterate over its contents
			for root, dirs, files in os.walk("./read"):
				for name in files:
						
					if name.endswith(".txt"):
						meta = name.split("/")
						meta = meta[-1]
						meta = meta.split(".")
						region = meta[1]
						country = meta[2]
						language = meta[3]

						name = os.path.join(root, name)
							
						#Format the current language
						with codecs.open(name, "r", encoding = "utf-8") as f:
								
							doc_counter = 1
							line_counter = 0
							total_counter = 0
							char_counter = 0
							
							#Make temp dir
							if os.path.isdir(os.path.join(".", "temp")) == False:
								os.makedirs(os.path.join(".", "temp"))
								
							write_file = os.path.join(".", "temp", region + "." + country + "." + language + "." + str(doc_counter) + ".txt")
							fw = codecs.open(write_file, "w", encoding = "utf-8")
								
							#Iterate over lines in the raw file
							for line in f:
								
								line_counter += 1
								line = line.strip()

								#If this line ends the file, start a new file
								if line_counter > lines_per_file - 1:
									fw.close()
									line_counter = 0
									total_counter += 1
									doc_counter += 1
									write_file = os.path.join(".", "temp", region + "." + country + "." + language + "." + str(doc_counter) + ".txt")
									fw = codecs.open(write_file, "w", encoding = "utf-8")
										
								#Keep writing
								fw.write(line + "\n")

							#Finished with this domain-language file, print stats
							fw.close()
							print("\t\t", end = "")
							print(region, country, language, total_counter)
							os.remove(name)

							#Now iterate over temp directory and upload files
							s3_path = os.path.join(write_prefix, region, country, language)
							for file in os.listdir(os.path.join(".", "temp")):
							
								client.upload_file(os.path.join(".", "temp", file), s3_bucket, os.path.join(s3_path, file))
								os.remove(os.path.join(".", "temp", file))
								
			#Now remove empty folder and the zip
			os.rmdir(os.path.join(".", "temp"))
			shutil.rmtree("./read", ignore_errors = True)
								
	return 

#--------------------------------------------------------------------#
print("\nFormatting LID training data from Data_Raw to Data_Formatted.\n")
items = [x for x in range(1,100000)]

#Get lines_per_file
choice = input("\nHow many samples per files? ")
if int(choice) in items:
	lines_per_file = int(choice)
else:
	print("\nInvalid value. Use integer between 1 and 100,000.\n")
	sys.kill()
	
s3_bucket = input("\nEnter s3 bucket name: ")
read_prefix = input("\nEnter prefix for read directory: ")
write_prefix = input("\nEnter prefix for write directory: ")
	
#Run formatting
format_corpus(s3_bucket, read_prefix, write_prefix, lines_per_file)