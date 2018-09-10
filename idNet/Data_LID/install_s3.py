import time
import os
import codecs
import warnings
import zipfile
import pandas as pd
import numpy as np
import boto3
import multiprocessing as mp
from functools import partial
from random import randint

def format_corpus(s3_bucket, read_prefix, write_prefix, lines_per_file, chars_per_line):
	
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
		if zip_file.endswith(".zip"):
			client.download_file(s3_bucket, zip_file, "temp.zip")
			go_flag = 1
		
		else:
			print("\tError: Expected .zip files.")
			go_flag = 0
			
		if go_flag == 1:
		
			#Open zip file and iterate over its contents
			with zipfile.ZipFile("temp.zip") as z:
				for name in z.namelist():
						
					if name.endswith(".txt"):
						meta = name.split("/")
						domain = meta[0]
						language = meta[1][0:3]
							
						#Format the current language
						with z.open(name) as f:
								
							doc_counter = 1
							line_counter = 0
							total_counter = 0
							char_counter = 0
							
							#Make temp dir
							if os.path.isdir(os.path.join(".", "temp")) == False:
								os.makedirs(os.path.join(".", "temp"))
								
							write_file = os.path.join(".", "temp", language + "." + domain + "." + str(doc_counter) + ".txt")
							fw = codecs.open(write_file, "w", encoding = "utf-8")
								
							#Iterate over lines in the raw file
							for line in f:
									
								line = line.decode("utf-8")
								line = line.strip()
								fw.write(" ")									
									
								#Iterate over characters in the line
								for char in line:
				
									char_counter += 1
									
									#If we've reached the threshold, make a new line
									if char_counter > chars_per_line:

										#Only break between words
										if char == " ":
											char_counter = 0
											fw.write("\n")
											line_counter += 1
											char = ""
												
											#If this line ends the file, start a new file
											if line_counter > lines_per_file - 1:
												fw.close()
												line_counter = 0
												total_counter += 1
												doc_counter += 1
												write_file = os.path.join(".", "temp", language + "." + domain + "." + str(doc_counter) + ".txt")
												fw = codecs.open(write_file, "w", encoding = "utf-8")
										
									#Keep writing
									fw.write(str(char))

									#Finished with this domain-language file, print stats
							fw.close()
							print("\t\t", end = "")
							print(domain, language, total_counter)	

							#Now iterate over temp directory and upload files
							s3_path = os.path.join(write_prefix, domain, language)
							for file in os.listdir(os.path.join(".", "temp")):
							
								client.upload_file(os.path.join(".", "temp", file), s3_bucket, os.path.join(s3_path, file))
								os.remove(os.path.join(".", "temp", file))
								
			#Now remove empty folder and the zip
			os.remove("temp.zip")
			os.rmdir(os.path.join(".", "temp"))
								
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
	
#Get chars_per_line
choice = input("\nHow many characters per observation? (Over-estimate desired training size!) ")
if int(choice) in items:
	chars_per_line = int(choice)
else:
	print("\nInvalid value. Use integer between 1 and 100,000.\n")
	sys.kill()
	
s3_bucket = input("\nEnter s3 bucket name: ")
read_prefix = input("\nEnter prefix for read directory: ")
write_prefix = input("\nEnter prefix for write directory: ")
	
#Run formatting
format_corpus(s3_bucket, read_prefix, write_prefix, lines_per_file, chars_per_line)