import time
import os
import codecs
import warnings
import zipfile
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
from random import randint

def format_corpus(read_directory, write_directory, lines_per_file, chars_per_line):
	
	print("\nSplitting and formatting dataset.")
	
	#Make sure the output directory exists
	if os.path.isdir(write_directory) == False:
		os.makedirs(write_directory)
		print("Creating output folder")

	#Walk through the Data folder looking for zip files
	for subdir, dirs, files in os.walk(read_directory):

		for file in files:
			if file.endswith(".zip"):
			
				print("\tStarting: " + file)
				
				#Open zip file and iterate over its contents
				with zipfile.ZipFile(os.path.join(read_directory, file)) as z:
					for name in z.namelist():
						
						if name.endswith(".txt"):
							meta = name.split("/")
							domain = meta[0]
							language = meta[1][0:3]
						
							#Check domain directory
							if os.path.isdir(os.path.join(write_directory, domain)) == False:
								os.makedirs(os.path.join(write_directory, domain))

							#Check domain/lang directory
							if os.path.isdir(os.path.join(write_directory, domain, language)) == False:
								os.makedirs(os.path.join(write_directory, domain, language))
							
							#Format the current language
							with z.open(name) as f:
								
								doc_counter = 1
								line_counter = 0
								total_counter = 0
								char_counter = 0
								
								write_file = os.path.join(write_directory, domain, language, language + "." + domain + "." + str(doc_counter) + ".txt")
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
													write_file = os.path.join(write_directory, domain, language, language + "." + domain + "." + str(doc_counter) + ".txt")
													fw = codecs.open(write_file, "w", encoding = "utf-8")
										
										#Keep writing
										fw.write(str(char))

										#Finished with this domain-language file, print stats
								fw.close()
								print("\t\t", end = "")
								print(domain, language, total_counter)			
		
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
	
#Run formatting
format_corpus("./!ZIPS", "./!SPLIT", lines_per_file, chars_per_line)