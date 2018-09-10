lidNet Training Data
-----------------------

lidNet is a language identification model that is trained across many domains. This file documents the structure of the training data. For more information about training lidNet, see the main README.md

The data comes in two formats: **Raw** data is zipped and unsplit so that it is easier to store and prepare. **Split** data is divided into folders and batch files for training. The data provided here is Raw. To prepare this data for training a lidNet model, run the *install_local.py* script in this directory. If you are adding new domains to the dataset, drop the zipped files into the Data directory first. See *Adding Data* below for more details.

Formatting Raw Datasets
------------------------

Raw data is organized into sub-folders by domain. The name of the folder is used as the name of the domain. Within domain specific folders, each language is contained in a single file (use utf-8 encoding). The first three letters of this file need to be the desired ISO 639-3 language code.

To convert the raw datasets into unpacked training data, run *install_local.py*. This will:

(1) Unzip the files

(2) Split each language file into sequential observations of N characters. This is before pre-processing; if you're going to train with 50 char samples, it's fine to split a bit larger (lidNet training will enforce sample size)

(3) Create folders for each language under each domain

(4) Split each language into files containing N samples, for training purposes
	
lidNet can be trained using s3 storage in order to reduce instance costs. If you are installing the dataset on an s3 bucket, use *install_s3.py*. This script assumes that you are both reading from and writing to the same bucket and that the instance already has the necessary permissions (i.e., upload the zipped dataset to the s3 bucket first). This script requires boto3.
	
Adding New Domains
-------------------

To add new domains, the data needs to be organized in the same way as the current dataset: located in a directory with the name of the new domain, one language per file. The file must be named so that the first three letters indicate the language's ISO 639-3 language code. Make sure this is correct. If you add a new domain before installing the dataset, just add the zip file to the Data directory. If you add a new domain later, run the relevant install script on the directory which contains the relevant zip file (i.e., place the zip file in an otherwise empty "Data_Raw" folder).

There are some important things to keep in mind when adding new domains:

(1) Language-Domain pairs are sampled equally so it doesn't matter if the new domain is very unbalanced. This will be taken care of during training.

(2) The training algorithm has a pre-processing component that will deal with punctuation, symbols, emojis, etc. There is no need to clean the text before adding it.

(3) Add all the languages available for the domain, even if a language doesn't have many samples. During training you will specify the number of samples and domains required to include a language in the model.