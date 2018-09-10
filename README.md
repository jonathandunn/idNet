idNet
--------

This package supports a generalized architecture for language identification (*LID*) and dialect identification (*DID*) using a multi-layer perceptron built using Keras. *DID* also supports a Linear SVM classifier using scikit-learn.

To load a model:

	from idNet import idNet_Enrich
	
	lid = idNet_Enrich("Path to model file", s3_bucket)
	did = idNet_Enrich("Path to model file", s3_bucket)
	
s3_bucket takes a str containing an optional s3 bucket to load the model from. The model filename must contain the necessary prefixes.
	
Once a LID model is loaded, it has the following properties:

| lid.n_features	| Number of features in the model (i.e., hashing bins) |
| lid.n_classes		| Number of languages in the model |
| lid.lang_mappings	| Dictionary of {"iso_code": "language_name"} mappings for all ISO 639-3 codes |
| lid.langs		| List of ISO 639-3 codes for languages present in the current model |
	
Once a DID model is loaded, it has the following properties:

	did.n_features		# Number of features in the grammar used to learn the model
	did.n_classes		# Number of countries in the model
	did.country_mappings	# Dictionary of {"iso_code": "country_name"} mappings for all country codes used
	did.countries		# List of country codes for regional dialects (country-level) present in the current model
	
Loaded models perform the following tasks:

	lid.predict(data)	# Takes an array of strings and returns an array of predicted language codes. Also takes individual strings.
	did.predict(data)	# Takes an array of strings and returns an array of predicted country codes. Also takes individual strings.
	
Note: Model filenames need to include ".DID"/".LID" and ".MLP"/".SVM" because this information is used to determine the model type!

Training New Models
----------------------

To train new models, the training data needs to be prepared. This process is automated; see the *Data_DID* and *Data_LID* directories for directions and scripts.

	from idNet import idNet_Train
	
	id = idNet_train()

        type					#(str): Whether to work with language or dialect identification
        input					#(str): Path to input folder
        output					#(str): Path to output folder
        s3 = False				#(boolean): If True, use boto3 to interact with s3 bucket
        s3_bucket = ""				#(str): s3 bucket name as string
        nickname = "Language"			#(str): The nickname for saving / loading models
        divide_data = True			#(boolean): If True, crawl for dataset; if False, just load it
        test_samples = 20			#(int): The number of files for each class to use for testing
        threshold = 100				#(int): Number of files required before language or country is included in the model
        samples_per_epoch = 5			#(int): Number of samples to use per training epoch for language-domain or language-country pairs
        language = ""				#(str): For DID, specifies the language of the current model
        lid_sample_size = 200			#(int): For LID, the number of characters to allow per sample
        did_sample_size	= 1			#(int): For DID, the number of 100 word samples to combine
        preannotate_cxg = False			#(boolean): For DID, if True enrich and save all CxG vectors
        preannotated_cxg = False		#(boolean): For DID, if True just load pre-enriched CxG vectors
        cxg_workers = 1				#(int):	For DID, if pre-enriching dataset, number of workers to use
        class_constraints = []			#(list of strs): Option to constrain the number of classes
        merge_dict = {}				#(dict): Original:New name keys

    id.train(model_type, lid_features, lid_ngrams, did_grammar, c2xg_workers, mlp_sizes, cross_val, dropout, activation, optimizer)

        model_type = "MLP"			#(str): MLP or SVM
        lid_features = 524288			#(int): Number of character n-gram features to allow, hashing only
        lid_ngrams = (1,3)			#(tuple of ints): Range of n-grams to hash
        did_grammar = ".Grammar.p"		#(str): Name of C2xG grammar to use for annotation; allows comparison of different grammars
        c2xg_workers = 1			#(int): For DID, number of workers for c2xg enrichments
        mlp_sizes = (300, 300, 300)		#(tuple of ints): Size and number of layers; e.g., 3 layers at 300 neurons each
        cross_val = False			#(boolean): Whether to use cross-validation rather than a held-out test set
	dropout = 0.25				#(float): The amount of dropout to apply to each layer
	activation = "relu"			#(str): The type of activation; just passes name to Keras
	optimizer = "sgd"			#(str): The type of optimization algorithm; just passes name to Keras
