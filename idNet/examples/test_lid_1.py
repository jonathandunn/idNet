from idNet import idNet_Train

id = idNet_Train(type = "LID",
				input = "Data_LID_Split",
				output = "LID_OUT",
				s3 = True,
				s3_bucket = "gsproto-lingscan",
				nickname = "500k.1-4grams.400x3",
				divide_data = True,
				test_samples = 20,
				threshold = 200,
				samples_per_epoch = 5,
				language = "eng",
				lid_sample_size = 50,
				did_sample_size = 10
				)
				
id.train(lid_features = 524288,
		lid_ngrams = (1,4),
		did_grammar = ".Grammar.p",
		c2xg_workers = 30,
		mlp_sizes = (400, 400, 400)
		)