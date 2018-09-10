from idNet import idNet_Train

class_constraints = ['ae', 'at', 'az', 'bb', 'bm', 'br', 'bz', 'ca',
					 'cd', 'ch', 'cl', 'cn', 'cr', 'cx', 'dk', 'es',
					 'fj', 'fm', 'fr', 'ge', 'gm', 'gy', 'hk', 'hr',
					 'ie', 'im', 'in', 'jp', 'lk', 'mn', 'mu', 'my',
					 'ng', 'nl', 'nz', 'ph', 'pk', 'pw', 'qa', 're',
					 'rs', 'ru', 'sc', 'sg', 'sk', 'tc', 'tw', 'ug', 
					 'uk', 'us', 'vc', 'vg', 'vu', 'za']
merge_dict = {}

id = idNet_Train(type = "DID",
				input = "Data_CC_GDC_Split",
				output = "DID_OUT",
				s3 = True,
				s3_bucket = "gsproto-lingscan",
				nickname = "eng.Merge.Round3",
				divide_data = True,
				test_samples = 10,
				threshold = 100,
				samples_per_epoch = 100,
				language = "eng",
				lid_sample_size = 50,
				did_sample_size = 20,
				preannotate_cxg = False,
				preannotated_cxg = True,
				cxg_workers = 30,
				class_constraints = class_constraints,
				merge_dict = merge_dict
				)
				
id.train(model_type = "SVM",
		lid_features = 262144,
		lid_ngrams = (1,3),
		did_grammar = ".Grammar.p",
		c2xg_workers = 30,
		mlp_sizes = (600, 600, 600),
		cross_val = False
		)
		