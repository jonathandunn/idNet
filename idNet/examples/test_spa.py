from idNet import idNet_Train

class_constraints = []
merge_dict = {}

id = idNet_Train(type = "DID",
				input = "Data_CC_GDC_Split",
				output = "DID_OUT",
				s3 = True,
				s3_bucket = "gsproto-lingscan",
				nickname = "spa.Merge.Round1",
				divide_data = True,
				test_samples = 10,
				threshold = 100,
				samples_per_epoch = 100,
				language = "spa",
				lid_sample_size = 50,
				did_sample_size = 10,
				preannotate_cxg = True,
				preannotated_cxg = True,
				cxg_workers = 30,
				class_constraints = class_constraints,
				merge_dict = merge_dict
				)
				
id.train(model_type = "MLP",
		lid_features = 262144,
		lid_ngrams = (1,3),
		did_grammar = ".Grammar.p",
		c2xg_workers = 30,
		mlp_sizes = (1000, 1000, 1000)
		)