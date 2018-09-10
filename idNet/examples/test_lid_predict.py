from idNet import idNet_Enrich

test_texts = ["This is the first sentence just in case", "But this is actually the second English sentence."]

id = idNet_Enrich("Model.LID.MLP.400kx3_hash.1-3grams.262k.Callback.hdf")
results = id.predict(test_texts)
print(results)