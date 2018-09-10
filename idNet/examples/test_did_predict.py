from idNet import idNet_Enrich

test_texts = ["This is the first sentence just in case", "But this is actually the second English sentence."]

id = idNet_Enrich("Model.eng.DID.SVM.eng.p", language = "eng")
results = id.predict(test_texts)
print(results)	