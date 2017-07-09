import nltk
from nltk.corpus import wordnet as wn

emo_dictionary = {"angry": ["angry"], "surprise": ["surprised", "awesome", "great", "worst"], "trust": ["trust", "hope", "faith", "confidence"], "sad": ["unhappy", "sad"], "happy": ["happy", "good"], "disgust": ["bad", "worse", "worst"], "fear": ["fear", "afraid"]}

target = {"angry": [], "surprise": ["!", ":O"], "trust": [], "sad": [":("], "happy": [":)", ":D"], "disgust": [], "fear": []}


def get_similar_words(word, root, emodict, target_dict):
	for ss in wn.synsets(word):
		target_dict[root] += ss.lemma_names()
		for sim in ss.similar_tos():
			target_dict[root] += sim.lemma_names()

def populate_emo(emodict, target_dict):
	for i, val in enumerate(emodict):
		words = emodict[val]
		for _, word in enumerate(words):
			get_similar_words(word, val, emodict, target_dict)

populate_emo(emo_dictionary, target)

# dataset creation
# with open("./train-data.txt") as f:
# 	lines = f.readlines()

# for block in range(194):
# 	output_file = open('./training/trained-set' + str(block) + '.txt', 'w')
# 	total = []
# 	for line in lines[block*1000:(block*1000)+1000]:
# 			tags = {"angry": 0, "surprise": 0, "trust": 0, "sad": 0, "happy": 0, "disgust": 0, "fear": 0}
# 			for i, val in enumerate(target):
# 				for word in target[val]:
# 					if word in line:
# 						tags[val] += 1
# 			total += line[:-1] + " ---> " + max(tags, key=tags.get) + "\n"
# 	output_file.writelines(total)

# training

def feature (sentence):
	tags = {"angry": 0, "surprise": 0, "trust": 0, "sad": 0, "happy": 0, "disgust": 0, "fear": 0}
	for i, val in enumerate(target):
		for word in target[val]:
			if word in sentence:
				tags[val] += 1
	return tags

tagged_sentences = []
with open("./training/trained-set0.txt") as f:
	for line in f:
		this_feat = feature(line)
		tagged_sentences.append((this_feat, max(this_feat, key=this_feat.get)))

print(len(tagged_sentences[:-4]), len(tagged_sentences[-4:]))


classifier = nltk.NaiveBayesClassifier.train(tagged_sentences[:-4])
print(nltk.classify.accuracy(classifier, tagged_sentences[-4:]))
classifier.show_most_informative_features(5)
sent = raw_input("Please enter something: ")
print(classifier.classify(feature(sent)))
print(feature(sent))
