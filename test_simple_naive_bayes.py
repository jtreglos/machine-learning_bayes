from simple_naive_bayes import *

NEAR = True
FAR = False
SLOW = False
FAST = True
BRAKE_YES = True
BRAKE_NO = False

def run():
	data = [
		{'attributes': [NEAR, SLOW], 'result': BRAKE_YES},
		{'attributes': [NEAR, FAST], 'result': BRAKE_YES},
		{'attributes': [FAR, FAST], 'result': BRAKE_YES},
		{'attributes': [FAR, SLOW], 'result': BRAKE_YES},
		{'attributes': [NEAR, FAST], 'result': BRAKE_YES},
		{'attributes': [FAR, FAST], 'result': BRAKE_NO},
		{'attributes': [NEAR, SLOW], 'result': BRAKE_NO},
	]

	classifier = NaiveBayesClassifier(2)
	for example in data:
		classifier.update(example['attributes'], example['result'])

	print(classifier.predict([FAR, SLOW]))

run()