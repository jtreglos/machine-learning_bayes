class NaiveBayesClassifier:
	def __init__(self, nb_attributes):
		self.nb_attributes = nb_attributes
		
		self.examples_count_positive = 0
		self.examples_count_negative = 0

		self.attribute_counts_positive = [0] * nb_attributes
		self.attribute_counts_negative = [0] * nb_attributes

	
	def update(self, attributes, label):
		if label:
			self.attribute_counts_positive += attributes
			self.examples_count_positive += 1
		else:
			self.attribute_counts_negative += attributes
			self.examples_count_negative += 1


	def predict(self, attributes):
		x = self.naiveProbabilities(
				attributes,
				self.attribute_counts_positive,
				self.examples_count_positive,
				self.examples_count_negative
			)
		y = self.naiveProbabilities(
				attributes,
				self.attribute_counts_negative,
				self.examples_count_negative,
				self.examples_count_positive
			)

		return x >= y


	def naiveProbabilities(self, attributes, counts, m, n):
		prior = m / (m + n)

		p = 1.0
		for i in range(self.nb_attributes):
			p /= m
			if attributes[i]:
				p *= counts[i]
			else:
				p *= m - counts[i]

		return prior * p