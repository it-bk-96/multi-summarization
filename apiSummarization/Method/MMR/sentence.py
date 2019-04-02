class sentence(object):

	def __init__(self, preproWords, originalWords):
		self.preproWords = preproWords
		self.wordFrequencies = self.sentenceWordFreq()
		self.originalWords = originalWords

	def getPreProWords(self):
		return self.preproWords

	def getOriginalWords(self):
		return self.originalWords

	def getWordFreq(self):
		return self.wordFrequencies	

	def sentenceWordFreq(self):
		wordFreq = {}
		for word in self.preproWords:
			if word not in wordFreq.keys():
				wordFreq[word] = 1
			else:
				wordFreq[word] += + 1

		return wordFreq