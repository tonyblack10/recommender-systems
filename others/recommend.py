import math
from dataset import data
from pearson_similarity import pearson_similarity
from euclidean import euclidean_similarity

def recommend(person, bound, similarity=pearson_similarity):
	scores = [(similarity(person, other), other) for other in data if other != person]

	scores.sort()
	scores.reverse()
	scores = scores[0:bound]

	print (scores)

	recomms = {}

	for sim, other in scores:
		ranked = data[other]

		for item in ranked:
			if item not in data[person]:
				weight = sim * ranked[item]

				if item in recomms:
					s, weights = recomms[item]
					recomms[item] = (s + sim, weights + [weight])
				else:
					recomms[item] = (sim, [weight])

	for r in recomms:
		sim, item = recomms[r]
		recomms[r] = sum(item) / sim

	return recomms