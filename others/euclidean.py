from dataset import data

def euclidean_similarity(person1, person2):
    common_ranked_items = [item for item in data[person1]
        if item in data[person2]]

    rankings = [(data[person1][item], data[person2][item])
        for item in common_ranked_items]

    distance = [pow(rank[0] - rank[1], 2) for rank in rankings]

    return 1 / (1 + sum(distance))