from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import KFold

data = Dataset.load_builtin('ml-100k')

kf = KFold(n_splits=3)

algo = SVD()

for trainset, testset in kf.split(data):

    algo.fit(trainset)
    predictions = algo.test(testset)

    accuracy.rmse(predictions, verbose=True)