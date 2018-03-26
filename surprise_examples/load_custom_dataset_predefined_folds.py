import os

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold

files_dir = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/')

reader = Reader('ml-100k')

train_file = files_dir + 'u%d.base'
test_file = files_dir + 'u%d.test'
folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()

algo = SVD()

for trainset, testset in pkf.split(data):

    algo.fit(trainset)
    predictions = algo.test(testset)

    accuracy.rmse(predictions, verbose=True)