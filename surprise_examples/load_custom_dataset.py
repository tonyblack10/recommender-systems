import os
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')

reader = Reader(line_format='user item rating timestamp', sep='\t')

data = Dataset.load_from_file(file_path, reader=reader)

cross_validate(BaselineOnly(), data, verbose=True)
