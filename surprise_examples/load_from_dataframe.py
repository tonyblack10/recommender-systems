import pandas as pd

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                'userID': [9, 32, 2, 45, 'user_foo'],
                'rating': [3, 2, 4, 3, 1]}

df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

cross_validate(NormalPredictor(), data, cv=2)