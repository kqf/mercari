import gc
import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb

from sklearn.pipeline import make_pipeline, FeatureUnion
from model.data import SelectColumnsTransfomer, DataFrameSelector, DataFrameFunctionTransformer, CategoricalSelector
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer


from sklearn.base import TransformerMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso


class Timer(object):
    sstart = time.clock()
    def __init__(self, msg):
        super(Timer, self).__init__()
        self.msg = '[{0:08.2f} sec] ' + msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        t = time.clock() - self.sstart
        print(self.msg.format(t))

class SillyTransformer(TransformerMixin):

    def fit(self, X, y, *args, **kwargs):
        return self

    def predict(self, X, *args, **kwargs):
        return np.random.randn(X.shape[0])

    def transform(self, X, *_):
        return self.predict(X)

class LassoTransformer(Lasso, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X)

class LassoTransformer(Lasso, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X)

class RidgeTransformer(Ridge, TransformerMixin):


    def transform(self, X, *_):
        return self.predict(X)


class RandomForestTransformer(RandomForestRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X)


class KNeighborsTransformer(KNeighborsRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X)

def func(x):
    return x.astype("category")

def data_reader():
    condition = make_pipeline(
        DataFrameSelector(["item_condition_id"]),
        StandardScaler()
    )

    brand = make_pipeline(
        SelectColumnsTransfomer(["brand_name"]),
        DataFrameFunctionTransformer(func),
        CategoricalSelector(["brand_name"]),
        DictVectorizer()
    )

    union = FeatureUnion([
        ('brand', brand),
        ('condition', condition)
    ])
    return union


def build_model():
    ridge_transformer = Pipeline(steps=[
        # ('scaler', StandardScaler(with_mean=False)),
        # ('poly_feats', PolynomialFeatures()),
        ('data_reader', data_reader()),
        ('ridge', RidgeTransformer())
    ])

    pred_union = FeatureUnion(
        transformer_list=[
            ('ridge', ridge_transformer)
            # ('rand_forest', RandomForestTransformer()),
            # ('lasso1', LassoTransformer()),
            # ('lasso2', LassoTransformer())
            # ('knn', KNeighborsTransformer())
        ],
        n_jobs=-1
    )

    model = Pipeline(steps=[
        # ('read_data', data_reader()),
        ('pred_union', pred_union),
        ('lin_regr', LinearRegression())
    ])

    return model



def data(size = 1000):
    train = pd.read_table('input/train.tsv', engine='c')#.head(size)
    test = pd.read_table('input/test.tsv', engine='c')#.head(size)
    train_data = train.dropna()
    print(train_data.head())
    return train_data, np.log1p(train_data[['price']]), test.dropna()

def main():
    with Timer("reading the data"):
        X, y, X_test = data()

    model = build_model()

    X_input = X[["item_condition_id", "brand_name"]]
    print(X.shape)
    print(y.values.reshape(-1, 1).shape)
    with Timer("Training the model"):
        model.fit(X_input, y.values)
        
    with Timer("Making the predictions"):
        preds = model.predict(X=X_test)

    print(preds.shape)
    X_test['price'] = preds
    X_test['price'].to_csv("some-submission.csv", index=False)

if __name__ == '__main__':
    main()