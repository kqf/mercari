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

SIZE = 10
NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 0
MAX_FEATURES_ITEM_DESCRIPTION = 50000


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



def build_model():
    ridge_transformer = Pipeline(steps=[
        # ('scaler', StandardScaler(with_mean=False)),
        # ('poly_feats', PolynomialFeatures()),
        ('ridge', RidgeTransformer())
    ])

    pred_union = FeatureUnion(
        transformer_list=[
            ('ridge', ridge_transformer),
            # ('rand_forest', RandomForestTransformer()),
            ('lasso', SillyTransformer())
            # ('knn', KNeighborsTransformer())
        ],
        n_jobs=2
    )

    model = Pipeline(steps=[
        ('pred_union', pred_union),
        ('lin_regr', LinearRegression())
    ])

    return SillyTransformer()


def handle_missing_inplace(dataset):
    dataset['item_description'].fillna(value='NaN', inplace=True)
    dataset['category_name'].fillna(value='NaN', inplace=True)
    dataset['brand_name'].fillna(value='NaN', inplace=True)


def remove_missing(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'NaN'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'NaN'

    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'NaN'].index[:NUM_BRANDS]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'NaN'


def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


def data():
    with Timer('Finished to load data'):
        train = pd.read_table('../input/train.tsv', engine='c').head(SIZE)
        test = pd.read_table('../input/test.tsv', engine='c').head(SIZE)

        print('Train shape: ', train.shape)
        print('Test shape: ', test.shape) 

    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    merge = pd.concat([train, test])
    submission = test[['test_id']]

    del train
    del test
    gc.collect()

    with Timer('Finished to handle missing'):
        handle_missing_inplace(merge)

    with Timer('Finished to remove missing items'):
        remove_missing(merge)

    with Timer('Finished to convert categorical'):
        to_categorical(merge)
        print(merge.head())

    with Timer('Finished count vectorize `name`'):
        cv = CountVectorizer(min_df=NAME_MIN_DF)
        X_name = cv.fit_transform(merge['name'])

    with Timer('Finished count vectorize `category_name`'):
        cv = CountVectorizer()
        X_category = cv.fit_transform(merge['category_name'])

    # with Timer('Finished TFIDF vectorize `item_description`'):
    #     tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
    #                          ngram_range=(1, 1),
    #                          stop_words='english')
    #     X_description = tv.fit_transform(merge['item_description'])

    with Timer('Finished label binarize `brand_name`'):
        lb = LabelBinarizer(sparse_output=True)
        X_brand = lb.fit_transform(merge['brand_name'])

    with Timer('Finished to get dummies on `item_condition_id` and `shipping`'):
        X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)

    with Timer('Finished to create sparse merge'):
        sparse_merge = hstack((X_dummies, X_brand, X_category, X_name)).tocsr()
        X = sparse_merge[:nrow_train]
        X_test = sparse_merge[nrow_train:]

    return X, y, X_test, submission
        


def main():
    X, y, X_test, submission = data()
    print(X.shape)
    print(X_test.shape)

    with Timer('Finished to train the pipeline'):
        model = build_model()
        model.fit(X, y.values.reshape(-1, 1))
        
    with Timer('Finished to predict'):
        preds = model.predict(X=X_test)


    submission['price'] = np.expm1(preds)
    submission.to_csv("submission_lgbm_ridge_5.csv", index=False)

if __name__ == '__main__':
    main()