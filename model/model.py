# Explanation:
#    This is a truncated version of winning solution
#    taken from the winners kernel:
#    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
#

import os
import time
from contextlib import contextmanager
from functools import partial
from multiprocessing.pool import ThreadPool
from operator import itemgetter

import keras as ks
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict
from typing import List

os.environ["OMP_NUM_THREADS"] = "1"


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns, records=False):
        self.columns = columns
        self.records = records

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.records:
            return X[self.columns].to_dict(orient="records")
        return X[self.columns]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["name"] = df["name"].fillna("") + " " + df["brand_name"].fillna("")
    df["text"] = (df["item_description"].fillna("") + " " +
                  df["name"] + " " + df["category_name"].fillna(""))
    return df[["name", "text", "shipping", "item_condition_id"]]


def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(
        FunctionTransformer(itemgetter(f), validate=False), *vec)


def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient="records")


def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        use_per_session_threads=1,
        inter_op_parallelism_threads=1)

    with tf.Session(graph=tf.Graph(), config=config) as sess, timer("fit_get"):
        ks.backend.set_session(sess)
        model_in = ks.Input(
            shape=(X_train.shape[1],), dtype="float32", sparse=True)
        out = ks.layers.Dense(192, activation="relu")(model_in)
        out = ks.layers.Dense(64, activation="relu")(out)
        out = ks.layers.Dense(64, activation="relu")(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss="mean_squared_error",
                      optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(3):
            with timer(f"epoch {i + 1}"):
                model.fit(x=X_train, y=y_train, batch_size=2 ** (11 + i),
                          epochs=1, verbose=0)
        # NB: return without quiting the session
        return model.predict(X_test)[:, 0]


def evaluate(train):
    vectorizer = make_union(
        make_pipeline(
            PandasSelector("name"),
            Tfidf(max_features=100000, token_pattern=r"\w+"),
        ),
        make_pipeline(
            PandasSelector("text"),
            Tfidf(max_features=100000, token_pattern=r"\w+",
                  ngram_range=(1, 2)),
        ),
        make_pipeline(
            PandasSelector(["shipping", "item_condition_id"], records=True),
            DictVectorizer()
        ),
        n_jobs=4)
    y_scaler = StandardScaler()

    with timer("process train"):
        train = train[train["price"] > 0].reset_index(drop=True)
        cv = KFold(n_splits=20, shuffle=True, random_state=42)
        train_ids, valid_ids = next(cv.split(train))
        train, valid = train.iloc[train_ids], train.iloc[valid_ids]
        y_train = y_scaler.fit_transform(
            np.log1p(train["price"].values.reshape(-1, 1)))
        X_train = vectorizer.fit_transform(
            preprocess(train)).astype(np.float32)
        print(f"X_train: {X_train.shape} of {X_train.dtype}")
        del train

    with timer("process valid"):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)

    with ThreadPool(processes=4) as pool:
        Xb_train, Xb_valid = [x.astype(np.bool).astype(
            np.float32) for x in [X_train, X_valid]]
        xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
        y_pred = np.mean(
            pool.map(partial(fit_predict, y_train=y_train), xs), axis=0)

    y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
    print("Valid RMSLE: {:.4f}".format(
        np.sqrt(mean_squared_log_error(valid["price"], y_pred))))
