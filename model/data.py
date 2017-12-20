import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.base import BaseEstimator, TransformerMixin



# Create a class to select numerical or categorical columns 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def transform(self, X):
        return X[self.attribute_names].to_dict(orient='records')

    def fit(self, X, y=None):
        return self

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def transform(self, X):
        return X[self.attribute_names].values

    def fit(self, X, y=None):
        return self

class SelectColumnsTransfomer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transform_params):
        trans = X[self.columns].copy() 
        return trans    

    def fit(self, X, y=None, **fit_params):
        return self


class DataFrameFunctionTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, func):
        self.func = func

    def transform(self, X, **transformparams):
        return pd.DataFrame(X).apply(self.func).copy()

    def fit(self, X, y=None, **fitparams):
        return self
    
    
class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):

    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers
        
    def transform(self, X, **transformparamn):
        concatted = pd.concat([transformer.transform(X)
                            for transformer in
                            self.fitted_transformers_], join='inner', axis=1).copy()
        return concatted


    def fit(self, X, y=None, **fitparams):
        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self
    

class ToDummiesTransformer(BaseEstimator, TransformerMixin):
    
    def transform(self, X, **transformparams):
        trans = pd.get_dummies(X).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        return self
