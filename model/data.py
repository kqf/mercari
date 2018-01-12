import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.base import BaseEstimator, TransformerMixin



# Create a class to select numerical or categorical columns 
class CategoricalSelector(BaseEstimator, TransformerMixin):
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
        
class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, dtype=None, inverse=False,
                 return_vector=True):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector

        if isinstance(self.columns, str):
            self.columns = [self.columns]

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError('Keys are missing in the record: %s' %
                               missing_columns_)

    def transform(self, x):
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError('Input is not a pandas DataFrame')

        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)

        # if the column was selected and inversed = False make sure the column
        # is in the DataFrame
        self._check_if_all_columns_present(x)

        # if only 1 column is returned return a vector instead of a dataframe
        if len(selected_cols) == 1 and self.return_vector:
            return x[selected_cols[0]]
        else:
            return x[selected_cols]