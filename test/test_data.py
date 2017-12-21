import unittest
import pandas as pd
from sklearn.pipeline import make_pipeline, FeatureUnion
from model.data import SelectColumnsTransfomer, DataFrameSelector, DataFrameFunctionTransformer, CategoricalSelector
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer



class ExploreData(unittest.TestCase):

	# def test_column_transfer(self):
	# 	df = pd.read_csv("input/train.tsv", sep='\t')
	# 	selector = SelectColumnsTransfomer(['item_condition_id'])
	# 	print(selector.fit(df).transform(df))


	def test_data_content(self):
		df = pd.read_csv("input/train.tsv", sep='\t').dropna()

		condition = make_pipeline(
			DataFrameSelector(["item_condition_id"]),
			StandardScaler()
		)

		brand = make_pipeline(
			SelectColumnsTransfomer(["brand_name"]),
			DataFrameFunctionTransformer(lambda x: x.astype("category")),
			# DataFrameFunctionTransformer(lambda x: x.dropna()),
			CategoricalSelector(["brand_name"]),
			DictVectorizer()
		)

		union = FeatureUnion([
			('brand', brand),
			('condition', condition)
		])
		print(union.fit_transform(df))










