import unittest
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from model.data import SelectColumnsTransfomer, DataFrameSelector, ToDummiesTransformer, DataFrameFunctionTransformer, DataFrameFeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



class ExploreData(unittest.TestCase):

	# def test_column_transfer(self):
	# 	df = pd.read_csv("input/train.tsv", sep='\t')
	# 	selector = SelectColumnsTransfomer(['item_condition_id'])
	# 	print(selector.fit(df).transform(df))


	def test_data_content(self):
		df = pd.read_csv("input/train.tsv", sep='\t')

		condition = make_pipeline(
			SelectColumnsTransfomer(["item_condition_id"]),
			ToDummiesTransformer()
		)

		brand = make_pipeline(
			SelectColumnsTransfomer(["brand_name"]),
			DataFrameFunctionTransformer(lambda x: x.astype("category")),
			DataFrameFunctionTransformer(lambda x: x.dropna())
		)

		union = DataFrameFeatureUnion([brand, condition])
		print(union.fit_transform(df))










