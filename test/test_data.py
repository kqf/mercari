import unittest
import pandas as pd
from model.data import SelectColumnsTransfomer, DataFrameSelector



class ExploreData(unittest.TestCase):

	def test_column_transfer(self):
		df = pd.read_csv("input/train.tsv", sep='\t')
		selector = SelectColumnsTransfomer(['item_condition_id'])
		print(selector.fit(df).transform(df))


	def test_dataframe_selector(self):
		df = pd.read_csv("input/train.tsv", sep='\t')
		selector = DataFrameSelector(['item_condition_id'])
		print(selector.fit(df).transform(df))




