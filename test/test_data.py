import unittest
import pandas as pd
from model.data import SelectColumnsTransfomer



class ExploreData(unittest.TestCase):

	def test(self):
		df = pd.read_csv("input/train.tsv", sep='\t')
		selector = SelectColumnsTransfomer(['item_condition_id'])
		print(selector.fit(df).transform(df))

		# print(df[['train_id', 'item_condition_id', 'brand_name', 'price', 'shipping']])
		# print(df.columns.values)


