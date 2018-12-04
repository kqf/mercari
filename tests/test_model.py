import pytest
import pandas as pd
from model.model import evaluate


@pytest.mark.parametrize("sample_size", (200, 400, 1000))
def test_model(sample_size):
    train = pd.read_table("data/train.tsv")
    evaluate(train.sample(sample_size))
