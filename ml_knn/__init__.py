import pandas as pd

def one():
    return 1

def zero():
    return 0

class Classifier:
    def fit(self, df: pd.DataFrame):
        """ Train classifier """

    def predict_row(self, row: pd.Series) -> int:
        """ Predict results """

    def predict(self, df: pd.DataFrame) -> list[int]:
        """ Predict results """
        res = []
        for _, row in df.iterrows():
            res.append(self.predict_row(row))
        return res