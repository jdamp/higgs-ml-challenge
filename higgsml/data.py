import requests
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


class DataSet:
    def __init__(
        self,
        url: str = "http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz",
        fname: str = "atlas-higgs-challenge-2014-v2.csv.gz",
    ) -> None:
        self.basedir = Path(__file__).parent.resolve()
        self.url = url
        self.fname = fname

        self._download()
        self.data = pd.read_csv(
            "data/atlas-higgs-challenge-2014-v2.csv.gz", compression="gzip"
        )

    def _download(self) -> None:
        fpath = Path(f"{self.basedir}/../data/{self.fname}")
        if fpath.is_file():
            print("File already downloaded, skipping...")
            return

        response = requests.get(self.url, stream=True)
        total = int(response.headers.get("content-length", 0))

        with fpath.open("wb") as file, tqdm(
            desc=self.fname, total=total, unit="iB"
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

    @staticmethod
    def split_data_labels(df: pd.DataFrame):
        labels = df.Label
        weights = df.Weight
        df = df.drop(columns=["Label", "KaggleSet", "Weight"])
        return df, labels, weights

    def get_full_data(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        return self.split_data_labels(self.data)

    def get_train_data(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Returns a tuple of data, labels and weights for the training set
        """
        train_data = self.data[self.data.KaggleSet == "t"]
        X, y, w = self.split_data_labels(train_data)
        w = self.rescale_weights(w, y)
        return X, y, w

    def get_test_data(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Returns a tuple of data, labels and weights for the public test set
        """
        test_data = self.data[self.data.KaggleSet == "b"]
        X, y, w = self.split_data_labels(test_data)
        w = self.rescale_weights(w, y)
        return X, y, w

    def get_data_features(self):
        """
        Returns a list of all features that actually belong to the datasets.
        Removes additional variables in the data frame, such as "Weight", "Label",
        "KaggleSet" or "KaggleWeight".
        """
        data_features = []
        for feat in self.data.columns:
            if feat.startswith("PRI") or feat.startswith("DER"):
                data_features.append(feat)
        return data_features

    def rescale_weights(self, w_sel: pd.Series, y_sel: pd.Series) -> pd.Series:
        """Rescales weights such that the sum of weights in a subset of the original
        data still sums up to the expected number of events

        Args:
            w_sel (pd.Series): weights in the subset of the data
            y_sel (pd.Series): labels in the subset of the data

        Returns:
            pd.Series: rescaled weights in the subset of the data
        """
        w = self.data.Weight
        y = self.data.Label
        w_new = w_sel.copy()
        w_new[y_sel == "s"] = (
            w_new[y_sel == "s"] * w[y == "s"].sum() / w_sel[y_sel == "s"].sum()
        )
        w_new[y_sel == "b"] = (
            w_new[y_sel == "b"] * w[y == "b"].sum() / w_sel[y_sel == "b"].sum()
        )

        return w_new
