import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


def split_by_njets(
    data: pd.DataFrame, labels: pd.Series, weights: pd.Series
) -> dict[int, tuple[pd.DataFrame, pd.Series]]:
    """_summary_

    Args:
        data (pd.DataFrame): _description_

    Returns:
        dict[int, pd.DataFrame]: _description_
    """
    selection_njets0 = data["PRI_jet_num"] == 0
    data_njets0 = data[selection_njets0].drop(
        columns=[
            "DER_deltaeta_jet_jet",
            "DER_mass_jet_jet",
            "DER_prodeta_jet_jet",
            "DER_lep_eta_centrality",
            "PRI_jet_leading_pt",
            "PRI_jet_leading_eta",
            "PRI_jet_leading_phi",
            "PRI_jet_subleading_pt",
            "PRI_jet_subleading_eta",
            "PRI_jet_subleading_phi",
            "PRI_jet_num",
        ]
    )
    selection_njets1 = data["PRI_jet_num"] == 1
    data_njets1 = data[selection_njets1].drop(
        columns=[
            "DER_deltaeta_jet_jet",
            "DER_mass_jet_jet",
            "DER_prodeta_jet_jet",
            "DER_lep_eta_centrality",
            "PRI_jet_subleading_pt",
            "PRI_jet_subleading_eta",
            "PRI_jet_subleading_phi",
            "PRI_jet_num",
        ]
    )
    selection_njets2p = data["PRI_jet_num"] >= 2
    data_njets2p = data[selection_njets2p]

    return {
        0: (data_njets0, labels[selection_njets0], weights[selection_njets0]),
        1: (data_njets1, labels[selection_njets1], weights[selection_njets1]),
        2: (data_njets2p, labels[selection_njets2p], weights[selection_njets2p]),
    }


def scale_data(data: pd.DataFrame, scaler: TransformerMixin) -> np.array:
    return scaler.fit_transform(data)
