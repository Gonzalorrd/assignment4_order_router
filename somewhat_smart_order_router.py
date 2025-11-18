"""Order router that selects the exchange with the best price improvement.

This module loads per-exchange regression models and generates the
'best_price_improvement' function, which can be called by other code.
"""

# Standard library
from __future__ import annotations

from typing import Dict, Optional


# Third-party
import pandas as pd
import numpy as np
from joblib import load
from sklearn.base import RegressorMixin


MODELS_PATH = "per_exchange_price_improvement_models.joblib"

ModelsByExchange = Dict[str, RegressorMixin]
_models: Optional[ModelsByExchange] = None


def _load_models() -> Dict[str, RegressorMixin]:
    """Load and cache the per-exchange models from disk.

    Returns:
        A dictionary mapping exchange codes to trained regressors
    """
    global _models

    if _models is None:
        _models = load(MODELS_PATH)
    return _models


def best_price_improvement(
        symbol:         str,
        side:           str,
        quantity:       int,
        limit_price:    float,
        bid_price:      float,
        ask_price:      float,
        bid_size:       int,
        ask_size:       int,
) -> str:
    """Return the exchange with the highest predicted price improvement.

    The features passed here must match those used when training the models

    Args:
        symbol: Ticker symbol of the order (currently unused).
        side: Order side, 'B' for buy or 'S' for sell.
        quantity: Order quantity.
        limit_price: Order limit price.
        bid_price: Current NBBO bid price.
        ask_price: Current NBBO ask price.
        bid_size: Displayed size at the bid.
        ask_size: Displayed size at the ask.

    Returns:
        The name of the exchange whose model predicts the largest
        price improvement for this order.

    Raises:
        RuntimeError: If no models are loaded from disk.
    """

    #erasing symbol as it it not necessary
    del symbol

    models = _load_models()
    if not models:
        raise RuntimeError("No models were loaded.")

    # encode +1 for buy, -1 for sell
    side_num = 1 if side.upper() == "B" else -1

    features = pd.DataFrame(
        {
            "side_num": [side_num],
            "OrderQty": [quantity],
            "LimitPrice": [limit_price],
            "bid_price": [bid_price],
            "ask_price": [ask_price],
            "bid_size": [bid_size],
            "ask_size": [ask_size],
        },
        dtype="float32",
    )

    best_exchange: Optional[str] = None
    best_prediction = -np.inf

    # Select the best exchange
    for exchange, model in models.items():
        prediction = float(model.predict(features)[0])
        if prediction > best_prediction:
            best_prediction = prediction
            best_exchange = exchange

    # At least one model should exist and select one exchange
    if best_exchange is None:
        raise RuntimeError("No valid prediction could be made.")

    return best_exchange
