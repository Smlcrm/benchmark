import pandas as pd
import numpy as np

def value_equality(value_0, value_1):
  assert type(value_0) == type(value_1), f"Type '{type(value_0)}' for the first value does not match Type '{type(value_1)}' of the second value."
  if isinstance(value_0, pd.Series):
    return (value_0 == value_1).all()