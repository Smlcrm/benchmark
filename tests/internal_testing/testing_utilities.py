import pandas as pd
import numpy as np

def value_equality(value_0, value_1):
    """Test if two values are equal, handling pandas Series and other types."""
    assert type(value_0) == type(value_1), f"Type '{type(value_0)}' for the first value does not match Type '{type(value_1)}' of the second value."

    if isinstance(value_0, pd.Series):
        try:
            return np.allclose(np.array(value_0.tolist(), dtype=np.float64),
                              np.array(value_1.tolist(), dtype=np.float64))
        except Exception as e:
            return value_0.equals(value_1)
    else:
        return value_0 == value_1

def less_than(value_0, value_1):
    """Test if first value is less than second value, handling different types."""
    assert type(value_0) == type(value_1), f"Type '{type(value_0)}' for the first value does not match Type '{type(value_1)}' of the second value."

    if isinstance(value_0, dict):
        for key in value_0:
            if value_0[key] >= value_1[key]:
                return False
        return True
    elif isinstance(value_0, (list, tuple)):
        return all(v0 < v1 for v0, v1 in zip(value_0, value_1))
    else:
        return value_0 < value_1
