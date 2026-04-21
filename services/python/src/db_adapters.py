"""Register psycopg2 adapters for numpy scalar types.

Without these, passing a numpy scalar (e.g. the output of `df["x"] / df["y"]` or
most sklearn predictions) through `sqlalchemy.text(...).execute({"col": value})`
ends up calling `repr()` on the scalar and inlining the string `np.float64(3.69)`
into the SQL — which postgres parses as `schema.function` and raises
`InvalidSchemaName: schema "np" does not exist`, silently breaking every INSERT.

Importing this module has the side effect of registering the adapters process-wide.
Do it once, as early as possible in any script that writes numpy values to postgres.
"""

import numpy as np
from psycopg2.extensions import AsIs, Float, register_adapter


def _adapt_numpy_float(value):
    return Float(float(value))


def _adapt_numpy_int(value):
    return AsIs(int(value))


register_adapter(np.float64, _adapt_numpy_float)
register_adapter(np.float32, _adapt_numpy_float)
register_adapter(np.int64, _adapt_numpy_int)
register_adapter(np.int32, _adapt_numpy_int)
register_adapter(np.bool_, lambda v: AsIs("TRUE" if bool(v) else "FALSE"))
