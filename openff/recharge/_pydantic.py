try:
    from pydantic.v1 import (
        BaseModel,
        Field,
        constr,
        validator,
        PositiveFloat,
        ValidationError,
    )
except ModuleNotFoundError:
    from pydantic import (
        BaseModel,
        Field,
        constr,
        validator,
        PositiveFloat,
        ValidationError,
    )
