#A package to find z-reion's linear bias free parameters value based on 21cmFAST physical parameters inputs or density fields.
from pathlib import Path

try:
    DATA_PATH = Path(__file__).parent / "data"
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass


from . import (
    project_driver,
    statistical_analysis as sa,

)

