#A package to find z-reion's linear bias free parameters value based on 21cmFAST physical parameters inputs or density fields.
from pathlib import Path

try:
    DATA_PATH = Path(__file__).parent / "data"
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass


from . import (
    statistical_analysis as sa,
    z_re_field as zre,
    plot_params as pp,
    z_reion_comparison as zrcomp,
)

from .project_driver import (
    get_params_values,
    params_changing_run,
    parameter_2Dspace_run,
    compute_several_21cmFASt_fields,
    make_bt_movie,
)