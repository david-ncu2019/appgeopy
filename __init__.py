import warnings
warnings.filterwarnings("ignore")

# Legacy modules (will be removed in a future version)
from .analysis import *
from .data_io import *
from .datetime_handle import *
from .geospatial import *
from .modeling import *
from .smoothing import *
from .visualize import *
from .interpolate import *
from .geocube import *
from .insartools import *
from .gwatertools import *
from .mlcwtools import *
from .timeseriestools import *

# Core classes (new package) - imported last to take precedence over legacy names
from .core import TimeSeries, TableArray, SpatialTableArray, DataCube
