# Edited on 2025-06-06 to trigger commit
import warnings  # Warning control

# Filter Warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

# Clean namespace - import key classes only
from .data import TimeSeries, SpatialSeries, IO, Storage
from .temporal import DateTime, TrendAnalysis, SeasonalAnalysis, PeakDetection, Models
from .spatial import Proximity, Geometry, Displacement
from .visualization import TimePlots, SpatialPlots, Charts
from .applications import MLCW

__version__ = "0.2.0"