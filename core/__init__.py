"""
appgeopy.core - Core data classes for geospatial time-series analysis.

Classes
-------
TimeSeries : pd.Series subclass for 1D time-indexed data
TableArray : pd.DataFrame subclass for tabular data with I/O
SpatialTableArray : gpd.GeoDataFrame subclass for geospatial tabular data
DataCube : xarray.Dataset wrapper for spatial-temporal grids

Quick Start
-----------
>>> from appgeopy.core import TimeSeries, TableArray, SpatialTableArray, DataCube

>>> # Time-series analysis
>>> ts = TimeSeries(data, index=dates)
>>> trend, slope = ts.get_trend()
>>> smoothed = ts.smooth(window=7)

>>> # Tabular data I/O
>>> ta = TableArray.from_excel('data.xlsx', sheet_name='GPS')
>>> ts = ta.extract_timeseries('displacement', index_col='date')

>>> # Geospatial operations
>>> spatial = SpatialTableArray.from_shapefile('stations.shp')
>>> clipped = spatial.clip_to_polygon(region)

>>> # Data cubes
>>> cube = DataCube.from_netcdf('grid.nc')
>>> ts = cube.extract_timeseries(var='temperature', x=500000, y=2000000)

For detailed help on each class, use:
>>> help(TimeSeries)
>>> help(SpatialTableArray)
"""

from .timeseries import TimeSeries
from .tablearray import TableArray
from .spatialtablearray import SpatialTableArray
from .datacube import DataCube

__all__ = ["TimeSeries", "TableArray", "SpatialTableArray", "DataCube"]
