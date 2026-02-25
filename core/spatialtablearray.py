"""
SpatialTableArray - A GeoDataFrame subclass for geospatial tabular data.

Extends geopandas.GeoDataFrame with methods for spatial operations
(buffering, clipping, line segmentation) and multi-format I/O
(Shapefile, GeoJSON, GeoPackage, KML, CSV with coordinates).

Examples
--------
>>> from appgeopy.core import SpatialTableArray
>>> spatial = SpatialTableArray.from_shapefile('stations.shp')
>>> spatial = SpatialTableArray.from_csv('data.csv', x_col='lon', y_col='lat', crs='EPSG:4326')
>>> help(SpatialTableArray)
"""

from __future__ import annotations

import math
import os
import warnings
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from shapely.geometry import Point

from .timeseries import TimeSeries

warnings.filterwarnings("ignore", category=FutureWarning)


class SpatialTableArray(gpd.GeoDataFrame):
    """
    A GeoDataFrame subclass with spatial operations and multi-format I/O.

    Inherits all geopandas.GeoDataFrame functionality. Adds file loaders
    for common geospatial formats, spatial query methods (buffer, clip,
    segmentation), and conversion to TimeSeries.

    Examples
    --------
    >>> spatial = SpatialTableArray.from_shapefile('monitoring_stations.shp')
    >>> clipped = spatial.clip_to_polygon(region)
    >>> ts = clipped.extract_timeseries('displacement', index_col='date')
    """

    _metadata = ["_source_file"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "_source_file"):
            self._source_file = None

    @property
    def _constructor(self):
        return SpatialTableArray

    @property
    def _constructor_sliced(self):
        return pd.Series

    # =========================================================================
    # File I/O: Loading
    # =========================================================================

    @classmethod
    def from_shapefile(cls, filepath, **kwargs):
        """
        Load from an ESRI Shapefile (.shp).

        Parameters
        ----------
        filepath : str
            Path to the .shp file.
        **kwargs
            Additional arguments passed to gpd.read_file().

        Returns
        -------
        SpatialTableArray

        Examples
        --------
        >>> spatial = SpatialTableArray.from_shapefile('stations.shp')
        """
        gdf = gpd.read_file(filepath, **kwargs)
        result = cls(gdf)
        result._source_file = filepath
        return result

    @classmethod
    def from_geojson(cls, filepath, **kwargs):
        """
        Load from a GeoJSON file (.geojson or .json).

        Parameters
        ----------
        filepath : str
            Path to the GeoJSON file.
        **kwargs
            Additional arguments passed to gpd.read_file().

        Returns
        -------
        SpatialTableArray

        Examples
        --------
        >>> spatial = SpatialTableArray.from_geojson('data.geojson')
        """
        gdf = gpd.read_file(filepath, driver="GeoJSON", **kwargs)
        result = cls(gdf)
        result._source_file = filepath
        return result

    @classmethod
    def from_geopackage(cls, filepath, layer=None, **kwargs):
        """
        Load from a GeoPackage file (.gpkg).

        Parameters
        ----------
        filepath : str
            Path to the .gpkg file.
        layer : str, optional
            Layer name. If None, reads the first layer.
        **kwargs
            Additional arguments passed to gpd.read_file().

        Returns
        -------
        SpatialTableArray

        Examples
        --------
        >>> spatial = SpatialTableArray.from_geopackage('data.gpkg', layer='points')
        """
        gdf = gpd.read_file(filepath, layer=layer, **kwargs)
        result = cls(gdf)
        result._source_file = filepath
        return result

    @classmethod
    def from_kml(cls, filepath, **kwargs):
        """
        Load from a KML file (.kml).

        Parameters
        ----------
        filepath : str
            Path to the .kml file.
        **kwargs
            Additional arguments passed to gpd.read_file().

        Returns
        -------
        SpatialTableArray

        Examples
        --------
        >>> spatial = SpatialTableArray.from_kml('survey.kml')

        Notes
        -----
        Requires the 'fiona' library with KML driver support.
        You may need to enable the KML driver via:
        ``import fiona; fiona.drvsupport.supported_drivers['KML'] = 'rw'``
        """
        try:
            import fiona
            fiona.drvsupport.supported_drivers["KML"] = "rw"
        except ImportError:
            pass
        gdf = gpd.read_file(filepath, driver="KML", **kwargs)
        result = cls(gdf)
        result._source_file = filepath
        return result

    @classmethod
    def from_csv(cls, filepath, x_col, y_col, crs="EPSG:4326", **kwargs):
        """
        Load from a CSV file with coordinate columns.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        x_col : str
            Column name for x-coordinates (longitude or easting).
        y_col : str
            Column name for y-coordinates (latitude or northing).
        crs : str, optional
            Coordinate Reference System. Default is 'EPSG:4326' (WGS84).
        **kwargs
            Additional arguments passed to pd.read_csv().

        Returns
        -------
        SpatialTableArray

        Examples
        --------
        >>> spatial = SpatialTableArray.from_csv(
        ...     'stations.csv', x_col='longitude', y_col='latitude',
        ...     crs='EPSG:4326'
        ... )
        """
        df = pd.read_csv(filepath, **kwargs)
        geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]
        gdf = gpd.GeoDataFrame(df, crs=CRS(crs), geometry=geometry)
        result = cls(gdf)
        result._source_file = filepath
        return result

    @classmethod
    def from_dataframe(cls, df, x_col, y_col, crs="EPSG:4326",
                       geom_col_name="geometry"):
        """
        Convert a pandas DataFrame to a SpatialTableArray.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame with coordinate columns.
        x_col : str
            Column name for x-coordinates.
        y_col : str
            Column name for y-coordinates.
        crs : str, optional
            Coordinate Reference System. Default is 'EPSG:4326'.
        geom_col_name : str, optional
            Name of the geometry column. Default is 'geometry'.

        Returns
        -------
        SpatialTableArray

        Examples
        --------
        >>> spatial = SpatialTableArray.from_dataframe(
        ...     df, x_col='easting', y_col='northing', crs='EPSG:32647'
        ... )
        """
        geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]
        gdf = gpd.GeoDataFrame(df, crs=CRS(crs), geometry=geometry)
        if geom_col_name != "geometry":
            gdf = gdf.rename_geometry(geom_col_name)
        return cls(gdf)

    # =========================================================================
    # File I/O: Saving
    # =========================================================================

    def to_shapefile(self, filepath, **kwargs):
        """
        Save to an ESRI Shapefile (.shp).

        Parameters
        ----------
        filepath : str
            Output file path.
        **kwargs
            Additional arguments passed to GeoDataFrame.to_file().

        Examples
        --------
        >>> spatial.to_shapefile('output.shp')
        """
        self.to_file(filepath, driver="ESRI Shapefile", **kwargs)

    def to_geojson(self, filepath, **kwargs):
        """
        Save to a GeoJSON file.

        Parameters
        ----------
        filepath : str
            Output file path.
        **kwargs
            Additional arguments passed to GeoDataFrame.to_file().

        Examples
        --------
        >>> spatial.to_geojson('output.geojson')
        """
        self.to_file(filepath, driver="GeoJSON", **kwargs)

    def to_geopackage(self, filepath, layer=None, **kwargs):
        """
        Save to a GeoPackage file (.gpkg).

        Parameters
        ----------
        filepath : str
            Output file path.
        layer : str, optional
            Layer name.
        **kwargs
            Additional arguments passed to GeoDataFrame.to_file().

        Examples
        --------
        >>> spatial.to_geopackage('output.gpkg', layer='stations')
        """
        self.to_file(filepath, driver="GPKG", layer=layer, **kwargs)

    def to_kml(self, filepath, **kwargs):
        """
        Save to a KML file.

        Parameters
        ----------
        filepath : str
            Output file path.
        **kwargs
            Additional arguments passed to GeoDataFrame.to_file().

        Examples
        --------
        >>> spatial.to_kml('output.kml')

        Notes
        -----
        Requires KML driver support in fiona.
        """
        try:
            import fiona
            fiona.drvsupport.supported_drivers["KML"] = "rw"
        except ImportError:
            pass
        self.to_file(filepath, driver="KML", **kwargs)

    # =========================================================================
    # Spatial Operations
    # =========================================================================

    def find_neighbors(self, target_gdf, key_column, buffer_radius):
        """
        Find target points within a buffer radius from each point.

        For each row in this GeoDataFrame, finds all points in target_gdf
        that are within buffer_radius, and assigns the key value.

        Parameters
        ----------
        target_gdf : GeoDataFrame
            Target points to search.
        key_column : str
            Column containing the key values to assign.
        buffer_radius : float
            Buffer radius in CRS units.

        Returns
        -------
        SpatialTableArray
            Combined results with key column assigned.

        Examples
        --------
        >>> wells = SpatialTableArray.from_shapefile('wells.shp')
        >>> stations = SpatialTableArray.from_shapefile('stations.shp')
        >>> nearby = wells.find_neighbors(stations, 'well_id', buffer_radius=1000)
        """
        results = []
        for _, row in self.iterrows():
            key_value = row[key_column]
            buffer_geom = row.geometry.buffer(buffer_radius)
            mask = target_gdf.geometry.within(buffer_geom)
            selected = target_gdf[mask].copy()
            selected[key_column] = key_value
            results.append(selected)

        if results:
            combined = pd.concat(results, ignore_index=True)
            return SpatialTableArray(combined)
        return SpatialTableArray(columns=target_gdf.columns)

    def clip_to_polygon(self, polygon_geom):
        """
        Extract points located within a polygon.

        Handles various polygon input types: GeoDataFrame, GeoSeries,
        Shapely geometry, or objects with __geo_interface__.

        Parameters
        ----------
        polygon_geom : GeoDataFrame, GeoSeries, or Shapely Polygon
            The polygon to clip to.

        Returns
        -------
        SpatialTableArray
            Points within the polygon.

        Examples
        --------
        >>> region = SpatialTableArray.from_shapefile('study_area.shp')
        >>> points_in = stations.clip_to_polygon(region)
        """
        # Convert to GeoDataFrame
        if isinstance(polygon_geom, gpd.GeoDataFrame):
            poly_gdf = polygon_geom
        elif isinstance(polygon_geom, gpd.GeoSeries):
            poly_gdf = gpd.GeoDataFrame(
                geometry=polygon_geom, crs=polygon_geom.crs
            )
        elif hasattr(polygon_geom, "__geo_interface__"):
            geojson = polygon_geom.__geo_interface__
            if geojson["type"].lower() == "featurecollection":
                if len(geojson["features"]) > 0:
                    geom = shapely.geometry.shape(geojson["features"][0]["geometry"])
                else:
                    raise ValueError("FeatureCollection has no features.")
            else:
                geom = shapely.geometry.shape(geojson)
            poly_gdf = gpd.GeoDataFrame(geometry=[geom], crs=self.crs)
        elif isinstance(
            polygon_geom,
            (shapely.geometry.Polygon, shapely.geometry.MultiPolygon),
        ):
            poly_gdf = gpd.GeoDataFrame(geometry=[polygon_geom], crs=self.crs)
        else:
            raise TypeError(
                "polygon_geom must be a GeoDataFrame, GeoSeries, "
                "or Shapely Polygon/MultiPolygon."
            )

        # Ensure CRS match
        if self.crs != poly_gdf.crs:
            poly_gdf = poly_gdf.to_crs(self.crs)

        result = gpd.sjoin(self, poly_gdf, predicate="within")
        result = result.drop(columns=["index_right"], errors="ignore")
        return SpatialTableArray(result)

    def segment_lines(self, min_segment_length=50, line_name_field="LineName",
                      output_file=None):
        """
        Split line features into segments with minimum specified length.

        Parameters
        ----------
        min_segment_length : float, optional
            Minimum length for each segment (in CRS units). Default is 50.
        line_name_field : str, optional
            Field name containing line identifiers. Default is 'LineName'.
        output_file : str, optional
            Path to save output shapefile. If None, does not save.

        Returns
        -------
        SpatialTableArray
            DataFrame containing segmented lines with 'PARENT' and 'Name'.

        Examples
        --------
        >>> lines = SpatialTableArray.from_shapefile('levee.shp')
        >>> segments = lines.segment_lines(min_segment_length=100)
        """
        from osgeo import ogr

        geodata = self.set_index(line_name_field)
        cache = {"PARENT": [], "Name": []}
        line_geom = []

        for seg_name in geodata.index:
            line_obj = geodata.loc[seg_name, "geometry"]
            if isinstance(line_obj, gpd.GeoSeries):
                shapely_geom = line_obj.iloc[0]
            else:
                shapely_geom = line_obj

            n_segs = max(1, int(shapely_geom.length / min_segment_length))
            ogr_line = ogr.CreateGeometryFromWkt(shapely_geom.wkt)
            split_ogr = self._split_line_multiple(ogr_line, n_pieces=n_segs)
            split_shapely = [self._ogr_to_shapely(g) for g in split_ogr]

            count = len(split_shapely)
            cache["PARENT"].extend([seg_name] * count)
            cache["Name"].extend(
                [f"{seg_name}_Seg_{str(i+1).zfill(3)}" for i in range(count)]
            )
            line_geom.extend(split_shapely)

        result = SpatialTableArray(
            data=cache, geometry=line_geom, crs=self.crs
        )
        if output_file:
            result.to_file(output_file)
        return result

    # =========================================================================
    # Data Extraction
    # =========================================================================

    def extract_timeseries(self, column, index_col=None):
        """
        Extract a column as a TimeSeries object.

        Parameters
        ----------
        column : str
            Column name to extract.
        index_col : str, optional
            Column to use as datetime index. If None, uses existing index.

        Returns
        -------
        TimeSeries

        Examples
        --------
        >>> ts = spatial.extract_timeseries('displacement', index_col='date')
        """
        if index_col is not None:
            idx = pd.to_datetime(self[index_col])
            return TimeSeries(self[column].values, index=idx, name=column)
        return TimeSeries(self[column])

    def generate_point_keys(self, x_col, y_col):
        """
        Generate unique point identifiers from coordinate columns.

        Creates strings in format "X{x_value}Y{y_value}" for each row.

        Parameters
        ----------
        x_col : str
            Column name for x-coordinates.
        y_col : str
            Column name for y-coordinates.

        Returns
        -------
        list of str
            Point key strings.

        Examples
        --------
        >>> keys = spatial.generate_point_keys('easting', 'northing')
        >>> spatial['PointKey'] = keys
        """
        return [
            f"X{int(x)}Y{int(y)}"
            for x, y in zip(self[x_col], self[y_col])
        ]

    # =========================================================================
    # Private helpers for line segmentation
    # =========================================================================

    @staticmethod
    def _distance(a, b):
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])
        return (dx ** 2 + dy ** 2) ** 0.5

    @staticmethod
    def _get_split_point(a, b, dist):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        m = dy / dx
        c = a[1] - (m * a[0])
        x = a[0] + (dist ** 2 / (1 + m ** 2)) ** 0.5
        y = m * x + c
        if not (a[0] <= x <= b[0]) and (a[1] <= y <= b[1]):
            x = a[0] - (dist ** 2 / (1 + m ** 2)) ** 0.5
            y = m * x + c
        return x, y

    @classmethod
    def _split_line_single(cls, line, length):
        from osgeo import ogr
        line_points = line.GetPoints()
        sub_line = ogr.Geometry(ogr.wkbLineString)

        while length > 0:
            d = cls._distance(line_points[0], line_points[1])
            if d > length:
                sp = cls._get_split_point(line_points[0], line_points[1], length)
                sub_line.AddPoint(line_points[0][0], line_points[0][1])
                sub_line.AddPoint(*sp)
                line_points[0] = sp
                break
            if d == length:
                sub_line.AddPoint(*line_points[0])
                sub_line.AddPoint(*line_points[1])
                line_points.remove(line_points[0])
                break
            if d < length:
                sub_line.AddPoint(*line_points[0])
                line_points.remove(line_points[0])
                length -= d

        remainder = ogr.Geometry(ogr.wkbLineString)
        for point in line_points:
            remainder.AddPoint(*point)
        return sub_line, remainder

    @classmethod
    def _split_line_multiple(cls, line, length=None, n_pieces=None):
        if length is None and n_pieces is None:
            raise ValueError("Either length or n_pieces must be specified.")
        if n_pieces is None:
            n_pieces = int(math.ceil(line.Length() / length))
        if length is None:
            length = line.Length() / float(n_pieces)

        segments = []
        remainder = line
        for _ in range(n_pieces - 1):
            segment, remainder = cls._split_line_single(remainder, length)
            segments.append(segment)
        segments.append(remainder)
        return segments

    @staticmethod
    def _ogr_to_shapely(ogr_geom):
        wkt = ogr_geom.ExportToWkt()
        return shapely.from_wkt(wkt)
