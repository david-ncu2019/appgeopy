# Add these to your existing imports
import math
import os
import warnings  # Warning control
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import pandas as pd
import shapely
from osgeo import ogr
from pyproj import CRS
from shapely.geometry import Point

# import pygeos

# Filter Warnings
warnings.filterwarnings("ignore")  # Suppress all warnings


def convert_to_geodata(
    df, xcoord_col, ycoord_col, crs_epsg, geom_col_name="geometry"
):
    """
    Convert a pandas DataFrame to a GeoPandas GeoDataFrame.

    Parameters:
    df : pandas.DataFrame
        The DataFrame containing the data.
    xcoord_col : str
        The name of the column containing the x-coordinates.
    ycoord_col : str
        The name of the column containing the y-coordinates.
    crs_epsg : str
        The EPSG code of the coordinate reference system (e.g., "EPSG:4326").
    geom_col_name : str, optional
        The name of the geometry column in the resulting GeoDataFrame (default is "geometry").

    Returns:
    geo_df : geopandas.GeoDataFrame
        The resulting GeoDataFrame with geometry and CRS set.
    """
    try:
        # Extract coordinates
        x_coord = df[xcoord_col]
        y_coord = df[ycoord_col]

        # Create geometry column
        geom_column = [Point(xy) for xy in zip(x_coord, y_coord)]

        # Define CRS
        crs = CRS(crs_epsg)

        # Create GeoDataFrame
        geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geom_column)

        # Rename geometry column if necessary
        if geom_col_name != "geometry":
            geo_df = geo_df.rename_geometry(geom_col_name)

        return geo_df

    except KeyError as e:
        raise KeyError(f"Column not found: {e}")

    except Exception as e:
        raise Exception(f"An error occurred during the conversion: {e}")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def find_point_neighbors(
    central_point, target_points_gdf, central_key_column, buffer_radius
):
    """
    Find points within a buffer radius from a single central point and assign the central point's key to the target points.

    Parameters:
    central_point : GeoSeries
        GeoSeries containing the central point.
    target_points_gdf : GeoDataFrame
        GeoDataFrame containing the target points.
    central_key_column : str
        The column name in central_point containing the key values to assign to target points.
    buffer_radius : float
        The radius of the buffer around the central point in the units of the GeoDataFrame's coordinate reference system.

    Returns:
    GeoDataFrame
        A GeoDataFrame of target points within the buffer, with the central point's key assigned.

    Example:

    # Apply the find_point_neighbors function to each row of central_gdf
    results = central_gdf.apply(
        lambda row: find_point_neighbors(row, target_points_gdf, central_key_colum, buffer_radius), axis=1
    )

    # Concatenate the results into a single GeoDataFrame
    result_gdf = pd.concat(results.tolist(), ignore_index=True)

    """
    # Extract the key value from the central point
    key_value = central_point[central_key_column]

    # Create a buffer around the central point
    central_point_buffer = central_point.geometry.buffer(buffer_radius)

    # Find target points within the buffer
    filter_cond = target_points_gdf.geometry.within(central_point_buffer)
    selected_points = target_points_gdf[filter_cond].copy()

    # Assign the central point's key value to the target points
    key_value_series = pd.Series(
        data=[key_value] * len(selected_points), index=selected_points.index
    )
    selected_points.insert(
        loc=len(selected_points.columns)-1,
        column=central_key_column,
        value=key_value_series,
    )

    # selected_points[central_key_column] = key_value

    return selected_points

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# def find_point_neighbors_pygeos(central_point, target_points_gdf, central_key_column, buffer_radius):
#     # Convert geometries to pygeos geometries
#     central_point_geom = pygeos.from_shapely(central_point.geometry)
#     target_points_geom = pygeos.from_shapely(target_points_gdf.geometry)

#     # Create buffer around the central point
#     buffer = pygeos.buffer(central_point_geom, buffer_radius)

#     # Find points within the buffer
#     within_buffer = pygeos.within(target_points_geom, buffer)

#     # Filter the target points
#     selected_points = target_points_gdf[within_buffer]

#     # Assign the central point's key to selected points
#     selected_points[central_key_column] = central_point[central_key_column]

#     return selected_points

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def find_points_within_polygon(points_gdf, polygon_geom):
    """
    Extract points located within specified polygon geometry.
    
    Args:
        points_gdf (GeoDataFrame): Points data with geometry column
        polygon_geom (Geometry or GeoDataFrame): Polygon geometry or GeoDataFrame containing polygons
        
    Returns:
        GeoDataFrame: Points contained within the polygon
    """
    # Input validation
    if not isinstance(points_gdf, gpd.GeoDataFrame):
        raise TypeError("Input points must be a GeoDataFrame")
    
    # Handle case where polygon_geom is a GeoDataFrame or GeoSeries
    if isinstance(polygon_geom, gpd.GeoDataFrame):
        polygon_gdf = polygon_geom
    elif isinstance(polygon_geom, gpd.GeoSeries):
        polygon_gdf = gpd.GeoDataFrame(geometry=polygon_geom, crs=polygon_geom.crs)
    # Handle case where polygon_geom has __geo_interface__ (FeatureCollection)
    elif hasattr(polygon_geom, '__geo_interface__'):
        geojson = polygon_geom.__geo_interface__
        
        # Extract geometry from FeatureCollection
        if geojson['type'].lower() == 'featurecollection':
            # Extract first feature's geometry
            if len(geojson['features']) > 0:
                geom = shapely.geometry.shape(geojson['features'][0]['geometry'])
                polygon_gdf = gpd.GeoDataFrame(geometry=[geom], crs=points_gdf.crs)
            else:
                raise ValueError("FeatureCollection contains no features")
        else:
            # Direct conversion of GeoJSON geometry 
            geom = shapely.geometry.shape(geojson)
            polygon_gdf = gpd.GeoDataFrame(geometry=[geom], crs=points_gdf.crs)
    # Handle direct Shapely geometry
    elif isinstance(polygon_geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
        polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_geom], crs=points_gdf.crs)
    else:
        raise TypeError("Polygon must be a GeoDataFrame, GeoSeries, or Shapely geometry")
    
    # Ensure CRS compatibility
    if points_gdf.crs != polygon_gdf.crs:
        polygon_gdf = polygon_gdf.to_crs(points_gdf.crs)
    
    # Perform spatial join
    points_within = gpd.sjoin(points_gdf, polygon_gdf, predicate='within')
    
    # Return result without join artifacts
    return points_within.drop(columns=['index_right'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Line Segmentation Functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def _distance(a, b):
    """Calculate Euclidean distance between two points.
    
    Args:
        a: First point as (x, y) tuple
        b: Second point as (x, y) tuple
        
    Returns:
        float: Distance between points
    """
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    return (dx**2 + dy**2) ** 0.5


def _get_split_point(a, b, dist):
    """Calculate point along line a→b at specified distance from point a.
    
    Args:
        a: Start point as (x, y) tuple
        b: End point as (x, y) tuple
        dist: Distance from point a
        
    Returns:
        tuple: (x, y) coordinates of split point
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    m = dy / dx
    c = a[1] - (m * a[0])

    x = a[0] + (dist**2 / (1 + m**2)) ** 0.5
    y = m * x + c
    
    # Handle correct solution (two mathematical solutions exist)
    if not (a[0] <= x <= b[0]) and (a[1] <= y <= b[1]):
        x = a[0] - (dist**2 / (1 + m**2)) ** 0.5
        y = m * x + c

    return x, y


def split_line_single(line, length):
    """Split line at specified length from start.
    
    Args:
        line: OGR LineString geometry object
        length: Distance from start to split point
        
    Returns:
        tuple: (segment, remainder) where segment is first part of length 'length'
               and remainder is the rest of the line
    """
    line_points = line.GetPoints()
    sub_line = ogr.Geometry(ogr.wkbLineString)

    while length > 0:
        d = _distance(line_points[0], line_points[1])
        if d > length:
            split_point = _get_split_point(line_points[0], line_points[1], length)
            sub_line.AddPoint(line_points[0][0], line_points[0][1])
            sub_line.AddPoint(*split_point)
            line_points[0] = split_point
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


def split_line_multiple(line, length=None, n_pieces=None):
    """Split OGR LineString into multiple segments of equal length.
    
    Args:
        line: OGR LineString geometry object
        length: Length of each segment (specify either length or n_pieces)
        n_pieces: Number of segments to create (specify either length or n_pieces)
        
    Returns:
        list: List of OGR LineString geometries representing segments
        
    Raises:
        ValueError: If neither length nor n_pieces is specified
    """
    if length is None and n_pieces is None:
        raise ValueError("Either length or n_pieces must be specified")
        
    if n_pieces is None:
        n_pieces = int(math.ceil(line.Length() / length))
    if length is None:
        length = line.Length() / float(n_pieces)

    line_segments = []
    remainder = line

    for i in range(n_pieces - 1):
        segment, remainder = split_line_single(remainder, length)
        line_segments.append(segment)
    else:
        line_segments.append(remainder)

    return line_segments


def from_ogr_to_shapely(ogr_geom):
    """Convert OGR geometry object to Shapely geometry.
    
    Args:
        ogr_geom: OGR geometry object
        
    Returns:
        shapely.geometry: Equivalent Shapely geometry
    """
    # Convert to WKT format to avoid WKB compatibility issues
    wkt_string = ogr_geom.ExportToWkt()
    
    # Use Shapely's from_wkt function which is more reliable across versions
    shapely_geom = shapely.from_wkt(wkt_string)
    
    return shapely_geom


def segment_lines(input_file, output_file=None, min_segment_length=50, 
                                 line_name_field="LineName"):
    """
    Split line features into segments with minimum specified length.
    
    Args:
        input_file: Path to input shapefile with line features
        output_file: Path to save output shapefile
        min_segment_length: Minimum length for each segment (in CRS units)
        line_name_field: Field name containing line identifiers
        
    Returns:
        GeoDataFrame: DataFrame containing segmented lines
    """
    # Load data
    geodata = gpd.read_file(input_file)
    geodata = geodata.set_index(line_name_field)
    
    cache = {"PARENT": [], "Name": []}
    line_geom = []
    
    for segment_name in geodata.index:
        # Extract the actual Shapely geometry
        line_object = geodata.loc[segment_name, "geometry"]
        
        # Handle GeoSeries vs direct geometry
        if isinstance(line_object, gpd.GeoSeries):
            shapely_geom = line_object.iloc[0]
        else:
            shapely_geom = line_object
            
        # Calculate required number of segments based on minimum length
        line_length = shapely_geom.length
        n_segments = max(1, int(line_length / min_segment_length))
        
        # Convert to OGR for processing
        line_shapely_to_ogr = ogr.CreateGeometryFromWkt(shapely_geom.wkt)
        
        # Split line into calculated number of segments
        split_output_ogr = split_line_multiple(
            line=line_shapely_to_ogr, 
            n_pieces=n_segments
        )
        
        # Convert back to Shapely
        transform_to_shapely = [from_ogr_to_shapely(line_ogr) for line_ogr in split_output_ogr]
        
        # Store results
        segment_count = len(transform_to_shapely)
        cache["PARENT"].extend([segment_name] * segment_count)
        cache["Name"].extend(
            [f"{segment_name}_Seg_{str(i+1).zfill(3)}" for i in range(segment_count)]
        )
        line_geom.extend(transform_to_shapely)
    
    # Create output GeoDataFrame
    output_gdf = gpd.GeoDataFrame(
        data=cache, 
        geometry=line_geom,
        crs=geodata.crs
    )
    
    # Save to file if output path provided
    if output_file:
        output_gdf.to_file(output_file)
    
    return output_gdf