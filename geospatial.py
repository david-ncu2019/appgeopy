import warnings  # Warning control
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point

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

    # Apply the point_within_buffer function to each row of central_gdf
    results = central_gdf.apply(
        lambda row: point_within_buffer(row, target_points_gdf, central_key_colum,n buffer_radius), axis=1
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