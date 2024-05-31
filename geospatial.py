import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point


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