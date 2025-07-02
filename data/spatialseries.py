import geopandas as gpd

from .timeseries import TimeSeries


class SpatialSeries(TimeSeries):
    """Inherits TimeSeries + adds spatial coordinates"""

    def __init__(self, data, coordinates, crs="EPSG:3826", index=None):
        super().__init__(data, index)
        self.coordinates = coordinates
        self.crs = crs

    # Move from geospatial.py:
    def convert_to_geodata(self, xcoord_col, ycoord_col, crs_epsg):
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

    def find_point_neighbors(self, target_points, buffer_radius):
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
            loc=len(selected_points.columns) - 1,
            column=central_key_column,
            value=key_value_series,
        )

        # selected_points[central_key_column] = key_value

        return selected_points

    def find_points_within_polygon(self, polygon_geom):
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
            polygon_gdf = gpd.GeoDataFrame(
                geometry=polygon_geom, crs=polygon_geom.crs
            )
        # Handle case where polygon_geom has __geo_interface__ (FeatureCollection)
        elif hasattr(polygon_geom, "__geo_interface__"):
            geojson = polygon_geom.__geo_interface__

            # Extract geometry from FeatureCollection
            if geojson["type"].lower() == "featurecollection":
                # Extract first feature's geometry
                if len(geojson["features"]) > 0:
                    geom = shapely.geometry.shape(
                        geojson["features"][0]["geometry"]
                    )
                    polygon_gdf = gpd.GeoDataFrame(
                        geometry=[geom], crs=points_gdf.crs
                    )
                else:
                    raise ValueError("FeatureCollection contains no features")
            else:
                # Direct conversion of GeoJSON geometry
                geom = shapely.geometry.shape(geojson)
                polygon_gdf = gpd.GeoDataFrame(
                    geometry=[geom], crs=points_gdf.crs
                )
        # Handle direct Shapely geometry
        elif isinstance(
            polygon_geom,
            (shapely.geometry.Polygon, shapely.geometry.MultiPolygon),
        ):
            polygon_gdf = gpd.GeoDataFrame(
                geometry=[polygon_geom], crs=points_gdf.crs
            )
        else:
            raise TypeError(
                "Polygon must be a GeoDataFrame, GeoSeries, or Shapely geometry"
            )

        # Ensure CRS compatibility
        if points_gdf.crs != polygon_gdf.crs:
            polygon_gdf = polygon_gdf.to_crs(points_gdf.crs)

        # Perform spatial join
        points_within = gpd.sjoin(points_gdf, polygon_gdf, predicate="within")

        # Return result without join artifacts
        return points_within.drop(columns=["index_right"])

    # Move from insartools/ts_disp.py:
    def get_LOS_disp(self, dN, dE, dU, incidence_angle=37, heading_angle=347.6):
        """
        Calculates the line-of-sight (LOS) displacement for a given set of north, east, and up displacement components,
        as well as an incidence angle and heading angle.

        Args:
        dN (float): North displacement component in meters
        dE (float): East displacement component in meters
        dU (float): Up displacement component in meters
        incidence_angle (float): Incidence angle in degrees (default=37)
        heading_angle (float): Heading angle in degrees (default=347.6)

        Returns:
        float: The LOS displacement in meters
        """
        # Convert incidence and heading angles from degrees to radians
        incidence_rad = degree_to_radian(incidence_angle)
        azi_rad = degree_to_radian(heading_angle)

        # Calculate LOS displacement using the formula
        LOS_disp = (
            dU * np.cos(incidence_rad)
            + dN * np.sin(incidence_rad) * np.sin(azi_rad)
            - dE * np.sin(incidence_rad) * np.cos(azi_rad)
        )

        return LOS_disp

    def compare_LOS_disp(self, other_data, mutual_index):
        """
        Compare Line-of-Sight (LOS) displacements between Persistent Scatterers (PSInSAR) and GPS measurements.

        Args:
            psc_df (pd.DataFrame): DataFrame containing displacements of persistent scatterers (PSInSAR).
            gps_df (pd.DataFrame): DataFrame containing GPS station measurements.
            mutual_index (pd.Index or list): Index or list of indices that both `psc_df` and `gps_df` have in common.

        Returns:
            pd.DataFrame: A DataFrame with the difference in LOS displacements between GPS and PSInSAR measurements for the mutual index.

        Raises:
            ValueError: If `mutual_index` is not in both DataFrames.
            TypeError: If inputs are not valid pandas DataFrames or if `mutual_index` is not a valid index type.
        """
        import pandas as pd

        # Type checking
        if not isinstance(psc_df, pd.DataFrame) or not isinstance(
            gps_df, pd.DataFrame
        ):
            raise TypeError(
                "Both `psc_df` and `gps_df` must be pandas DataFrames."
            )

        if not isinstance(mutual_index, (pd.Index, list)):
            raise TypeError(
                "`mutual_index` must be a pandas Index or a list of indices."
            )

        # Check if the mutual index exists in both DataFrames
        if not set(mutual_index).issubset(psc_df.index) or not set(
            mutual_index
        ).issubset(gps_df.index):
            raise ValueError(
                "The `mutual_index` must be present in both `psc_df` and `gps_df`."
            )

        mutual_index = pd.Index(mutual_index)
        psc_by_idx = psc_df.loc[mutual_index]
        gps_by_idx = gps_df.loc[mutual_index]

        if psc_by_idx.empty or gps_by_idx.empty:
            raise ValueError(
                "No common index found after dropping missing values."
            )

        # Calculate the difference in LOS displacements
        diff = gps_by_idx - psc_by_idx

        gps_by_idx = gps_by_idx.rename(columns={"LOS(mm)": "GPS(mm)"})
        psc_by_idx = psc_by_idx.rename(columns={"LOS(mm)": "InSAR(mm)"})

        merge = pd.concat([gps_by_idx, psc_by_idx], axis=1)

        return [diff, merge]

    def convert_cumdisp_to_disp(self):
        """
        Convert a series of cumulative displacement values into individual displacement values.

        Each element in the returned series represents the displacement from the previous element.
        The first value remains unchanged, as there is no preceding element to calculate the difference.

        Args:
        - cumulative_series (pd.Series): A pandas Series representing cumulative displacement values.

        Returns:
        - pd.Series: A new Series where the first value is the same as in the input series,
                     and each subsequent value is the difference between the current and the previous value.
        """
        # Create a copy of the input series to avoid modifying the original data.
        displacement_series = cumulative_series.copy()

        # Shift the series up by one index to align current and previous values.
        shifted_series = cumulative_series.shift(-1)

        # Calculate the difference between the shifted series and the original series.
        differences = shifted_series - cumulative_series

        # Set the first value of the displacement series as the first value of the cumulative series.
        # Assign the calculated differences to the displacement series, skipping the first element.
        displacement_series[1:] = differences[:-1]

        return displacement_series