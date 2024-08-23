import numpy as np
import pandas as pd
import h5py

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def load_input_data():
    """
    Simulates the loading of general time-series data.
    
    Returns:
    dict: A dictionary with synthetic data for each location and sensor. Each location key maps to another 
          dictionary, where each sensor key maps to a pandas Series of random time-series data.
    
    Example:
    data = load_input_data()
    print(data['Location_001']['Sensor_001'].head())
    """
    time_series_data = {
        'Location_001': {
            'Sensor_001': pd.Series(np.random.rand(365) * 10, index=pd.date_range('2022-01-01', periods=365)),
            'Sensor_002': pd.Series(np.random.rand(365) * 15, index=pd.date_range('2022-01-01', periods=365)),
            'Sensor_003': pd.Series(np.random.rand(365) * 20, index=pd.date_range('2022-01-01', periods=365)),
        },
        'Location_002': {
            'Sensor_001': pd.Series(np.random.rand(365) * 8, index=pd.date_range('2022-01-01', periods=365)),
            'Sensor_002': pd.Series(np.random.rand(365) * 12, index=pd.date_range('2022-01-01', periods=365)),
        },
        'Location_003': {
            'Sensor_001': pd.Series(np.random.rand(365) * 10, index=pd.date_range('2022-01-01', periods=365)),
            'Sensor_002': pd.Series(np.random.rand(365) * 18, index=pd.date_range('2022-01-01', periods=365)),
            'Sensor_003': pd.Series(np.random.rand(365) * 15, index=pd.date_range('2022-01-01', periods=365)),
            'Sensor_004': pd.Series(np.random.rand(365) * 20, index=pd.date_range('2022-01-01', periods=365)),
        }
    }
    
    return time_series_data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def prepare_metadata(time_series_data):
    """
    Prepare metadata for each location based on provided time-series data.
    
    Parameters:
    time_series_data (dict): A dictionary containing locations and their corresponding sensors' data as pandas Series.
    
    Returns:
    dict: Metadata dictionary with information about each location, including random longitude and latitude, 
          number of sensors, and start and end dates.
    
    Example:
    data = load_input_data()
    metadata = prepare_metadata(data)
    print(metadata['Location_001'])
    """
    metadata = {}
    
    for location, sensors in time_series_data.items():
        # Select the first sensor to get consistent START_DATE and END_DATE
        first_sensor_key = next(iter(sensors))
        start_date = sensors[first_sensor_key].index.min().strftime('%Y%m%d')
        end_date = sensors[first_sensor_key].index.max().strftime('%Y%m%d')
        
        # Build the metadata dictionary for the current location
        metadata[location] = {
            'Longitude': np.random.uniform(-180, 180),  # Random longitude
            'Latitude': np.random.uniform(-90, 90),     # Random latitude
            'Location Name': location,
            'Number of Sensors': len(sensors),          # Number of sensors
            "Start Date": start_date,
            "End Date": end_date,
        }
        
        # Add depth or other characteristic information for each sensor
        metadata[location].update({
            f"Characteristic_{i+1}": np.random.randint(50, 200)
            for i in range(len(sensors))
        })
    
    return metadata

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def transform_data_for_hdf5(time_series_data):
    """
    Transforms the pandas.Series data into a format suitable for HDF5 storage.
    
    Parameters:
    time_series_data (dict): A dictionary containing locations and their corresponding sensors' data as pandas Series.
    
    Returns:
    dict: A transformed dictionary with dates as strings and sensor data as numpy arrays.
    
    Example:
    data = load_input_data()
    transformed_data = transform_data_for_hdf5(data)
    print(transformed_data['Location_001']['date'][:3])  # ['20220101', '20220102', '20220103']
    """
    transformed_data = {}
    
    for location, sensors in time_series_data.items():
        # Extracting the date range from the first sensor
        date_range = sensors[next(iter(sensors))].index.strftime('%Y%m%d').tolist()
        
        transformed_data[location] = {
            'date': date_range
        }
        
        for sensor, series in sensors.items():
            # Ensure alignment with the date range and transform to numpy array
            transformed_data[location][sensor] = series.reindex(pd.to_datetime(date_range)).values
    
    return transformed_data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def initialize_hdf5_file(file_name: str, transformed_data: dict, metadata: dict = None, sensor_metadata: dict = None) -> h5py.File:
    """
    Initialize an HDF5 file with structure for storing time-series data and optional metadata.
    
    Parameters:
    file_name (str): The name of the HDF5 file to create.
    transformed_data (dict): A dictionary containing transformed locations and their corresponding sensors' data.
    metadata (dict, optional): Metadata for each location. Defaults to None.
    sensor_metadata (dict, optional): Metadata for each sensor. Defaults to None.
    
    Returns:
    h5py.File: The initialized HDF5 file object in append mode for further operations.
    
    Example:
    transformed_data = transform_data_for_hdf5(load_input_data())
    metadata = prepare_metadata(load_input_data())
    hdf5_file = initialize_hdf5_file('example.h5', transformed_data, metadata)
    hdf5_file.close()
    """
    with h5py.File(file_name, 'w') as hdf5_file:
        metadata = metadata or {}
        sensor_metadata = sensor_metadata or {}

        for location, data in transformed_data.items():
            # Create a group for each location
            location_group = hdf5_file.create_group(location)
            
            # Add metadata as individual attributes of the location group
            location_metadata = metadata.get(location, {})
            for key, value in location_metadata.items():
                location_group.attrs[key] = value
            
            # Save date list as a dataset
            location_group.create_dataset('date', data=np.array(data['date'], dtype='S10'))  # Save dates as string array
            
            # Save sensor data as datasets and add metadata
            for sensor, sensor_data in data.items():
                if sensor != 'date':
                    sensor_dataset = location_group.create_dataset(sensor, data=sensor_data)
                    # Add metadata for this sensor, if available
                    sensor_meta = sensor_metadata.get(location, {}).get(sensor, {})
                    for key, value in sensor_meta.items():
                        sensor_dataset.attrs[key] = value
    
    return h5py.File(file_name, 'a')  # Open in append mode for further operations
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def write_to_hdf5(hdf5_file: h5py.File, transformed_data: dict):
    """
    Write the processed data to an existing HDF5 file.
    
    Parameters:
    hdf5_file (h5py.File): The HDF5 file object to write to.
    transformed_data (dict): A dictionary containing transformed locations and their corresponding sensors' data.
    
    Example:
    with h5py.File('example.h5', 'a') as hdf5_file:
        transformed_data = transform_data_for_hdf5(load_input_data())
        write_to_hdf5(hdf5_file, transformed_data)
    """
    for location, data in transformed_data.items():
        if location not in hdf5_file:
            location_group = hdf5_file.create_group(location)
        else:
            location_group = hdf5_file[location]

        # Update date list and sensor data if necessary
        location_group['date'][...] = np.array(data['date'], dtype='S10')
        
        for sensor, sensor_data in data.items():
            if sensor != 'date':
                if sensor in location_group:
                    location_group[sensor][...] = sensor_data
                else:
                    location_group.create_dataset(sensor, data=sensor_data)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def load_hdf5_data(file_name: str, location: str = None, sensor: str = None) -> dict:
    """
    Load data from an HDF5 file for a specific location and sensor.
    
    Parameters:
    file_name (str): The name of the HDF5 file to load from.
    location (str, optional): The specific location to load. Default is None, which loads all locations.
    sensor (str, optional): The specific sensor to load. Default is None, which loads all sensors for the location.
    
    Returns:
    dict: A dictionary with the requested data.
    
    Example:
    data = load_hdf5_data('example.h5', 'Location_001', 'Sensor_001')
    print(data['Sensor_001'][:5])
    """
    data = {}
    
    with h5py.File(file_name, 'r') as hdf5_file:
        if location and location in hdf5_file:
            if sensor and sensor in hdf5_file[location]:
                data = {
                    'date': hdf5_file[location]['date'][...].astype(str),
                    sensor: hdf5_file[location][sensor][...]
                }
            else:
                data = {
                    'date': hdf5_file[location]['date'][...].astype(str),
                    **{sensor: hdf5_file[location][sensor][...] for sensor in hdf5_file[location].keys() if sensor != 'date'}
                }
        else:
            for location in hdf5_file.keys():
                data[location] = {
                    'date': hdf5_file[location]['date'][...].astype(str),
                    **{sensor: hdf5_file[location][sensor][...] for sensor in hdf5_file[location].keys() if sensor != 'date'}
                }
    
    return data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def update_hdf5_from_dict(file_name: str, updates_dict: dict):
    """
    Update an HDF5 file with new data and metadata from a dictionary.
    
    Parameters:
    file_name (str): The name of the HDF5 file to update.
    updates_dict (dict): A dictionary containing update instructions for locations, sensors, and metadata.
    
    Example:
    updates = {
        'Location_001': {
            'sensor_data': {'Sensor_001': np.array([1.5, 2.5, 3.5])},
            'metadata': {'Longitude': 50.0, 'Latitude': 10.0}
        }
    }
    update_hdf5_from_dict('example.h5', updates)
    """
    try:
        for location, location_data in updates_dict.items():
            sensor_data = location_data.get('sensor_data', {})
            sensor_metadata = location_data.get('sensor_metadata', {})
            metadata = location_data.get('metadata', {})

            # Update sensor data and their specific metadata
            if sensor_data:
                update_data(file_name, location, sensor_data, sensor_metadata)
            
            # Update location-specific metadata only if explicitly provided
            if metadata:
                update_metadata(file_name, location, metadata)
                
        print(f"Successfully updated the file '{file_name}'.")
    except Exception as e:
        print(f"An error occurred while updating the file '{file_name}': {e}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def update_data(file_name, location, sensor_data_dict, sensor_metadata=None):
    """
    Update or replace data for a specific location in the HDF5 file without altering existing metadata.
    
    Parameters:
    file_name (str): The name of the HDF5 file.
    location (str): The location to update or add data to.
    sensor_data_dict (dict): A dictionary where keys are sensor names and values are numpy arrays of the new data.
    sensor_metadata (dict, optional): Metadata for each sensor. Default is None.
    
    Example:
    update_data('example.h5', 'Location_001', {'Sensor_001': np.array([1.0, 2.0, 3.0])})
    """
    with h5py.File(file_name, 'a') as hdf5_file:  # 'a' mode allows for read/write
        if location not in hdf5_file:
            location_group = hdf5_file.create_group(location)
        else:
            location_group = hdf5_file[location]
        
        for sensor, new_data in sensor_data_dict.items():
            if sensor in location_group:
                # Preserve existing metadata
                old_metadata = {key: location_group[sensor].attrs[key] for key in location_group[sensor].attrs.keys()}
                
                # Delete the old dataset only if it's going to be replaced
                del location_group[sensor]
                
                # Create new dataset with the same name
                sensor_dataset = location_group.create_dataset(sensor, data=new_data)
                
                # Restore metadata, unless new metadata is provided
                if sensor_metadata and sensor in sensor_metadata:
                    for key, value in sensor_metadata[sensor].items():
                        sensor_dataset.attrs[key] = value
                else:
                    for key, value in old_metadata.items():
                        sensor_dataset.attrs[key] = value
            else:
                # Create new dataset if it does not exist
                sensor_dataset = location_group.create_dataset(sensor, data=new_data)
                
                # Apply metadata if provided
                if sensor_metadata and sensor in sensor_metadata:
                    for key, value in sensor_metadata[sensor].items():
                        sensor_dataset.attrs[key] = value

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def update_metadata(file_name, location, new_metadata_dict):
    """
    Update or replace metadata for a specific location in the HDF5 file.
    
    Parameters:
    file_name (str): The name of the HDF5 file.
    location (str): The location to update or add metadata to.
    new_metadata_dict (dict): A dictionary where keys are metadata attribute names and values are the new metadata values.
    
    Example:
    update_metadata('example.h5', 'Location_001', {'Longitude': 120.0, 'Latitude': 35.0})
    """
    with h5py.File(file_name, 'a') as hdf5_file:  # 'a' mode allows for read/write
        if location in hdf5_file:
            location_group = hdf5_file[location]
        else:
            location_group = hdf5_file.create_group(location)
        
        if new_metadata_dict:  # Only update if new metadata is provided
            for key, value in new_metadata_dict.items():
                location_group.attrs[key] = value  # Add or update the metadata attribute
        
        # Preserve existing metadata and do not set 'information' to 'null' if new_metadata_dict is empty
        if not new_metadata_dict and 'information' not in location_group.attrs:
            location_group.attrs['information'] = 'null'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def display_hdf5_structure(file, max_depth=None, current_depth=0, indent=0):
    """
    Recursively prints the structure of an HDF5 file.
    
    Parameters:
    file (h5py.File or h5py.Group): HDF5 file or group to explore.
    max_depth (int): Maximum depth to display in the structure. If None, display all layers.
    current_depth (int): The current depth in the structure (used for recursion).
    indent (int): The indentation level (used for recursion).
    
    Example:
    with h5py.File('example.h5', 'r') as hdf5_file:
        display_hdf5_structure(hdf5_file, max_depth=2)
    """
    if max_depth is not None and current_depth > max_depth:
        return
    
    # Determine the indentation based on the current depth
    indentation = '    ' * indent

    for key in file.keys():
        item = file[key]
        if isinstance(item, h5py.Group):
            print(f"{indentation}{key}/ (Group)")
            display_hdf5_structure(item, max_depth, current_depth + 1, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{indentation}{key} (Dataset): shape {item.shape}, dtype {item.dtype}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def display_dataset_info(file_name, dataset_name):
    """
    Display information about a specific dataset in the HDF5 file.
    
    Parameters:
    file_name (str): The path to the HDF5 file.
    dataset_name (str): The name of the dataset to display.
    
    Example:
    display_dataset_info('example.h5', 'Location_001/Sensor_001')
    """
    with h5py.File(file_name, 'r') as f:
        if dataset_name in f:
            dataset = f[dataset_name]
            data = dataset[...]

            print(f"Dataset: {dataset_name}")
            print(f"  Shape: {data.shape}")
            print(f"  Data Type: {data.dtype}")
            print(f"  Min: {np.nanmin(data)}, Max: {np.nanmax(data)}")
            print(f"  Number of NaNs: {np.sum(np.isnan(data))}")

            # Warning if dataset is empty
            if data.size == 0:
                print(f"WARNING: Dataset '{dataset_name}' is empty.")

            # Warning if dataset contains NaNs
            nan_count = np.sum(np.isnan(data))
            if nan_count > 0:
                print(f"WARNING: Dataset '{dataset_name}' contains {nan_count} NaN values.")

            # Warning if dataset shape is unusual (e.g., too many dimensions)
            if len(data.shape) > 4:
                print(f"WARNING: Dataset '{dataset_name}' has more than 4 dimensions, which may be unexpected.")
            
            # Warning if dataset has an unusual data type
            acceptable_dtypes = [np.float32, np.float64, np.int32, np.int64]
            if data.dtype not in acceptable_dtypes:
                print(f"WARNING: Dataset '{dataset_name}' has an unusual data type: {data.dtype}. Expected one of {acceptable_dtypes}.")
        
        else:
            print(f"Dataset '{dataset_name}' not found in the file.")
            available_datasets = list_datasets(f)
            print(f"Available datasets in the file are: {available_datasets}")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def list_datasets(file):
    """
    Recursively list all datasets in an HDF5 file or group.
    
    Parameters:
    file (h5py.File or h5py.Group): HDF5 file or group to explore.
    
    Returns:
    list: A list of all dataset paths in the file or group.
    
    Example:
    with h5py.File('example.h5', 'r') as hdf5_file:
        datasets = list_datasets(hdf5_file)
        print(datasets)
    """
    datasets = []
    
    def visit_func(name, node):
        if isinstance(node, h5py.Dataset):
            datasets.append(name)
    
    file.visititems(visit_func)
    
    return datasets

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def export_metadata_to_json(file_name: str, json_file_name: str, dataset_names: list = None):
    """
    Export metadata of specified datasets from an HDF5 file to a JSON file.
    
    Parameters:
    file_name (str): The name of the HDF5 file.
    json_file_name (str): The name of the JSON file to export to.
    dataset_names (list, optional): List of dataset names to export metadata for. If None, all datasets will be exported.
    
    Example:
    export_metadata_to_json('example.h5', 'metadata.json', ['Location_001/Sensor_001'])
    """
    metadata_dict = {}

    with h5py.File(file_name, 'r') as hdf5_file:
        available_datasets = list_datasets(hdf5_file)

        if dataset_names is None:
            dataset_names = available_datasets
        else:
            # Validate provided dataset names
            invalid_datasets = [name for name in dataset_names if name not in available_datasets]
            if invalid_datasets:
                print(f"Invalid datasets provided: {invalid_datasets}")
                print(f"Available datasets: {available_datasets}")
                return
        
        # Extract metadata for each specified dataset
        for dataset_name in dataset_names:
            metadata_dict[dataset_name] = {}
            dataset = hdf5_file[dataset_name]
            
            for key, value in dataset.attrs.items():
                metadata_dict[dataset_name][key] = value

    # Export to JSON
    with open(json_file_name, 'w') as json_file:
        json.dump(metadata_dict, json_file, indent=4)

    print(f"Metadata successfully exported to '{json_file_name}'.")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def export_data_to_dataframe(file_name, location_name, sensor_name, datetime_attr='date'):
    """
    Export monitoring data from a specific location and sensor in an HDF5 file to a pandas DataFrame.
    
    Parameters:
    file_name (str): The path to the HDF5 file.
    location_name (str): The name of the location group in the HDF5 file.
    sensor_name (str): The name of the sensor dataset in the HDF5 file.
    datetime_attr (str): The name of the attribute containing datetime values. Default is 'date'.
    
    Returns:
    pd.DataFrame: A DataFrame containing the datetime and measurement data.
    
    Example:
    df = export_data_to_dataframe('example.h5', 'Location_001', 'Sensor_001')
    print(df.head())
    """
    try:
        with h5py.File(file_name, 'r') as hdf5_file:
            # Construct the full path to the dataset
            location_path = f"{location_name}/{sensor_name}"
            
            if location_name not in hdf5_file or sensor_name not in hdf5_file[location_name]:
                available_datasets = list_datasets(hdf5_file)
                print(f"Dataset '{location_path}' not found. Available datasets: {available_datasets}")
                return None
            
            # Extract the datetime data
            date_data = pd.to_datetime(hdf5_file[location_name][datetime_attr][...].astype(str), format='%Y%m%d')
            
            # Extract the measurement data
            measurement_data = hdf5_file[location_name][sensor_name][...]
            
            # Check if the measurement data is multi-dimensional
            if measurement_data.ndim == 1:
                # Single-dimensional data
                df = pd.DataFrame({
                    'datetime': date_data,
                    'value': measurement_data
                })
            else:
                # Multi-dimensional data
                num_columns = measurement_data.shape[1]
                column_names = [f'value{i+1}' for i in range(num_columns)]
                df = pd.DataFrame(measurement_data, columns=column_names)
                df.insert(0, 'datetime', date_data)

        return df

    except KeyError as e:
        print(f"Error: Key '{e}' not found. Please check your location, sensor, and datetime attribute names.")
        available_datasets = list_datasets(hdf5_file)
        print(f"Available datasets: {available_datasets}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""
# Example transformed data
transformed_data = {
    'Location_001': {
        'date': ['20220101', '20220102', '20220103'],
        'Sensor_001': np.array([1.0, 2.0, 3.0]),
        'Sensor_002': np.array([4.0, 5.0, 6.0]),
    },
    'Location_002': {
        'date': ['20220101', '20220102', '20220103'],
        'Sensor_001': np.array([7.0, 8.0, 9.0]),
        'Sensor_002': np.array([10.0, 11.0, 12.0]),
    },
}

# Example location metadata
metadata = {
    'Location_001': {'Longitude': 120.123, 'Latitude': 35.123, 'Location Name': 'Location_001'},
    'Location_002': {'Longitude': 121.123, 'Latitude': 34.123, 'Location Name': 'Location_002'}
}

# Example sensor metadata
sensor_metadata = {
    'Location_001': {
        'Sensor_001': {'unit': 'cm', 'description': 'Depth measurement'},
        'Sensor_002': {'unit': 'cm', 'description': 'Temperature measurement'},
    },
    'Location_002': {
        'Sensor_001': {'unit': 'm', 'description': 'Water level'},
        'Sensor_002': {'unit': 'm', 'description': 'Flow rate'},
    },
}

# Initialize the HDF5 file with sensor metadata
hdf5_file = initialize_hdf5_file('time_series_data.h5', transformed_data, metadata, sensor_metadata)

# Don't forget to close the file after writing
hdf5_file.close()
"""