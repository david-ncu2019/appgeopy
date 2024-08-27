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
    Prepare metadata for each location and each sensor based on provided time-series data.
    
    Parameters:
    time_series_data (dict): A dictionary containing locations and their corresponding sensors' data as pandas Series.
    
    Returns:
    tuple: Two dictionaries - one for location metadata and one for sensor metadata.
        - location_metadata: Metadata dictionary with information about each location, including random longitude, latitude,
          number of sensors, and start and end dates.
        - sensor_metadata: Metadata dictionary for each sensor within each location, including sensor type and measurement unit.
    
    Example:
    data = load_input_data()
    location_metadata, sensor_metadata = prepare_metadata(data)
    print(location_metadata['Location_001'])
    print(sensor_metadata['Location_001']['Sensor_001'])
    """
    location_metadata = {}
    sensor_metadata = {}

    for location, sensors in time_series_data.items():
        # Select the first sensor to get consistent START_DATE and END_DATE
        first_sensor_key = next(iter(sensors))
        start_date = sensors[first_sensor_key].index.min().strftime('%Y%m%d')
        end_date = sensors[first_sensor_key].index.max().strftime('%Y%m%d')
        
        # Build the metadata dictionary for the current location
        location_metadata[location] = {
            'Longitude': np.random.uniform(-180, 180),  # Random longitude
            'Latitude': np.random.uniform(-90, 90),     # Random latitude
            'Location Name': location,
            'Number of Sensors': len(sensors),          # Number of sensors
            "Start Date": start_date,
            "End Date": end_date,
        }
        
        # Initialize sensor metadata for the current location
        sensor_metadata[location] = {}
        
        # Add metadata for each sensor
        for sensor_name, series in sensors.items():
            sensor_metadata[location][sensor_name] = {
                'Sensor Type': 'Temperature' if '001' in sensor_name else 'Humidity',  # Example sensor types
                'Measurement Unit': 'Celsius' if '001' in sensor_name else '%',
                'Characteristic': np.random.randint(50, 200)  # Random characteristic
            }
    
    return location_metadata, sensor_metadata
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
    location_metadata, sensor_metadata = prepare_metadata(load_input_data())
    hdf5_file = initialize_hdf5_file('example.h5', transformed_data, location_metadata, sensor_metadata)
    hdf5_file.close()
    """
    with h5py.File(file_name, 'w') as hdf5_file:
        for location, data in transformed_data.items():
            # Create a group for each location
            location_group = hdf5_file.create_group(location)
            
            # Add location metadata as individual attributes of the location group
            if metadata and location in metadata:
                for key, value in metadata[location].items():
                    location_group.attrs[key] = value
            
            # Save date list as a dataset
            location_group.create_dataset('date', data=np.array(data['date'], dtype='S10'))  # Save dates as string array
            
            # Save sensor data as datasets and add metadata for each sensor
            for sensor, sensor_data in data.items():
                if sensor != 'date':
                    sensor_dataset = location_group.create_dataset(sensor, data=sensor_data)
                    
                    # Add sensor-specific metadata, if available
                    if sensor_metadata and location in sensor_metadata and sensor in sensor_metadata[location]:
                        for key, value in sensor_metadata[location][sensor].items():
                            sensor_dataset.attrs[key] = value
    
    # Explicitly reopen the file in append mode for further operations
    return h5py.File(file_name, 'a')


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

def update_hdf5(file_name: str, updates_dict: dict):
    """
    Updates both data and metadata in an HDF5 file.

    Parameters:
    file_name (str): The name of the HDF5 file to update.
    updates_dict (dict): A dictionary with updates for data and metadata.

    Example:
    update_hdf5('example.h5', updates)
    """
    with h5py.File(file_name, 'a') as hdf5_file:
        for location, data in updates_dict.items():
            if location not in hdf5_file:
                location_group = hdf5_file.create_group(location)
            else:
                location_group = hdf5_file[location]

            # Update location metadata if available
            if 'metadata' in data:
                for key, value in data['metadata'].items():
                    location_group.attrs[key] = value

            # Update sensor metadata and data
            if 'sensor_metadata' in data:
                for sensor, sensor_meta in data['sensor_metadata'].items():
                    if sensor in location_group:
                        sensor_dataset = location_group[sensor]
                        for key, value in sensor_meta.items():
                            sensor_dataset.attrs[key] = value

            # Update sensor data if available (but we are not modifying measurement data in this task)
            for sensor, sensor_data in data.get('sensor_data', {}).items():
                if sensor in location_group:
                    existing_dataset = location_group[sensor]
                    # Assuming we are keeping the original data intact
                    existing_dataset[...] = sensor_data  # This line will just keep the data as is

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
