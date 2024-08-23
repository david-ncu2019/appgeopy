import numpy as np
import pandas as pd
import h5py

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def load_input_data():
    """
    Simulates the loading of general time-series data.
    Returns a dictionary with synthetic data for each location and sensor.
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
    Prepare metadata for each location.
    
    Parameters:
    time_series_data (dict): The dictionary containing locations and their corresponding sensors' data.
    
    Returns:
    dict: Metadata dictionary with information about each location.
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
    Transforms the pandas.Series data into the format suitable for HDF5 storage.
    
    Parameters:
    time_series_data (dict): The dictionary containing locations and their corresponding sensors' data as pandas.Series.
    
    Returns:
    dict: A transformed dictionary with dates as strings and sensor data as numpy arrays.
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

def initialize_hdf5_file(file_name, transformed_data, metadata=None, sensor_metadata=None):
    """
    Initialize an HDF5 file with the structure for storing time-series data and default null metadata.
    
    Parameters:
    file_name (str): The name of the HDF5 file to create.
    transformed_data (dict): The dictionary containing transformed locations and their corresponding sensors' data.
    metadata (dict): The metadata dictionary with information about each location. If None, metadata will default to {'information': 'null'}.
    sensor_metadata (dict): Optional. Dictionary with metadata for each sensor. Keys are location names, values are dicts where keys are sensor names and values are metadata dictionaries.
    
    Returns:
    h5py.File: The initialized HDF5 file object.
    """
    # Create an HDF5 file
    hdf5_file = h5py.File(file_name, 'w')

    # If metadata is not provided, default it to an empty dict, so we can handle each location individually
    if metadata is None:
        metadata = {}
    
    # If sensor_metadata is not provided, default it to an empty dict
    if sensor_metadata is None:
        sensor_metadata = {}

    for location, data in transformed_data.items():
        # Create a group for each location
        location_group = hdf5_file.create_group(location)
        
        # Add metadata as individual attributes of the location group
        location_metadata = metadata.get(location, {'information': 'null'})  # Default metadata to 'null' if not provided
        for key, value in location_metadata.items():
            location_group.attrs[key] = value
        
        # Save date list as a dataset
        location_group.create_dataset('date', data=np.array(data['date'], dtype='S10'))  # Save dates as string array
        
        # Save sensor data as datasets and add metadata
        for sensor, sensor_data in data.items():
            if sensor != 'date':
                sensor_dataset = location_group.create_dataset(sensor, data=sensor_data)

                # Add metadata for this sensor, if available
                sensor_meta = sensor_metadata.get(location, {}).get(sensor, {'information': 'null'})
                for key, value in sensor_meta.items():
                    sensor_dataset.attrs[key] = value
    
    return hdf5_file
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def write_to_hdf5(hdf5_file, transformed_data):
    """
    Write the processed data to the HDF5 file.
    
    Parameters:
    hdf5_file (h5py.File): The HDF5 file object to write to.
    transformed_data (dict): The dictionary containing transformed locations and their corresponding sensors' data.
    """
    for location, data in transformed_data.items():
        location_group = hdf5_file[location]
        
        # Update date list and sensor data if necessary
        location_group['date'][...] = np.array(data['date'], dtype='S10')
        
        for sensor, sensor_data in data.items():
            if sensor != 'date':
                location_group[sensor][...] = sensor_data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def load_hdf5_data(file_name, location=None, sensor=None):
    """
    Load data from an HDF5 file for a specific location and sensor.
    
    Parameters:
    file_name (str): The name of the HDF5 file to load from.
    location (str): The specific location to load (optional).
    sensor (str): The specific sensor to load (optional).
    
    Returns:
    dict: A dictionary with the requested data.
    """
    data = {}
    
    with h5py.File(file_name, 'r') as hdf5_file:
        if location and sensor:
            data = {
                'date': hdf5_file[location]['date'][...].astype(str),
                sensor: hdf5_file[location][sensor][...]
            }
        elif location:
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

def update_hdf5_from_dict(file_name, updates_dict):
    """
    Update an HDF5 file with new data and metadata from a dictionary.

    Parameters:
    file_name (str): The name of the HDF5 file to update.
    updates_dict (dict): A dictionary containing update instructions, organized as follows:
        {
            'Location_001': {
                'sensor_data': {
                    'Sensor_001': np.array([...]),
                    'Sensor_002': np.array([...])
                },
                'sensor_metadata': {
                    'Sensor_001': {'unit': 'cm', 'description': 'Depth data for 2023'},
                    'Sensor_002': {'unit': 'C', 'description': 'Temperature data for 2023'}
                },
                'metadata': {
                    'Longitude': 120.456,
                    'Latitude': 35.789
                }
            },
            ...
        }
    """
    try:
        for location, location_data in updates_dict.items():
            sensor_data = location_data.get('sensor_data', {})
            sensor_metadata = location_data.get('sensor_metadata', {})
            metadata = location_data.get('metadata', {})

            # Update sensor data and their specific metadata
            if sensor_data:
                update_data(file_name, location, sensor_data, sensor_metadata)
            
            # Update location-specific metadata
            if metadata:
                update_metadata(file_name, location, metadata)
                
        print(f"Successfully updated the file '{file_name}'.")
    except Exception as e:
        print(f"An error occurred while updating the file '{file_name}': {e}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def update_data(file_name, location, sensor_data_dict, sensor_metadata=None):
    """
    Update or replace data and optionally add metadata for a specific location in the HDF5 file.
    
    Parameters:
    file_name (str): The name of the HDF5 file.
    location (str): The location to update or add data to.
    sensor_data_dict (dict): A dictionary where keys are sensor names and values are numpy arrays of the new data.
    sensor_metadata (dict, optional): A dictionary where keys are sensor names and values are metadata dictionaries.
    
    Example:
    update_data('time_series_data.h5', 'Location_001', {'Sensor_001': new_data_array}, {'Sensor_001': {'unit': 'cm'}})
    """
    with h5py.File(file_name, 'a') as hdf5_file:  # 'a' mode allows for read/write
        if location in hdf5_file:
            location_group = hdf5_file[location]
        else:
            location_group = hdf5_file.create_group(location)
        
        for sensor, new_data in sensor_data_dict.items():
            if sensor in location_group:
                del location_group[sensor]  # Remove the old dataset
            sensor_dataset = location_group.create_dataset(sensor, data=new_data)
            
            # Add or update sensor-specific metadata if provided
            if sensor_metadata and sensor in sensor_metadata:
                for key, value in sensor_metadata[sensor].items():
                    sensor_dataset.attrs[key] = value
            elif 'information' not in sensor_dataset.attrs:
                sensor_dataset.attrs['information'] = 'null'  # Default null metadata for sensors


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
    update_metadata('time_series_data.h5', 'Location_001', {'Longitude': 50.1234, 'Latitude': 10.5678})
    """
    with h5py.File(file_name, 'a') as hdf5_file:  # 'a' mode allows for read/write
        if location in hdf5_file:
            location_group = hdf5_file[location]
        else:
            location_group = hdf5_file.create_group(location)
        
        for key, value in new_metadata_dict.items():
            location_group.attrs[key] = value  # Add or update the metadata attribute
        
        # Ensure default null metadata if none exists
        if not location_group.attrs.keys():
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

    # Example usage
    with h5py.File('your_file.h5', 'r') as hdf5_file:
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
    """
    datasets = []
    
    def visit_func(name, node):
        if isinstance(node, h5py.Dataset):
            datasets.append(name)
    
    file.visititems(visit_func)
    
    return datasets

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def export_metadata_to_json(file_name, json_file_name, dataset_names=None):
    """
    Export metadata of specified datasets from an HDF5 file to a JSON file.

    Parameters:
    file_name (str): The name of the HDF5 file.
    json_file_name (str): The name of the JSON file to export to.
    dataset_names (list, optional): List of dataset names to export metadata for. If None, all datasets will be exported.
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

def export_data_to_dataframe(file_name, dataset_name):
    """
    Return a pandas DataFrame containing the monitoring data of a specified dataset from an HDF5 file.

    Parameters:
    file_name (str): The name of the HDF5 file.
    dataset_name (str): The name of the dataset to export.

    Returns:
    pd.DataFrame: A DataFrame with datetime and measurement data.
    """
    with h5py.File(file_name, 'r') as hdf5_file:
        available_datasets = list_datasets(hdf5_file)

        if dataset_name not in available_datasets:
            print(f"Dataset '{dataset_name}' not found in the file.")
            print(f"Available datasets: {available_datasets}")
            return None

        # Load data and convert date strings to datetime objects
        date_data = pd.to_datetime(hdf5_file[dataset_name]['date'][...].astype(str), format='%Y%m%d')
        measurement_data = hdf5_file[dataset_name][dataset_name][...]

        # Create a DataFrame
        df = pd.DataFrame({
            'datetime': date_data,
            'measurement': measurement_data
        })

        return df

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