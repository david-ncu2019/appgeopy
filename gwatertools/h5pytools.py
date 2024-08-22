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

def initialize_hdf5_file(file_name, transformed_data, metadata):
    """
    Initialize an HDF5 file with the structure for storing time-series data.
    
    Parameters:
    file_name (str): The name of the HDF5 file to create.
    transformed_data (dict): The dictionary containing transformed locations and their corresponding sensors' data.
    metadata (dict): The metadata dictionary with information about each location.
    
    Returns:
    h5py.File: The initialized HDF5 file object.
    """
    # Create an HDF5 file
    hdf5_file = h5py.File(file_name, 'w')
    
    for location, data in transformed_data.items():
        # Create a group for each location
        location_group = hdf5_file.create_group(location)
        
        # Add metadata as individual attributes of the location group
        for key, value in metadata[location].items():
            location_group.attrs[key] = value
        
        # Save date list as a dataset
        location_group.create_dataset('date', data=np.array(data['date'], dtype='S10'))  # Save dates as string array
        
        # Save sensor data as datasets
        for sensor, sensor_data in data.items():
            if sensor != 'date':
                location_group.create_dataset(sensor, data=sensor_data)
    
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