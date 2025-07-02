import h5py
import numpy as np

class Storage:
    """HDF5 operations for scientific datasets"""
    
    @staticmethod
    def show_dictkeys_recursive(d, indent=0):
        """
        Recursively prints the keys of a dictionary layer by layer until there are no more keys,
        then prints the value of the lowest-level key.

        Parameters:
        d (dict): The dictionary to print keys from.
        indent (int): The current indentation level for nested dictionaries (used for formatting output).
        """
        for key, value in d.items():
            # Print the current key with indentation
            print('    ' * indent + str(key))
            
            # If the value is a dictionary, call the function recursively
            if isinstance(value, dict):
                show_dictkeys_recursive(value, indent + 1)
            else:
                # If it's not a dictionary, print the value (lowest-level key)
                print('    ' * (indent + 1) + str(value))
        
    @staticmethod
    def data_to_hdf5(hdf5_group, data):
        """
        Recursively creates an HDF5 structure from a nested dictionary.

        Parameters:
        hdf5_group (h5py.Group): The current HDF5 group where data will be added.
        data (dict): A dictionary where keys represent group or dataset names, 
                     and values are either further dictionaries or data to be stored.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                # If the value is a dictionary, create or get a group and recurse
                if key in hdf5_group:
                    subgroup = hdf5_group[key]
                else:
                    subgroup = hdf5_group.create_group(key)
                data_to_hdf5(subgroup, value)
            else:
                # If the value is not a dictionary, create a dataset
                if key in hdf5_group:
                    del hdf5_group[key]  # Remove existing dataset if it exists
                hdf5_group.create_dataset(key, data=value)
        
    @staticmethod
    def hdf5_to_data_dict(hdf5_group):
        """
        Recursively extracts data from an HDF5 group into a nested dictionary structure.

        Parameters:
        hdf5_group (h5py.Group): The HDF5 group to extract data from.

        Returns:
        dict: A nested dictionary where keys represent group or dataset names, 
              and values are either further dictionaries or data from datasets.
        """
        data_dict = {}
        for key, item in hdf5_group.items():
            if isinstance(item, h5py.Group):
                # Recursively process groups
                data_dict[key] = hdf5_to_data_dict(item)
            elif isinstance(item, h5py.Dataset):
                # Directly extract datasets
                data_dict[key] = item[...]
        return data_dict
        
    @staticmethod
    def open_HDF5(fpath):
        """
        Quickly open HDF5 file and extract the information.
        """
        # Extract existing data and metadata
        with h5py.File(fpath, "r") as hdf5_file:
            data_dict = hdf5_to_data_dict(hdf5_file)
            metadata_dict = hdf5_to_metadata_dict(hdf5_file)

        return [data_dict, metadata_dict]
        
    @staticmethod
    def metadata_to_hdf5(hdf5_group, metadata):
    """
    Recursively adds metadata to HDF5 groups based on a nested dictionary structure.

    Parameters:
    hdf5_group (h5py.Group): The HDF5 group to add metadata to.
    metadata (dict): A dictionary where keys represent group or attribute names, 
                     and values are either further dictionaries or metadata values to be stored as attributes.
    """
    for key, value in metadata.items():
        if isinstance(value, dict):
            # Check if the subgroup already exists, if not, create it
            if key in hdf5_group:
                subgroup = hdf5_group[key]
            else:
                subgroup = hdf5_group.create_group(key)
            metadata_to_hdf5(subgroup, value)
        else:
            # Add the metadata as an attribute to the current group
            hdf5_group.attrs[key] = value

    @staticmethod
    def hdf5_to_metadata_dict(hdf5_group):
        """
        Recursively extracts metadata from an HDF5 group into a nested dictionary structure.

        Parameters:
        hdf5_group (h5py.Group): The HDF5 group to extract metadata from.

        Returns:
        dict: A nested dictionary where keys represent group names, 
              and values are dictionaries of metadata attributes or further nested dictionaries.
        """
        metadata_dict = {}
        for key, item in hdf5_group.items():
            if isinstance(item, h5py.Group):
                # Process nested groups
                metadata_dict[key] = hdf5_to_metadata_dict(item)
            elif isinstance(item, h5py.Dataset):
                # Datasets are not processed for metadata here
                continue

        # Extract attributes (metadata) for the current group
        for attr_key, attr_value in hdf5_group.attrs.items():
            metadata_dict[attr_key] = attr_value

        return metadata_dict


    @staticmethod
    def update_data_dict(existing_data, new_data):
        """
        Update existing data dictionary with new data.

        Parameters:
        existing_data (dict): The original data dictionary extracted from HDF5.
        new_data (dict): The user-provided data dictionary for updates or additions.

        Returns:
        dict: The updated data dictionary.
        """
        for key, value in new_data.items():
            if key in existing_data:
                if isinstance(value, dict) and isinstance(existing_data[key], dict):
                    # Recursively update nested dictionaries
                    existing_data[key] = update_data_dict(existing_data[key], value)
                else:
                    # Replace the existing dataset
                    existing_data[key] = value
            else:
                # Add new dataset
                existing_data[key] = value
        return existing_data

    @staticmethod
    def update_metadata_dict(existing_metadata, new_metadata):
        """
        Update existing metadata dictionary with new metadata.

        Parameters:
        existing_metadata (dict): The original metadata dictionary extracted from HDF5.
        new_metadata (dict): The user-provided metadata dictionary for updates or additions.

        Returns:
        dict: The updated metadata dictionary.
        """
        for key, value in new_metadata.items():
            if key in existing_metadata:
                if isinstance(value, dict) and isinstance(existing_metadata[key], dict):
                    # Recursively update nested dictionaries
                    existing_metadata[key] = update_metadata_dict(existing_metadata[key], value)
                else:
                    # Replace the existing metadata
                    existing_metadata[key] = value
            else:
                # Add new metadata
                existing_metadata[key] = value
        return existing_metadata

    @staticmethod
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


    @staticmethod
    def merge_dicts(*dicts):
        """Recursively merges multiple dictionaries."""
        result = {}
        for dictionary in dicts:
            for key, value in dictionary.items():
                if isinstance(value, Mapping) and key in result:
                    # If both values are dictionaries, recursively merge them
                    result[key] = merge_dicts(result[key], value)
                else:
                    # Otherwise, just set the value from the current dictionary
                    result[key] = value
        return result