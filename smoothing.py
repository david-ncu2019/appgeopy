import numpy as np

import numpy as np

def simple_moving_average(num_arr, window_size=7):
    """
    Calculate the simple moving average of an input array with a given window size.

    Parameters:
        num_arr (array-like): Input array or list of numerical values.
        window_size (int, optional): The size of the moving window. Default is 7. Must be an odd number.

    Returns:
        list: A list containing the moving averages. If a window contains NaN values, the average will ignore them.

    Example:
        >>> simple_moving_average([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, np.nan], window_size=3)
        [1.3333333333333333, 1.5, 3.0, 4.5, 4.5, 6.0, 7.5, 8.0, 8.5, 9.0]
    """
    
    # Get the length of the input array
    data_length = len(num_arr)

    # Calculate the half window size
    half_window_size = int((window_size - 1) / 2)

    # Initialize a list to store the moving averages
    window_average_cache = []

    # Iterate through each element in the input array
    for i in range(data_length):
        # Initialize a temporary list to store values within the window
        temp = []

        # Iterate through the window range centered around the current element
        for j in list(range(-int(window_size / 2), int(window_size / 2) + 1, 1)):
            try:
                # Calculate the index of the element within the window
                out_index = i + j
                
                # If the index is out of bounds (negative), append the current element value
                if out_index < 0:
                    temp.append(num_arr[i])
                else:
                    # Otherwise, append the value at the calculated index
                    temp.append(num_arr[out_index])
            except Exception as e:
                # raise ValueError(f"{e}")
                pass

        # Calculate the average of the values in the window, ignoring NaNs
        window_average = np.nanmean(temp)
        
        # Append the calculated average to the result list
        window_average_cache.append(window_average)

    return window_average_cache

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
