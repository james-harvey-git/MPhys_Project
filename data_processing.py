import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def vertical_profile_plot(
    molecule_name, molecule_data, flag_data, date_index=0, n_surrounding=1, screen_neg_values = True
):
    """
    Plots the vertical gas profile (VMR against altitude) for the inputted molecule at the given date_index,
    along with `n_surrounding` profiles before and after it which have the same occultation type (sunset or sunrise).

    Parameters:
    molecule_name (String): Name of the molecule for the given data.
    molecule_data (Xarray): Xarray for ACE-FTS molecule NetCDF file.
    flag_data (Xarray): Xarray for ACE-FTS molecule quality flags NetCDF file.
    date_index (int): Index/List of the profile(s) to examine (default is 0). If a list is provided, the current profile should 
    correspond to the last index in the list.
    n_surrounding (int): Number of profiles before and after the current one to include in the plot.
    screen_neg_values (bool): If True, negative VMR values are removed from the plot.

    Returns:
    None
    """
    # Convert times to datetime for easy comparison
    years = molecule_data['year'].values
    months = molecule_data['month'].values
    days = molecule_data['day'].values
    hours = molecule_data['hour'].values
    times = pd.to_datetime({
        'year': years,
        'month': months,
        'day': days,
        'hour': hours
    })

    # Extract sunset_sunrise values
    sunset_sunrise = molecule_data['sunset_sunrise'].values

    # Define occultation type: 0 or 2 -> Sunset, 1 or 3 -> Sunrise
    occultation_type_value = np.where((sunset_sunrise == 0) | (sunset_sunrise == 2), "Sunset", "Sunrise")

    # If statement to handle single date_index input
    if isinstance(date_index, np.int32):
        current_date_index = date_index
        current_occultation = occultation_type_value[current_date_index]

        # Sort times and indices
        sorted_indices = np.argsort(times.to_numpy())
        sorted_occultation = occultation_type_value[sorted_indices]

        # Locate the index of the current profile in the sorted array
        current_sorted_index = np.where(sorted_indices == current_date_index)[0][0]

        # Get surrounding indices with the same occultation type
        same_occultation_indices = [
            idx for idx in range(max(0, current_sorted_index - n_surrounding),
                                min(len(sorted_indices), current_sorted_index + n_surrounding + 1))
            if sorted_occultation[idx] == current_occultation
        ]

        surrounding_indices = sorted_indices[same_occultation_indices]
    else:
        surrounding_indices = date_index
        current_date_index = date_index[-1]
        current_occultation = occultation_type_value[current_date_index]

    # Compute offset for logscale plot if negative values are not screened
    offset = 0
    if screen_neg_values == False:
        min_vmr = 0
        for i in surrounding_indices:
            molecule_vmr = molecule_data[f'{molecule_name}'][i, :].values.flatten()
            mask = (flag_data['quality_flag'].sel(orbit=molecule_data['orbit'][i].values, sunset_sunrise=sunset_sunrise[i]).values != 9) & (flag_data['quality_flag'].sel(orbit=molecule_data['orbit'][i].values, sunset_sunrise=sunset_sunrise[i]).values != 8)
            molecule_vmr = molecule_vmr[mask]
            if len(molecule_vmr) == 0:
                continue
            min_vmr_i = np.min(molecule_vmr)
            if min_vmr_i <= 0 and min_vmr_i < min_vmr:
                min_vmr = min_vmr_i
            
        if min_vmr < 0:
            offset = abs(min_vmr) + 1e-7


    # Helper function to plot a single profile
    def plot_single_profile(index, label_suffix, color, add_legend_labels=False):
        if index is None:
            return  # Skip if no profile exists

        # Extract relevant data
        orbit_number = molecule_data['orbit'][index].values
        ss = molecule_data['sunset_sunrise'][index].values
        if ss in [0,2]:
            occultation_type = 'Sunset'
        else:
            occultation_type = 'Sunrise'
        flags_profile = flag_data['quality_flag'].sel(orbit=orbit_number, sunset_sunrise=ss).values
        molecule_vmr = molecule_data[f'{molecule_name}'][index, :].values.flatten()
        latitude = molecule_data['latitude'][index].values

        # Masking
        mask = (flags_profile != 9) & (flags_profile != 8)
        molecule_vmr = molecule_vmr[mask] + offset
        altitude = molecule_data['altitude'].values[mask]
        flags_profile = flags_profile[mask]

        if screen_neg_values:
            positive_mask = molecule_vmr >= 0
            molecule_vmr = molecule_vmr[positive_mask]
            altitude = altitude[positive_mask]
            flags_profile = flags_profile[positive_mask]

        # Identify flagged values
        outlier_mask = (flags_profile == 4) | (flags_profile == 5) | (flags_profile == 6)
        #print(outlier_mask)
        outlier_vmr = molecule_vmr[outlier_mask]
       # print(outlier_vmr)
        outlier_altitude = altitude[outlier_mask]
        #print("Outlier Altitude Range:", outlier_altitude.min(), outlier_altitude.max())
        inst_error_mask = (flags_profile == 7)
        error_vmr = molecule_vmr[inst_error_mask]
        error_altitude = altitude[inst_error_mask]

        # Plot
        plt.plot(
            molecule_vmr, altitude,
            label=f"{molecule_name} {label_suffix}, {occultation_type}, latitude = {latitude}",
            color=color, alpha=0.6
        )
        if add_legend_labels:
            plt.scatter(
                outlier_vmr, outlier_altitude,
                marker = '^' ,facecolors="none", edgecolors="red", s=70, linewidths=1.5, label="Flagged unnatural outliers", alpha=0.7
            )
            #plt.scatter(
                #error_vmr, error_altitude,
                #marker = 's',facecolors="none", edgecolors="black", s=70, linewidths=1.5, label="Flagged instrument errors", alpha = 0.7
            #)
        else:
            plt.scatter(
                outlier_vmr, outlier_altitude,
                marker = '^',facecolors="none", edgecolors="red", s=70, linewidths=1.5, alpha = 0.7
            )
            #plt.scatter(
                #error_vmr, error_altitude,
                #marker= 's', facecolors="none", edgecolors="black", s=70, linewidths=1.5, alpha = 0.7
            #)

    # Create the plot
    plt.figure(figsize=(10, 12))

    # Assign colors to profiles
    colors = plt.cm.viridis(np.linspace(0, 1, len(surrounding_indices)))
    colors = plt.cm.turbo(np.linspace(0, 1, len(surrounding_indices)))
    #colors = plt.cm.tab10(np.linspace(0, 1, len(surrounding_indices)))

    # Plot surrounding profiles
    for i, idx in enumerate(surrounding_indices):
        if idx == current_date_index:
            current_profile_i = i
            break

    for i, idx in enumerate(surrounding_indices):
        label_suffix = (
            "Current Profile" if idx == current_date_index 
            else f"Profile {i - current_profile_i}"
            )
        add_legend_labels = (i == len(surrounding_indices)-1)  # Only add legend labels for the last profile
        plot_single_profile(idx, label_suffix, colors[i], add_legend_labels)

    #plt.scatter([], [], label = f'Offset = {offset:.2e}', color = 'none', edgecolor = 'none')

    # Add labels, title, and legend
    plt.xlabel(f"{molecule_name} VMR [ppv]", fontsize = 20)
    plt.ylabel("Altitude [km]", fontsize = 20)
    plt.xscale("log")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    current_time = times[current_date_index]
    plt.title(
        f"Vertical Profiles for {molecule_name} on and around {current_time.strftime('%Y-%m-%d %H:%M:%S')} ({current_occultation})",
        #f"Vertical Profile for {molecule_name} on {current_time.strftime('%Y-%m-%d %H:%M:%S')} ({current_occultation})",
        fontsize = 20
    )
    plt.legend(fontsize = 17)
    plt.show()

# Example usage:
# vertical_profile_plot("H2O", molecule_data, flag_data, date_index=100, n_surrounding=3)

def filter_date_indices(molecule_name, molecule_data, flag_data, date_indices):
    """
    Filters date indices to remove those with instrumental error flags (7) or missing flag profiles.

    Parameters:
    - molecule_name: String containing the name of the molecule for the given data.
    - molecule_data: Xarray dataset with molecule data.
    - flag_data: Xarray dataset with quality flags.
    - date_indices: List of date indices to filter.

    Returns:
    - filtered_indices: List of valid date indices.
    """
    filtered_indices = []

    for date_index in date_indices:
        # Extract orbit and sunset_sunrise values
        orbit_number = molecule_data['orbit'][date_index].values
        ss = molecule_data['sunset_sunrise'][date_index].values

        try:
            # Extract the flag profile
            flags_profile = flag_data['quality_flag'].sel(orbit=orbit_number, sunset_sunrise=ss).values

            # Check if the profile contains instrumental error flags (7)
            if np.any(flags_profile == 7):
                print(f"Skipping due to instrumental error: orbit={orbit_number}, sunset_sunrise={ss}")
                continue  # Skip this date index

            mask = (flags_profile != 8) & (flags_profile != 9)
            vmr = molecule_data[molecule_name][date_index, :].values[mask]
            if len(vmr) == 0:
                print(f"VMR for date index {date_index} is empty after screening")
                continue


            # Add the valid date index to the list
            filtered_indices.append(date_index)

        except KeyError:
            # Skip if the flag profile doesn't exist
            print(f"Skipping missing flag profile: orbit={orbit_number}, sunset_sunrise={ss}")
            continue

    return filtered_indices

def encode_time_features(year, month, day, hour):
    """Encodes time features (cyclic encoding for periodicity)."""
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_sin = np.sin(2 * np.pi * day / 31)
    day_cos = np.cos(2 * np.pi * day / 31)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    year_normalized = (year - 2004) / (2024 - 2004)  # Normalize between 0 and 1
    return np.array([year_normalized, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos])

def preprocess_profile_data(
    molecule_name, molecule_data, flag_data, date_indices, n_surrounding=2, n_altitudes=100, filter_indices=False
):
    """
    Preprocesses data for training the CNN-LSTM model.

    Parameters:
    - molecule_name: String containing the name of the molecule for the given data.
    - molecule_data: Xarray dataset with molecule data.
    - flag_data: Xarray dataset with quality flags.
    - date_indices: List of indices for profiles to process.
    - n_surrounding: Number of surrounding profiles to include.
    - n_altitudes: Fixed number of altitude levels (for padding/truncation).
    - filter_indices: Boolean indicating whether to filter date indices.

    Returns:
    - X: Feature array of shape (n_samples, n_profiles, n_altitudes, n_features).
    - y: Label array of shape (n_samples,).
    """
    features = []
    labels = []
    feature_indices = []

    # Extract time and surrounding profiles
    years = molecule_data['year'].values
    months = molecule_data['month'].values
    days = molecule_data['day'].values
    hours = molecule_data['hour'].values
    times = pd.to_datetime({
        'year': years,
        'month': months,
        'day': days,
        'hour': hours
    })

    # Extract sunset_sunrise values
    sunset_sunrise = molecule_data['sunset_sunrise'].values

    # Define occultation type: 0 or 2 -> Sunset, 1 or 3 -> Sunrise
    occultation_type_value = np.where((sunset_sunrise == 0) | (sunset_sunrise == 2), "Sunset", "Sunrise")

    # Filter date indices to remove profiles with instrumental error flags (7) or missing flag profiles
    if filter_indices:
        date_indices = filter_date_indices(molecule_name, molecule_data, flag_data, date_indices)


    # Iterate through each profile in the dataset
    for date_index in date_indices:

        current_occultation = occultation_type_value[date_index]

        # Sort profiles by time
        sorted_indices = np.argsort(times.to_numpy())
        sorted_occultation = occultation_type_value[sorted_indices]

        # Locate the index of the current profile in the sorted array
        current_sorted_index = np.where(sorted_indices == date_index)[0][0]

        # Dynamically find preceding indices which are valid and have the same occultation type
        preceding_indices = []
        current_pointer = current_sorted_index - 1  # Start looking before the current profile

        while len(preceding_indices) < n_surrounding and current_pointer >= 0:
            idx = sorted_indices[current_pointer]
            if sorted_occultation[current_pointer] == current_occultation and idx in date_indices:
                preceding_indices.append(idx)  # Store the index in sorted array
            current_pointer -= 1  # Move to the previous index

        # Reverse the list so that earlier indices come first
        preceding_indices = list(reversed(preceding_indices))

        # Add the current profile to the surrounding indices
        surrounding_indices = preceding_indices + [date_index]


        # Process profiles
        profile_features = []
        is_bad = False  # Initialize flag for bad profile
        skip_sample = False # Initialize flag for skipping sample if any flag profile is missing
        for idx in surrounding_indices:
            # Extract data for the profile
            orbit_number = molecule_data['orbit'][idx].values
            ss = molecule_data['sunset_sunrise'][idx].values
            flags_profile = flag_data['quality_flag'].sel(orbit=orbit_number, sunset_sunrise=ss).values
            vmr = molecule_data[molecule_name][idx, :].values.flatten()
            altitude = molecule_data['altitude'].values
            temperature = molecule_data['temperature'][idx, :].values

            # Mask invalid values
            mask = (flags_profile != 9) & (flags_profile != 8)
            vmr = vmr[mask]
            altitude = altitude[mask]
            flags_profile = flags_profile[mask]
            temperature = temperature[mask]

            # Screen for negative VMR values
            positive_mask = vmr >= 0
            vmr = vmr[positive_mask]
            altitude = altitude[positive_mask]
            flags_profile = flags_profile[positive_mask]
            temperature = temperature[positive_mask]

            # Normalize VMR, altitude values
            if len(vmr) == 0:
                print(f"BUG: VMR for date index {idx} is empty after screening. Surrounding indices: {surrounding_indices}")
            vmr = vmr / np.max(vmr)
            altitude = altitude / np.max(molecule_data['altitude'].values)
            temperature = temperature / np.max(temperature)

            # Identify if the current profile is bad
            if idx == date_index and np.any((flags_profile == 4) | (flags_profile == 5) | (flags_profile == 6)):
                is_bad = True

            # Pad or truncate to fixed altitude levels
            vmr_padded = np.pad(vmr, (0, n_altitudes - len(vmr)), constant_values=-0.1)[:n_altitudes]
            altitude_padded = np.pad(altitude, (0, n_altitudes - len(altitude)), constant_values=-0.1)[:n_altitudes]
            temperature_padded = np.pad(temperature, (0, n_altitudes - len(temperature)), constant_values = -0.1)[:n_altitudes]

            # Encode time features and latitude
            year = molecule_data['year'][idx].values
            month = molecule_data['month'][idx].values
            day = molecule_data['day'][idx].values
            hour = molecule_data['hour'][idx].values
            latitude = molecule_data['latitude'][idx].values/90  # Normalize latitude between -1 and 1
            time_features = encode_time_features(year, month, day, hour)

            # Add indicator for the current profile
            is_current_profile = 1 if idx == date_index else 0
            indicator = np.full(n_altitudes, is_current_profile)

            # Combine features for this profile
            profile_vector = np.column_stack([
                vmr_padded,             # VMR values
                altitude_padded,        # Altitude
                np.full(n_altitudes, latitude),  # Latitude (same for all altitudes)
                np.tile(time_features, (n_altitudes, 1)),  # Repeat time features for all altitudes
                temperature_padded,
                indicator  # Indicator for the current profile
            ])
            profile_features.append(profile_vector)
            

        if len(profile_features) < n_surrounding + 1:
            skip_sample = True

        # Skip the sample if any profile_feature does not contain sufficient profiles
        if skip_sample:
            continue

        # Append features and label for the current sample of profiles, as well as the date indices of each profile in the sample
        features.append(profile_features)
        labels.append(1 if is_bad else 0)
        feature_indices.append(surrounding_indices)
    

    # Convert to numpy arrays
    X = np.array(features)  # Shape: (n_samples, n_profiles, n_altitudes, n_features)
    y = np.array(labels)    # Shape: (n_samples,)
    Z = np.array(feature_indices) # Shape: (n_samples, n_profiles)
    return X, y, Z

# Example usage:
# X, y = preprocess_profile_data('H2O', h2o_data, h2o_flags, bad_profile_indices, n_surrounding=2, n_altitudes=100, filter_indices=True)