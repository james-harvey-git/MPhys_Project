# %%
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from keras import layers, models

# %%
h2o_data = xr.open_dataset('ACEFTS_L2_v5p2_H2O.nc')
h2o_flags = xr.open_dataset('ACEFTS_L2_v5p2_flags_H2O.nc')

# %%
h2o_data = h2o_data.set_index(index=['orbit', 'sunset_sunrise'])
h2o_flags = h2o_flags.set_index(index=['orbit', 'sunset_sunrise'])

# %%
vmr_data = h2o_data['H2O'].values
flag_data = h2o_flags['quality_flag']

# %%
def encode_time_features(year, month, day, hour):
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_sin = np.sin(2 * np.pi * day / 31)
    day_cos = np.cos(2 * np.pi * day / 31)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    year_normalized = (year - 2004) / (20)
    return [year_normalized, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos]

# %%
features = []
labels = []
max_altitudes = 150  # Constant for padding

for i in range(flag_data.shape[0]):
    vmr = vmr_data[i, :].values  # VMR values at different altitudes for the observation
    altitude = h2o_data['altitude'].values  # Altitude levels
    
    # Obtain orbit and sunset/sunrise identifier
    orbit_number = h2o_data['orbit'][i].values
    ss = h2o_data['sunset_sunrise'][i].values

    try:
        # Select flags for the specific orbit and sunset/sunrise
        flags = flag_data.sel(orbit=orbit_number, sunset_sunrise=ss).values
        
        # Mask unretrieved values and scaled a priori values
        mask = (flags != 8) & (flags != 9)
        vmr = vmr[mask]
        altitude = altitude[mask]
        flags = flags[mask]

        # Pad VMR and altitude to ensure uniform length
        vmr_padded = np.pad(vmr, (0, max_altitudes - len(vmr)), constant_values=np.nan)
        altitude_padded = np.pad(altitude, (0, max_altitudes - len(altitude)), constant_values=-999)
        flags_padded = np.pad(flags, (0, max_altitudes - len(flags)), constant_values=8)  # Use 8 for padded labels

        # Cyclic time encoding for seasonal and daily trends
        year = h2o_data['year'][i].values
        month = h2o_data['month'][i].values
        day = h2o_data['day'][i].values
        hour = h2o_data['hour'][i].values
        time_features = encode_time_features(year, month, day, hour)  # Expected output: array with cyclic time features

        # Latitude and sunset/sunrise status
        latitude = h2o_data['latitude'][i].values

        # Create feature array for each altitude level
        feature_matrix = np.column_stack([
            vmr_padded,              # VMR
            altitude_padded,         # Altitude
            np.full(max_altitudes, latitude),       # Latitude (same for all altitudes in this observation)
            np.full(max_altitudes, ss),             # Sunset/Sunrise status (same for all altitudes in this observation)
            np.tile(time_features, (max_altitudes, 1))  # Time features repeated for each altitude
        ])
        
        features.append(feature_matrix)
        labels.append(flags_padded)
        
    except KeyError:
        print(f"Skipping missing orbit/sunset_sunrise pair: orbit={orbit_number}, sunset_sunrise={ss}")
        continue

# Convert features and labels to numpy arrays
features_array = np.array(features)  # Shape: (num_observations, max_altitudes, num_features)
labels_array = np.array(labels)      # Shape: (num_observations, max_altitudes)

# %%
# Split the data into training and temp (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(features_array, labels_array, test_size=0.3, random_state=42)

# Split the temp data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set:", X_test.shape, y_test.shape)

# %%
# Define input shape as (max_altitudes, num_features)
num_features = 11
input_shape = (max_altitudes, num_features)

model = models.Sequential()
#model.add(layers.Masking(mask_value=-999, input_shape=(max_altitudes, num_features)))
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(max_altitudes * 9, activation='softmax'))
model.add(layers.Reshape((max_altitudes, 9)))  # Output shape: (num_altitudes, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
from keras._tf_keras.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# %%
from sklearn.utils.class_weight import compute_class_weight

# Flatten the labels to calculate class weights for each unique label
flattened_labels = y_train.flatten()  # assuming y_train is your label array
unique_classes = np.unique(flattened_labels)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=flattened_labels
)

# Convert to dictionary format for Keras
class_weights_dict = {i: class_weights[i] for i in range(len(unique_classes))}

# %%
history = model.fit(
    X_train, y_train,
    epochs=100,               # Set a high number with early stopping
    batch_size=32,            # Start with 32 and experiment if needed
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],  # Early stopping to monitor validation loss
    class_weight=class_weights_dict
)

# %%



