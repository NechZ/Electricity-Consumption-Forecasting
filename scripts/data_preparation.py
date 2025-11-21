import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import plotly.express as px

# Load the CSV file
df = pd.read_csv("../data/raw_data.csv", sep=';')

# Convert Unix timestamp to datetime
df['deviceTimestamp'] = pd.to_datetime(df['deviceTimestamp'])

# Convert 'value' column to numeric, treating 'NULL' as NaN
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# Extract date and hour from timestamp for hourly aggregation
df['date'] = df['deviceTimestamp'].dt.date
df['hour'] = df['deviceTimestamp'].dt.floor('H')

# Aggregate to hourly data (using mean)
hourly_df = df.groupby(['hour', 'deviceId'])['value'].mean().reset_index()

# Pivot the hourly table: hours as rows, deviceIds as columns
pivot_hourly_df = hourly_df.pivot(index='hour', columns='deviceId', values='value').sort_index()

# Create daily aggregated data for stationarity testing
daily_df = df.groupby(['date', 'deviceId'])['value'].mean().reset_index()
daily_df['date'] = pd.to_datetime(daily_df['date'])
pivot_daily_df = daily_df.pivot(index='date', columns='deviceId', values='value').sort_index()

# Optional: Rename columns to value_i format for both dataframes
device_columns = pivot_hourly_df.columns.tolist()
rename_map = {col: f"value_{idx}" for idx, col in enumerate(device_columns)}

pivot_hourly_df = pivot_hourly_df.rename(columns=rename_map)
pivot_daily_df = pivot_daily_df.rename(columns=rename_map)

value_columns = list(rename_map.values())
device_lookup = {new: original for original, new in rename_map.items()}

# Test stationarity on daily data
stationary_columns = ['deviceTimestamp']
for col in value_columns:
    if col in pivot_daily_df.columns:
        # Get non-null values for the column from daily data
        values = pivot_daily_df[col].dropna()
        if len(values) > 0:
            # Augmented Dickey-Fuller test for stationarity
            adf_result = adfuller(values, autolag='AIC')
            p_value = adf_result[1]
            
            # If p-value < 0.05, reject H0 (series is stationary)
            if p_value < 0.05:
                stationary_columns.append(col)
                print(f"{col}: Stationary (p-value: {p_value:.4f})")
            else:
                print(f"{col}: Non-stationary (p-value: {p_value:.4f})")

# Reset index to make hour a column (renamed to deviceTimestamp for consistency)
pivot_hourly_df.reset_index(inplace=True)
pivot_hourly_df = pivot_hourly_df.rename(columns={'hour': 'deviceTimestamp'})

# Filter hourly data to only include stationary columns (based on daily test)
pivot_hourly_df = pivot_hourly_df[stationary_columns]
pivot_hourly_df.to_csv("src/real_data.csv", index=False)

# Update value_columns to only include stationary columns
value_columns = [col for col in value_columns if col in stationary_columns]

print(f"Filtered out non-stationary columns. Kept {len(stationary_columns)-1} out of {len(device_columns)} value columns.")

# Filter columns that don't have continuous data from 2022-01-01 to 2025-10-19
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2025-10-19')

# Filter the dataframe to the specified date range
date_filtered_df = pivot_hourly_df[
    (pivot_hourly_df['deviceTimestamp'] >= start_date) & 
    (pivot_hourly_df['deviceTimestamp'] <= end_date)
].copy()

# Reindex to a complete hourly grid so interpolation/fill can work on true gaps
if not date_filtered_df.empty:
    full_idx = pd.date_range(
        start=date_filtered_df['deviceTimestamp'].min().floor('H'),
        end=date_filtered_df['deviceTimestamp'].max().ceil('H'),
        freq='H'
    )
    date_filtered_df = (
        date_filtered_df
        .set_index('deviceTimestamp')
        .reindex(full_idx)
        .rename_axis('deviceTimestamp')
        .reset_index()
    )

# Plot data availability for each series
fig, ax = plt.subplots(figsize=(15, 8))

# Create a matrix to show data availability
availability_matrix = []
series_names = []

for col in value_columns:
    if col in date_filtered_df.columns:
        # Create binary mask: 1 for available data, 0 for missing
        availability = (~date_filtered_df[col].isna()).astype(int)
        availability_matrix.append(availability.values)
        series_names.append(col)

if availability_matrix:
    availability_matrix = np.array(availability_matrix)
    
    # Create the heatmap
    im = ax.imshow(availability_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    
    # Set labels
    ax.set_yticks(range(len(series_names)))
    ax.set_yticklabels(series_names)
    ax.set_xlabel('Time (Hours)')
    ax.set_ylabel('Series')
    ax.set_title('Data Availability by Series (Green = Available, Red = Missing)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Data Available')
    
    # Set x-axis to show some date labels
    n_ticks = 10
    tick_positions = np.linspace(0, len(date_filtered_df)-1, n_ticks, dtype=int)
    tick_labels = date_filtered_df.iloc[tick_positions]['deviceTimestamp'].dt.strftime('%Y-%m-%d')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    
    plt.tight_layout()
    plt.show()

# Create histogram of series count by starting date
series_start_dates = {}

for col in value_columns:
    if col in date_filtered_df.columns:
        # Find the first non-null value date for each series
        first_valid_idx = date_filtered_df[col].first_valid_index()
        if first_valid_idx is not None:
            start_date = date_filtered_df.loc[first_valid_idx, 'deviceTimestamp']
            series_start_dates[col] = start_date

# Convert to DataFrame for easier plotting
start_dates_df = pd.DataFrame(list(series_start_dates.items()), columns=['series', 'start_date'])
start_dates_df['start_date'] = pd.to_datetime(start_dates_df['start_date'])

# Create histogram
plt.figure(figsize=(12, 6))
plt.hist(start_dates_df['start_date'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Starting Date')
plt.ylabel('Number of Series')
plt.title('Distribution of Series by Starting Date')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print some statistics
print(f"\nSeries Starting Date Statistics:")
print(f"Earliest starting series: {start_dates_df['start_date'].min()}")
print(f"Latest starting series: {start_dates_df['start_date'].max()}")
print(f"Total series with data: {len(start_dates_df)}")

# Find the most prevalent starting date
start_dates_df['start_date_only'] = start_dates_df['start_date'].dt.date
most_prevalent_date = start_dates_df['start_date_only'].mode().iloc[0]
most_prevalent_datetime = pd.to_datetime(most_prevalent_date)
count_at_prevalent = (start_dates_df['start_date_only'] == most_prevalent_date).sum()

print(f"Most prevalent starting date: {most_prevalent_date} (used by {count_at_prevalent} series)")

# Filter data to start from the most prevalent starting date
filtered_df = date_filtered_df[date_filtered_df['deviceTimestamp'] >= most_prevalent_datetime].copy()

# Ensure the final window is also a complete hourly grid
if not filtered_df.empty:
    final_idx = pd.date_range(
        start=filtered_df['deviceTimestamp'].min().floor('H'),
        end=filtered_df['deviceTimestamp'].max().ceil('H'),
        freq='H'
    )
    filtered_df = (
        filtered_df
        .set_index('deviceTimestamp')
        .reindex(final_idx)
        .rename_axis('deviceTimestamp')
        .reset_index()
    )

# Check which columns have continuous data from the latest start date
continuous_columns = ['deviceTimestamp']
for col in value_columns:
    if col in filtered_df.columns:
        # Calculate percentage of non-null values
        total_rows = len(filtered_df)
        non_null_count = filtered_df[col].notna().sum()
        availability_percentage = (non_null_count / total_rows) * 100
        
        # Check if column has any missing values from the latest start date
        if not availability_percentage < 100.0:
            continuous_columns.append(col)
            print(f"{col}: Has continuous data from {most_prevalent_datetime} (100.0% available)")
        elif availability_percentage >= 95.0:
            # Impute missing values using linear interpolation
            # Set deviceTimestamp as index temporarily for time-aware interpolation
            temp_df = filtered_df.set_index('deviceTimestamp')
            temp_df[col] = temp_df[col].interpolate(method='time', limit_direction='both')
            # For any remaining NaNs at the edges, use forward/backward fill
            temp_df[col] = temp_df[col].ffill().bfill()
            # Reset index and update the original dataframe
            filtered_df = temp_df.reset_index()
            continuous_columns.append(col)
            print(f"{col}: {availability_percentage:.1f}% available from {most_prevalent_datetime} - imputed and included")
        else:
            print(f"{col}: {availability_percentage:.1f}% available from {most_prevalent_datetime} - filtered out")

# Update the dataframe to only include continuous columns
pivot_hourly_df = filtered_df[continuous_columns]
value_columns = [col for col in value_columns if col in continuous_columns]

print(f"Filtered out columns with missing data. Kept {len(continuous_columns)-1} out of {len(device_columns)} value columns.")
print(f"Data starts from: {pivot_hourly_df['deviceTimestamp'].min()}")
print(f"Data ends at: {pivot_hourly_df['deviceTimestamp'].max()}")

# Rename columns to value_i format starting from 1
new_rename_map = {'deviceTimestamp': 'deviceTimestamp'}
for i, col in enumerate(value_columns, 1):
    new_rename_map[col] = f"value_{i}"

pivot_hourly_df = pivot_hourly_df.rename(columns=new_rename_map)
value_columns = [f"value_{i}" for i in range(1, len(value_columns) + 1)]

# Calculate split index (80% for training, 20% for testing)
split_index = int(0.8 * len(pivot_hourly_df))

# Split the data
train_data = pivot_hourly_df.iloc[:split_index].copy()
test_data = pivot_hourly_df.iloc[split_index:].copy()

print(f"Training data: {len(train_data)} rows")
print(f"Testing data: {len(test_data)} rows")
print(f"Split ratio: {len(train_data)/len(pivot_hourly_df):.2%} train, {len(test_data)/len(pivot_hourly_df):.2%} test")

# Save the split datasets
train_data.to_csv("../data/train_data.csv", index=False)
test_data.to_csv("../data/test_data.csv", index=False)