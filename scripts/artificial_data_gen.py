import pandas as pd
import numpy as np
from datetime import datetime

def generate_single_consumption_series(start_date, end_date, weekday_base=95000, weekend_base=55000, 
                                     seasonal_amplitude=0.10, peak_amplitude=0.15, 
                                     random_range=(0.95, 1.05), seed=42, seasonal_offset=0):
    """
    Generate a single consumption data series.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        weekday_base: Base daily consumption for weekdays
        weekend_base: Base daily consumption for weekends
        seasonal_amplitude: Amplitude of seasonal variation
        peak_amplitude: Amplitude of peak hour variations
        random_range: Tuple of (min, max) for random factor
        seed: Random seed for reproducibility
        seasonal_offset: Phase offset for seasonal patterns (in days)
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    
    # Generate base consumption pattern
    base_consumption = []
    
    for date in dates:
        if date.weekday() < 5:
            base_daily = weekday_base
        else:
            base_daily = weekend_base
        
        base_hourly = base_daily / 24
        
        hour = date.hour
        weekday_threshold = 5 + np.random.uniform(-0.2, 0.2)
        
        if date.weekday() < weekday_threshold:
            morning_peak = peak_amplitude * np.sin(2 * np.pi * (hour - 2 + seasonal_offset/24) / 12) if hour <= 12 else 0
            evening_peak = peak_amplitude * np.sin(2 * np.pi * (hour - 14 + seasonal_offset/24) / 12) if hour >= 12 else 0
            hourly_factor = 1.0 + morning_peak + evening_peak
        else:
            hourly_factor = 1.0
        
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1 + seasonal_amplitude * np.sin(2 * np.pi * (day_of_year - 350 + seasonal_offset) / 365)
        
        base_consumption.append(base_hourly * hourly_factor * seasonal_factor)
    
    # Apply random factors
    np.random.seed(seed)
    random_factors = np.random.uniform(random_range[0], random_range[1], len(base_consumption))
    
    series_values = np.array(base_consumption) * random_factors
    
    return pd.DataFrame({
        'deviceTimestamp': dates,
        'value': np.round(series_values, 2)
    })

def generate_multiple_series(start_date, end_date, num_series=5, series_variation=(-0.1, 0.1), 
                           seasonal_offset_range=(-30, 30), base_params=None):
    """
    Generate multiple consumption series with varied parameters.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        num_series: Number of series to generate
        series_variation: Tuple of (min, max) variation range for parameters
        seasonal_offset_range: Tuple of (min, max) days for seasonal offset variation
        base_params: Base parameters dict (optional)
    """
    if base_params is None:
        base_params = {
            'weekday_base': 95000,
            'weekend_base': 55000,
            'seasonal_amplitude': 0.10,
            'peak_amplitude': 0.15,
            'random_range': (0.95, 1.05),
            'seasonal_offset': 0
        }
    
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    df_data = {'deviceTimestamp': dates}
    
    for i in range(num_series):
        # Vary parameters for each series
        varied_params = base_params.copy()
        varied_params['weekday_base'] *= (1 + np.random.uniform(*series_variation))
        varied_params['weekend_base'] *= (1 + np.random.uniform(*series_variation))
        varied_params['seasonal_amplitude'] *= (1 + np.random.uniform(*series_variation))
        varied_params['peak_amplitude'] *= (1 + np.random.uniform(*series_variation))
        varied_params['seasonal_offset'] = np.random.uniform(*seasonal_offset_range)
        varied_params['seed'] = 42 + i
        
        # Generate single series
        single_series = generate_single_consumption_series(start_date, end_date, **varied_params)
        df_data[f'value_{i+1}'] = single_series['value']
    
    return pd.DataFrame(df_data)

# Generate the data
start_date = datetime(2019, 1, 1)
end_date = datetime(2023, 12, 31)

df = generate_multiple_series(start_date, end_date, series_variation=(-0.5, 0.5), seasonal_offset_range=(-30, 30), num_series=5)
df.to_csv('../data/gen_data.csv', index=False)