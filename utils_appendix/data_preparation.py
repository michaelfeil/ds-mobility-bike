"""
author: Louis, MichaelFeil
"""

import pandas as pd
from utils_appendix import load_bikeshare_frames
from utils_appendix import feature_engineering

def prep(csvs, pickups_only = False):
    """
    Louis:
    
    load csvs from database
    pickups_only: speed up when only pickups needed
    """
    df = load_bikeshare_frames.load_concat_multiple_csvs(csvs)
    df["feat_distance_trip"] = feature_engineering.haversine(
            lat1 = df.startstationlatitude.values.astype(float),
            lon1 = df.startstationlongitude.values.astype(float),
            lat2 = df.endstationlatitude.values.astype(float),
            lon2 = df.endstationlongitude.values.astype(float), 
            to_radians=True
        )
    
    pickups = feature_engineering.pickups_by_hour(df)
    
    if not pickups_only:
        df_time = feature_engineering.features_date_transform(pickups.index.to_series())
        weather_raw = feature_engineering.get_nyc_weather_stats()
        weather = feature_engineering.nyc_weather_feature_frame(weather_raw)
        angles =  feature_engineering.vectorized_get_sun_angle_nyc()
        full_df = pickups.join([df_time,weather,angles])
    else:
        # only pickups are needed
        full_df = pickups
    columns = pd.Series(full_df.columns).str.contains('[^pickups_per_h]').astype("bool").values
    
    # fill missing values
    full_df.loc[:, columns] = full_df.loc[:, columns].ffill(limit=4).bfill(limit=4)
    return full_df