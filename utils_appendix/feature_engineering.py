"""
author: MichaelFeil
"""


# general imports
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import os
from typing import Union
# sunangle
import astropy.coordinates
import astropy.time 
import astropy.units 

# weather
# Import Meteostat library and dependencies
from meteostat import Hourly

#[start pickup]
def pickups_by_hour(df):
    """
    extract hourly sampled from dataframe:
    - pickups per hour for NYC
    - mean age of hourly pickups
    - mean distance of riders 
    - mean duration for a ride
    """
    # selecting targeted features 
    columns_used = ["birthyear", "gender", "feat_distance_trip", "starttime", "stoptime"]
    df_resampled = df.loc[:,columns_used].resample('H')
    
    # computing number of pickups for NYC per hour
    pickups_by_hour = df_resampled["birthyear"].count().rename("pickups_per_h")

    # computing mean birthyear of pickups
    age = df_resampled["birthyear"].mean().rename("feat_pickupmeanbyear")

    # computing pickups splitted by gender
    pickup_gender = df_resampled.gender.value_counts().unstack(level=1).rename(
        columns={0:"feat_pickupgender_unk",1:"feat_pickupgender_m",2:"feat_pickupgender_f"}
    )
    # computing mean distance per timeslot
    distance_trip = df_resampled["feat_distance_trip"].mean().rename("feat_meandistancetrip")
    
    # computing mean trip time driven per hour
    mean_time = (df_resampled["stoptime"].mean() - df_resampled["starttime"].mean()).rename("feat_meantriptime")
    mean_time = mean_time.dt.total_seconds() # to seconds
    
    
    df_pick = pd.concat([pickups_by_hour, pickup_gender, age, mean_time, distance_trip], axis=1)
    
    # fill nan
    df_pick.feat_pickupgender_unk.fillna(0., inplace=True)
    df_pick.feat_pickupgender_m.fillna(0., inplace=True)
    df_pick.feat_pickupgender_f.fillna(0., inplace=True)
    df_pick.feat_meandistancetrip.fillna(df_pick.feat_meandistancetrip.mean(), inplace=True)
    df_pick.feat_pickupmeanbyear.fillna(df_pick.feat_pickupmeanbyear.mean(), inplace=True)
    df_pick.feat_meantriptime.fillna(df_pick.feat_meantriptime.mean(), inplace=True)
    
    # check no more nans
    assert df_pick.loc[df_pick.index.difference(df_pick.dropna().index)].empty 
    
    return df_pick
#[end pickup]

#[start sunangle]
def get_sun_angle_for_row(row) -> float:
    """
    using pd.apply for row based sun angle calculation
    
    author: Michael Feil
    """
    time = astropy.time.Time(row["starttime"], format='datetime', scale='utc')

    loc = coord.EarthLocation(
        lon=row["startstationlatitude"] * u.deg, 
        lat=row["startstationlongitude"] * u.deg
    )

    altaz = coord.AltAz(location=loc, obstime=time)
    current_altitude = get_sun(time).transform_to(altaz).alt.degree
    return current_altitude

def vectorized_get_sun_angle(df: pd.DataFrame) -> float:
    """
    using np vectorization for sun angle calculation for individual trips
    
    author: Michael Feil
    """
    UTC_time = df["starttime"].dt.tz_convert('UTC').values
    # convert array of times (into numpy.datetime64 array)
    time = astropy.time.Time(UTC_time, format='datetime64', scale='utc')
    
    # convert array of long and lat coordiantes (into numpy.float64 array)
    lat = astropy.units.Quantity(df["startstationlatitude"], unit='deg')
    lon = astropy.units.Quantity(df["startstationlongitude"], unit='deg')
    
    
    # convert into list of ITRS Coordinate tuples / arrays
    # e.g. loc is array of locations in space: (2675667.82148666, -9302852.15834796, 8293866.53436309)
    # assume earth is even
    loc = astropy.coordinates.EarthLocation.from_geodetic(
        lon, lat, height=astropy.constants.R_earth
    ).itrs
    
    # get location in Altitude-Azimuth frame
    altaz = astropy.coordinates.AltAz(location=loc, obstime=time)
    
    # get current position of sun on that point on earth
    current_sun_pos = astropy.coordinates.get_sun(time).transform_to(altaz)
    
    # return the degree above the horizon
    return current_sun_pos.alt.degree

def vectorized_get_sun_angle_nyc(
    start_time: datetime.datetime = datetime.datetime(2013, 5, 30), 
    end_time: datetime.datetime = datetime.datetime(datetime.date.today().year, 1, 2),
    lat: float = 40.73,
    long: float =-73.98,
    minutes: int = 60
):
    """

    function:
    get sun angle for each x minutes at (long, lat) coordinates 
    for each time between start to end using np vectorization
    
    author: Michael Feil
    """
    dates = np.arange(start_time, end_time, np.timedelta64(minutes,'m'), dtype='datetime64')
    df = pd.DataFrame(dates, columns=["starttime"])
    df["starttime"] = df["starttime"].dt.tz_localize('UTC')
    df["startstationlatitude"] = lat
    df["startstationlongitude"] = long
    angles = pd.DataFrame(
        vectorized_get_sun_angle(df),
        index = df["starttime"],
        columns = ["feat_sun_angle"]
    )
    
    return angles

#[end sunangle]
#[start weather]

def get_nyc_weather_stats(
    station: str = "74486", # NYC-JFK: 74486, Jersey-Newark: 72502, NYC-Yorkville: KNYC0
    allow_buffer: bool = True
) -> pd.DataFrame:
    """
    get weather from NYC (hourly, statistic data)
    see metrostat docs: https://dev.meteostat.net/python/hourly.html#example
    
        allow_buffer: get from buffered file.
        station: metrostat station, default 74486 -> e.g. https://meteostat.net/en/station/74486
    
    returns dataframe, see https://dev.meteostat.net/python/hourly.html#data-structure:
        Column	Description	Type
        station	The Meteostat ID of the weather station (only if query refers to multiple stations)	String
        time	The datetime of the observation	Datetime64
        temp	The air temperature in °C	Float64
        dwpt	The dew point in °C	Float64
        rhum	The relative humidity in percent (%)	Float64
        prcp	The one hour precipitation total in mm	Float64
        snow	The snow depth in mm	Float64
        wdir	The average wind direction in degrees (°)	Float64
        wspd	The average wind speed in km/h	Float64
        wpgt	The peak wind gust in km/h	Float64
        pres	The average sea-level air pressure in hPa	Float64
        tsun	The one hour sunshine total in minutes (m)	Float64
        coco	The weather condition code	Float64
    """
    
    # check allow_buffer
    if allow_buffer:
        path = f"./data_feature_saved/weather_{station}_pandas.pickle"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            # print("loading dataframe from ", path)
            try:
                weather_nyc = pd.read_pickle(path)
                return weather_nyc
            except Exception as e:
                print(f"Exception: loading dataframe from {path} failed due to {e}")
                            
    
    # Set the time period
    startp = datetime.datetime(2013, 6, 1) # July 2013
    today = datetime.date.today()
    endp = datetime.datetime(today.year, 1, 1)

    # Get hourly data for time period
    data = Hourly(station, startp, endp)
    weather_nyc = data.fetch()
    weather_nyc.index = weather_nyc.index.tz_localize('UTC')
    
    if weather_nyc.empty:
        raise BaseException(
            f"no weather for station {station} found. \n"
            f"Please check internet & station ID if https://meteostat.net/en/station/{station} is available"
        )
    
    if allow_buffer:
        print("saving dataframe to ", path)
        weather_nyc.to_pickle(path)
    
    return weather_nyc

def nyc_weather_feature_frame(
    data = pd.DataFrame, coco_processing: Union["raw", "processed"] = "processed"
) -> pd.DataFrame:
    
    feature_frame = data.copy(deep=True)
    
    # fillNaNs forward (use last published weather)
    feature_frame.coco.ffill(inplace=True) 
    
    feature_frame = feature_frame.drop(columns=[
        "dwpt", # dew point can be kept from humidity
        'snow', # snow not available
        'wdir', # wind direction not important
        "wpgt", # Wind Peak Gust not important
        "pres", # pressure nmopt imporant
        "tsun", # tsun not available
    ]) 
    
    if coco_processing == "raw":
        full_dummies = pd.get_dummies(feature_frame.coco, dummy_na=True, prefix="coco")
        feature_frame = pd.concat(
            [
                feature_frame, 
                full_dummies
            ], axis=1
        )       
        
    elif coco_processing == "processed":
        feature_frame["c_excellent_weather"] = feature_frame.coco.isin([1., 2.]) # >9. is also known as stay at home weather.
        feature_frame["c_cloudy"] = feature_frame.coco.isin([3., 4.])
        feature_frame["c_fog"] = feature_frame.coco.isin([5., 6.])
        feature_frame["c_rain"] = feature_frame.coco.isin([7., 8., 9., 10., 11., 17., 18.])
        feature_frame["c_snow"] = feature_frame.coco.isin([12., 13., 14., 15., 16., 19., 20., 21., 22.,])
        feature_frame["c_storm"] = feature_frame.coco.isin([23., 24., 25., 26., 27.])
        """
        partial_dummies = pd.get_dummies(feature_frame.coco, dummy_na=True, prefix="coco")
        feature_frame = pd.concat(
            [
                feature_frame, 
                partial_dummies
            ], axis=1
        )  
        """
    
    
    feature_frame = feature_frame.drop(columns=[
        "coco", # is now dummy var
    ]) 
    feature_frame.temp.ffill(inplace=True, limit=48) 
    feature_frame.rhum.ffill(inplace=True, limit=48) 
    feature_frame.wspd.ffill(inplace=True, limit=48) 
    feature_frame.prcp.ffill(inplace=True, limit=48) 
        
    feature_frame.columns = [f"feat_w_{c}" for c in feature_frame.columns]
    
    return feature_frame
#[end weather]

#[start date feature]
def features_date_transform(dates_timeseries: pd.Series):
    """
    dates_timeseries:
        pd.series[datetime[ns, UTC]]
    
    return pd.DataFrame[]
    """
    local_time_series = dates_timeseries.dt.tz_convert('America/New_York')
    # day of dayoftheyear as sin transfrom
    dayoftheyear = local_time_series.dt.dayofyear
    # day as categorical variable from 1 to 7
    feature_dayoftheweek = local_time_series.dt.dayofweek
   
    minuteofday = local_time_series.dt.hour * 60 + local_time_series.dt.minute
    # us holidays
    holidays = USFederalHolidayCalendar().holidays(
        start=local_time_series.min(), end=local_time_series.max()\
    )
    # bool variable
    # check if current time in US is a holiday
    feature_isholiday = local_time_series.dt.floor('d').isin(holidays)
    
    data = [
        dayoftheyear.rename("ana_dayoftheyear"),
        feature_dayoftheweek.rename("feat_dayoftheweek"), 
        minuteofday.rename("ana_minuteofday"), 
        feature_isholiday.rename("feat_isholiday")
    ]
    
    data += feature_cont_transform(dayoftheyear, 366, "feat_dayoftheyear")
    data += feature_cont_transform(minuteofday, 24*60, "feat_minuteofday")
        
    return pd.concat(
        objs = data,
        axis = 1   
    )    
    
def feature_cont_transform(feature_vector: pd.Series, period_lenght, name):
    """transform to continous variable over period_lenght.
    
    following best practice for datetime encoding in ML:
    https://stats.stackexchange.com/questions/311494/best-practice-for-encoding-datetime-in-machine-learning
    """
    sin = np.sin((2*np.pi/period_lenght) * feature_vector).rename(name+"_sin")
    cos = np.cos((2*np.pi/period_lenght) * feature_vector).rename(name+"_cos")
    return [sin, cos]
#[end date feature]

#[start geospatial analysis]

def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=astropy.constants.R_earth.value/1000):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))

#[end geospatial analysis]