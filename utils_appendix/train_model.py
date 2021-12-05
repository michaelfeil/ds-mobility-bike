"""
author: MichaelFeil
docs: Alvaro, MichaelFeil
"""

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import timedelta
import pandas as pd
import numpy as np
from typing import Union, List
from . import feature_engineering
from sklearn import preprocessing 
import warnings
import math
np.random.seed(42)

class ForwardPredicter:
    """Create dataframes to train and test to train/evaluate any model where the data is hourly evenly spaced,
    meaning that each row for the dataframe represents 1hour

    Attributes:
        - duration_rows_past: An integer representing the number of past hours to include for training the model
        - columns_past_features: List of column names (string) to add to our model input. during duration of "past"
        - duration_rows_past: An integer representing the number of past hours to include for training the model
        - sideline_fut_features: List of column names (string) to add to our model input. during duration of rows "futurte"
        - use_scaler: Sklearn Scaler, for a proper standarization of the data, in case needed
        - verbose: 0=Info, 1=warnings, 2=Error
        
        Available column names are:
            ['pickups_per_h', 'ana_dayoftheyear', 'feat_dayoftheweek',
           'ana_minuteofday', 'feat_isholiday', 'feat_dayoftheyear_sin',
           'feat_dayoftheyear_cos', 'feat_minuteofday_sin', 'feat_minuteofday_cos',
           'feat_sun_angle', 'feat_w_temp', 'feat_w_rhum', 'feat_w_prcp',
           'feat_w_wspd', 'feat_w_c_excellent_weather', 'feat_w_c_cloudy',
           'feat_w_c_fog', 'feat_w_c_rain', 'feat_w_c_snow', 'feat_w_c_storm']
    """
    def __init__(self, duration_rows_past = 48, duration_forecast = 7 * 24, columns_past_features: List[str] = [], use_scaler: Union[None,preprocessing.StandardScaler]  = preprocessing.StandardScaler, sideline_fut_features: List[str] = [], verbose: int = 0):
        """train frame cropped to allowed dates"""
        pd.options.mode.chained_assignment = None 
        # check that train_frame and test_frame are only 1 hour appart
        self.duration_rows_past = duration_rows_past # (48, n_features_input)
        self.duration_forecast = duration_forecast # (156,1) array to predict
        if use_scaler is not None:
            self.scaler = use_scaler()
        else:
            self.scaler = None
        self.columns_past_features = list(set(columns_past_features + ["pickups_per_h"]))
        self.sideline_fut_features = sideline_fut_features
        self.verbose = verbose
        if self.verbose < 1:
            print(f"columns_past_features: {self.columns_past_features}. columns_past_features: {self.columns_past_features}")
        
    def create_x_y_work(self, train_series: pd.Series, append_index: Union[None, pd.DatetimeIndex] = None, hours_between_samples: int = 24):
        """
        Return 3 nunmpy arrays:
            - X: standarized data with the independent variables
            - X_orig: X without standarization
            - y: the target value
        
        slice into self.duration_rows_past, self.duration_forecast and create X, y targets
        """
        X = []
        y = []
        
        # print(self.train_frame.head())
        # print(len(self.test_pickupsperh.index)/24)
        frame = self._get_frame_for_series(series = train_series, append_index=append_index)
        frame_w_o_target = frame.loc[:, frame.columns != 'pickups_per_h']
        itera = range(
            0, len(frame) - (self.duration_rows_past + self.duration_forecast), hours_between_samples
        )
        if self.verbose < 1:
            itera = tqdm(itera, desc="Train samples")
        for i in itera:
            # lagged stuff (I=0)= is 0 - 47, (I=1)= is 1 - 48
            slice_x = slice(i,i + self.duration_rows_past) 
            # (I=0)= is 48 - 72, (I=1)= is 49 - 73 .. - 72
            slice_y = slice(i + self.duration_rows_past, i + self.duration_rows_past + self.duration_forecast)
            
            x_stack, y_stack = self.get_x_y_for_slices(frame, frame_w_o_target, slice_x, slice_y)
            
            X.append(x_stack), y.append(y_stack) 
        
        X = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float) 
        assert not np.isnan(np.sum(X))
        assert not np.isnan(np.sum(y_array))
        # assert np.unique(X, axis=0).shape == X.shape # stupid test
        if y_array.shape[1] > 2:
            pass
            # assert np.unique(y_array, axis=0).shape == y_array.shape # stupid test
        X_orig = X
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        if self.verbose < 1:
            print(f"shape of X is {X.shape}, y {y_array.shape}")
        return X, X_orig, y
    
    def get_x_y_for_slices(self, frame, frame_w_o_target, slice_x, slice_y):
        x_stack = frame.iloc[slice_x].values.flatten() 
        
        if self.sideline_fut_features:
            x_stack = np.hstack(
                [x_stack, frame_w_o_target.iloc[slice_y].values.flatten()]
            ).flatten() # add additional features
        y_stack = frame.pickups_per_h[slice_y]
        return x_stack, y_stack
    
    @staticmethod
    def frame_w_o_pickups(frame):
        """Return frame without target pickups_per_h"""
        return frame.loc[:, frame.columns != 'pickups_per_h']
    
    def _get_frame_for_series(self, series: pd.Series, append_index: Union[None, pd.DatetimeIndex] = None):
        """
        creates from series with lenght (n,1) a feature frame shape (n,1+k_features) 
        
        if append_index:
            creates from series with lenght (n,1) AND append_index with (m,1) indices
            a feature frame shape (n+m,1+k_features), where the first n rows have a value 
            
        Attributes:
            - series
            - append_index
        """
        if append_index is not None:
            assert series.index[-1] + timedelta(hours=1) == append_index[0]
            helper = append_index.to_series(name='pickups_per_h')
            helper[:] = "to_predict"
            
            series = series.append(helper)
        
        df_time = feature_engineering.features_date_transform(series.index.to_series())
        
        frame = series.to_frame().join(
            [df_time]
        )
        
        # sun angle
        if "feat_sun_angle" in self.columns_past_features + self.sideline_fut_features:
            angles =  feature_engineering.vectorized_get_sun_angle_nyc(
                start_time = series.index[0], end_time = series.index[-1] + timedelta(hours=1)
            )
            frame = frame.join(
                [angles]
            )
        
        # weather data
        if any(colu.startswith("feat_w_") for colu in self.columns_past_features + self.sideline_fut_features):
            if self.verbose < 2:
                warnings.warn("Data Leakage Warning. You are using Weather Report data (feat_w_* columns).  Weather Report relies on recorded historic data, and therefore limits the assumption of a actual Forecasting Model", UserWarning)
            weather_raw = feature_engineering.get_nyc_weather_stats()
            weather = feature_engineering.nyc_weather_feature_frame(weather_raw)
            frame = frame.join(
                [weather]
            )
       
        assert frame.index.difference(frame.dropna().index).empty
        frame.replace("to_predict", np.nan, inplace=True)
        assert all([(colName in frame.columns) for colName in self.columns_past_features]), f"columns_past_features {self.columns_past_features} are not in frame columns {frame.columns}"
        assert all([(colName in frame.columns) for colName in self.columns_past_features]), f"sideline_fut_features {self.sideline_fut_features} are not in frame columns {frame.columns}"
        
        frame = frame.loc[:, self.columns_past_features]
        
        return frame
        
    def get_frame_for_predict(self, previous_pickups: pd.Series, predict_hours: int):
        """
        int predict_hours needs to be multiple of model output / self.duration_forecast
        """
        assert type(previous_pickups) == pd.Series
        assert type(predict_hours) == int
        if predict_hours % self.duration_forecast != 0:
            predict_hours_n = math.ceil(float(predict_hours)/self.duration_forecast)*self.duration_forecast
            if self.verbose < 2:
                warnings.warn(f"need to forecast multiples of {self.duration_forecast}, but got {predict_hours}. upcasting len to {predict_hours_n} ")
            predict_hours = predict_hours_n
            
            
            
        predict_index = pd.date_range(
            start=previous_pickups.index[-1] + timedelta(hours=1), 
            end=previous_pickups.index[-1] + timedelta(hours=predict_hours ),  freq="H"
        )
        frame = self._get_frame_for_series(series=previous_pickups, append_index=predict_index)
        return frame
    
    def repeat_predict_unknown_frame_forward(self, frame, classifier, predict_hours: int):
        """
        given a frame with missing target values fill forward 
        
        input:
            classifier: class which returns np.array with self.duration_forecast elements on classifier.predict()
            frame: dataframe with all columns used as input for model
            predict_hours: how many hours to use for y_pred
        
        returns:
           y_predicted: series of len(frame)-self.duration_rows_past
           frame: frame filled with predictions
        """
        frame_w_o_target = ForwardPredicter.frame_w_o_pickups(frame)
        assert frame_w_o_target.index.difference(frame_w_o_target.dropna().index).empty
        assert (len(frame) - self.duration_rows_past) % self.duration_forecast == 0
        assert frame[:self.duration_rows_past].index.difference(frame[:self.duration_rows_past].dropna().index).empty, f"nans in {frame[:self.duration_rows_past]}"
        
        range_frame = range(
            0, # start
            len(frame)-self.duration_rows_past, # stop
            self.duration_forecast # step
        )
        if self.verbose < 1:
            range_frame = tqdm(range_frame, desc=f"Forecast of {self.duration_forecast}h steps, to reach {len(frame)-self.duration_rows_past}")
        
        for i in range_frame:
            # lagged stuff (I=0)= is 0 - 47, (I=1)= is 1 - 48
            slice_x = slice(i,i + self.duration_rows_past) 
            # (I=0)= is 48 - 72, (I=1)= is 49 - 73 .. - 72
            slice_y = slice(i+ self.duration_rows_past, i + self.duration_rows_past + self.duration_forecast)
            
            X, y = self.get_x_y_for_slices(frame, frame_w_o_target, slice_x, slice_y)
            assert np.isnan(y.values).all(), f"X value are {frame.iloc[slice_x]} \n Y value are {frame.iloc[slice_y]}"
            # print(f"predicting {y.index.values} with array {np.asarray(X).shape}")
            X =np.expand_dims(X, axis=0)
            
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            y_pred = classifier.predict(X)
            # fill with predictions
            
            frame.loc[:,"pickups_per_h"].iloc[slice_y] = y_pred.flatten()
    
        return frame.pickups_per_h[self.duration_rows_past:self.duration_rows_past+predict_hours], frame