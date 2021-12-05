"""
author: MichaelFeil
"""

from sklearn import tree, neighbors, svm, linear_model, ensemble, neural_network, multioutput, preprocessing, metrics, model_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import optuna

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly
plotly.io.renderers.default = 'iframe' # "jupyterlab" #'iframe' #  'notebook' # 'colab' 
plotly.offline.init_notebook_mode()
optuna.logging.set_verbosity(optuna.logging.WARNING)

from importlib import reload
from . import train_model
reload(train_model)

np.random.seed(9)
sampler = optuna.samplers.TPESampler(seed=9) 

DATEONLY_COLUMNS =  [
    'feat_dayoftheweek',
           'feat_dayoftheyear_sin',
           'feat_dayoftheyear_cos', 'feat_minuteofday_sin', 'feat_minuteofday_cos',
]

PERMISSIBLE_COLUMNS = DATEONLY_COLUMNS + [
           'feat_isholiday', 
           'feat_sun_angle'
]

WEATHER_COLUMNS = PERMISSIBLE_COLUMNS + ['feat_w_temp', 'feat_w_rhum', 'feat_w_prcp',
           'feat_w_wspd', 'feat_w_c_excellent_weather', 'feat_w_c_cloudy',
           'feat_w_c_fog', 'feat_w_c_rain', 'feat_w_c_snow', 'feat_w_c_storm'
                                    ]
SCALER_SELECTION = {"None": None,
                    "preprocessing.StandardScaler": preprocessing.StandardScaler, 
                    "preprocessing.RobustScaler": preprocessing.RobustScaler, 
                    "preprocessing.MinMaxScaler":preprocessing.MinMaxScaler 
}
MODEL_SELECTION = {
            "tree.DecisionTreeRegressor()": tree.DecisionTreeRegressor(),
            "neighbors.KNeighborsRegressor(n_neighbors=2)": neighbors.KNeighborsRegressor(n_neighbors=2),
            "ensemble.BaggingRegressor(tree.DecisionTreeRegressor())": ensemble.BaggingRegressor(tree.DecisionTreeRegressor()), 
            "ensemble.ExtraTreesRegressor()": ensemble.ExtraTreesRegressor(),
            "ensemble.RandomForestRegressor()": ensemble.RandomForestRegressor()
}

class Objective(object):
    def __init__(self, train_series, val_series, test_series, tune_columns_past_features, tune_sideline_fut_features, tune_duration_rows_past, tune_duration_forecast, n_predict_hours: int = 24 * 7):
        # Hold this implementation specific arguments as the fields of the class.
        self.train_series = train_series
        self.val_series = val_series
        self.test_series = test_series
        self.tune_columns_past_features = tune_columns_past_features
        self.tune_sideline_fut_features = tune_sideline_fut_features
        self.tune_duration_rows_past = tune_duration_rows_past
        self.tune_duration_forecast = tune_duration_forecast
        self.n_predict_hours = n_predict_hours
        
        
    def tune_column_used(self, trial, columns: List[str], tune_name: str):
        col_used = []
        for col in columns:
            res = trial.suggest_categorical(f"use_{tune_name}_{col}", [True, False])
            if res:
                col_used.append(col)
        return col_used
        
    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        # sample parameters
        used_columns_past_features = self.tune_column_used(trial, self.tune_columns_past_features, "past")
        used_sideline_fut_features = self.tune_column_used(trial, self.tune_sideline_fut_features, "future")
        
        scaler_select = trial.suggest_categorical(f"scaler", list(SCALER_SELECTION.keys()))
        scaler = SCALER_SELECTION[scaler_select]
        
        model_select = trial.suggest_categorical(f"model", list(MODEL_SELECTION.keys()))
        model = MODEL_SELECTION[model_select]
                
        duration_rows_past = trial.suggest_int(f"duration_rows_past", **self.tune_duration_rows_past)
        duration_forecast = trial.suggest_int(f"duration_forecast", **self.tune_duration_forecast)
        
        # get dataset
        FP = train_model.ForwardPredicter(duration_rows_past = duration_rows_past, duration_forecast=duration_forecast, 
            columns_past_features = used_columns_past_features, # features used during duration_rows_past
            sideline_fut_features = used_sideline_fut_features, # features sidelined while duration_forecast, none
            use_scaler = scaler,
            verbose = 2
        )

        # preprocess data from train series on ForwardPredicter|
        X_train, X_unscaled, y_train = FP.create_x_y_work(
            train_series = self.train_series, hours_between_samples=1
        )
                
        clf = model.fit(X_train, np.squeeze(y_train))
        slice_test_load = slice(None,FP.duration_rows_past)
        slice_test_eval = slice(FP.duration_rows_past,FP.duration_rows_past+self.n_predict_hours)
        slice_val_load =  slice(-(self.n_predict_hours+FP.duration_rows_past),-self.n_predict_hours)
        slice_val_eval =  slice(-self.n_predict_hours,None)
        # val
        X_val_frame = FP.get_frame_for_predict(
            previous_pickups = self.val_series[slice_val_load], 
            predict_hours = self.n_predict_hours
        )
        y_val_true = self.val_series[slice_val_eval]
        
        y_pred_val, _ = FP.repeat_predict_unknown_frame_forward(X_val_frame, model, predict_hours = self.n_predict_hours)
        pd.testing.assert_index_equal(y_pred_val.index, y_val_true.index, check_names=False)
        r2_val = metrics.r2_score(y_val_true, y_pred_val)
        mae_val =  metrics.mean_absolute_error(y_val_true, y_pred_val)
        pd.testing.assert_index_equal(y_pred_val.index, y_val_true.index, check_names=False)
        
        # test
        X_test_frame = FP.get_frame_for_predict(
            previous_pickups = self.test_series[slice_test_load], 
            predict_hours = self.n_predict_hours
        )
        y_test_true = self.test_series[slice_test_eval]
        
        y_pred_test, _ = FP.repeat_predict_unknown_frame_forward(X_test_frame, model, predict_hours = self.n_predict_hours)
        pd.testing.assert_index_equal(y_pred_test.index, y_test_true.index, check_names=False)
        r2_test = metrics.r2_score(y_test_true, y_pred_test)
        mae_test =  metrics.mean_absolute_error(y_test_true, y_pred_test)
        

        trial.set_user_attr("r2_val", r2_val)
        trial.set_user_attr("mae_val", mae_val)
        trial.set_user_attr("r2_test", r2_test)
        trial.set_user_attr("mae_test", mae_test)
        
        return r2_val

    
def tune(train_series, val_series, test_series, 
     tune_columns_past_features, 
     tune_sideline_fut_features,
     tune_duration_rows_past = {"low":1, "high": 24},
     tune_duration_forecast = {"low":1, "high": 24},
     n_trials=100,
     n_predict_hours=24*7,
    ):
    """tuning hyperparameters"""
    print(
        f"starting hyperparameter optimization with {n_trials} trials \n"
        f"using series for train {train_series.head(2)}\n,"
        f"measureing performance by predicting test/val for duration of {n_predict_hours}\n"
        f"tuning the columns_past_features {tune_columns_past_features}\n"
        f"tuning the sideline_fut_features {tune_sideline_fut_features}\n"
        f"tunes the int duration_rows_past {tune_duration_rows_past}\n"
        f"tunes the int duration_forecast {tune_duration_forecast}\n"
    )
        
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(
        Objective(
            train_series, val_series, test_series, tune_columns_past_features, tune_sideline_fut_features, tune_duration_rows_past, tune_duration_forecast, n_predict_hours
        ),
        n_trials=n_trials, timeout=600
    )
    print(f"{'='*20} hyperparameter optimaization with {n_trials} trials is done {'='*20} \n")
    
    print(f"\t best trial has params:")
    for k, v in dict(study.best_trial.params).items():
        if type(v) is float:
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"\t And results: ")
    for k, v in dict(study.best_trial.user_attrs).items():
        if type(v) is float:
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
          
    optuna.visualization.matplotlib.plot_optimization_history(study)
    return study

# plotting

