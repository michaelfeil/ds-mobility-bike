"""
author: Clara, Louis, Alvaro
"""

import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import numpy as np

def configure_violinplot_timewise(some_df, by="dayofweek", rushhour=True, ax=None):
    """create pcickups grouped over weekday"""
    some_df = some_df.copy()
    data = {}
    args = {"y": "pickups_per_h", "ax":ax}
    some_df.index = some_df.index.tz_convert('America/New_York')
    
    if rushhour:
        some_df = some_df.loc[some_df.index.hour.to_series().between(5.99,18.01).to_numpy()]
        rush = some_df.index.hour.to_series()
        # between 6-9h or 15-18h US timezone
        rush = rush.between(5.99,9.01) | rush.between(14.99,18.01)
        data.update({"rushhour": rush.to_numpy()})
        args.update({"hue":"rushhour", "split":True})
        
    if by == "dayofweek":
        data.update({'dayofweek':some_df.index.dayofweek.to_numpy(), 'pickups_per_h': some_df.pickups_per_h})
        args.update({"x":"dayofweek"})
        
    elif by == "month":
        data.update({'month':some_df.index.month.to_numpy(), 'pickups_per_h': some_df.pickups_per_h})
        args.update({"x":"month"})
    
    args.update({"data":pd.DataFrame(
        data,
        index = some_df.index
    )})
    
    return args

def temperature_plot(full_df,sliced):
    """LOUIS ADD SOME DOCU"""
    plt.title("Number of pickups at noon as a function of temperature")
    return plt.scatter(full_df.loc[sliced, "feat_w_temp"], full_df.loc[sliced, "pickups_per_h"],s=7.5)

def yearly_hour_plot(full_df):
    """
    Plot of the pickups_per_h vs yearly hours. Every point is labelled within 0-23h
    """
    h_day = np.array([d.hour for d in full_df.index])

    scat = plt.scatter(range(len(full_df)) , full_df['pickups_per_h'], c=h_day, s=2)
    plt.ylabel('pickups_per_h')
    plt.xlabel('yearly hours')
    plt.legend(*scat.legend_elements(),
               loc="center left", bbox_to_anchor=(1, 0.5), title="hours")

def quantiles_plot(full_df):
    """
    """
    stress90 = np.percentile(full_df['pickups_per_h'], 90)
    stress50 = np.percentile(full_df['pickups_per_h'], 50)
    stress10 = np.percentile(full_df['pickups_per_h'], 10)

    # print(f"Pickups above the 'stress' level (90%): {len(full_df[full_df['pickups_per_h'] > stress90])}")
    # print(f"Pickups above the 'stress' level (50%): {len(full_df[full_df['pickups_per_h'] > stress50])}")
    # print(f"Pickups above the 'stress' level (10%): {len(full_df[full_df['pickups_per_h'] > stress10])}")

    ## PLOTS
    # We go from 0 to 24h and we will go one by one
    step = 1
    hours = np.arange(0, 24, step)
    # feature_engineering.features_date_transform(full_df.index.to_series())  # --> hour_of_day not retrieved
    full_df['hour_of_day'] = np.array([d.hour for d in full_df.index])

    q10_line = [full_df.loc[full_df['hour_of_day'] == h]['pickups_per_h'].quantile(.05) for h in hours]
    q90_line = [full_df.loc[full_df['hour_of_day'] == h]['pickups_per_h'].quantile(.9) for h in hours]
    mean_line = [full_df.loc[full_df['hour_of_day'] == h]['pickups_per_h'].quantile(.5) for h in hours]

    plt.plot(hours, q10_line, label='quantile 10', linestyle='--', linewidth=2, color='red')
    plt.plot(hours, mean_line, label='average', linewidth=2, color='blue')
    plt.plot(hours, q90_line, label='quantile 90', linestyle='--', linewidth=2, color='green')
    plt.xlabel('hour')
    plt.xticks(range(0,24,1))
    plt.xlim((0,23))
    plt.ylabel('pickups_per_h')
    plt.legend(loc='best', fontsize=20)

def weather_plot(full_df,sliced):
    """LOUIS ADD SOME DOCU"""
    sliced = full_df.loc[sliced]
    columns_weather = pd.Series(sliced.columns).str.contains('feat_w_c_').astype("bool").values
    we = pd.concat([sliced.loc[:,columns_weather].rename(columns=lambda x: x.replace("feat_w_c_","")) \
                                                    .idxmax(axis=1).rename("weather_typology"), sliced.pickups_per_h],axis=1)
    return sns.violinplot(x="weather_typology", y="pickups_per_h", data=we)

def sun_angle_plot(full_df):
    """LOUIS ADD SOME DOCU"""
    plt.title("Number of pickups at noon as a function of the sun angle")
    return  plt.scatter(full_df.loc[:,"feat_sun_angle"], full_df.loc[:,"pickups_per_h"],s=7.5)

def plots(full_df, width=30, height=40):
    """
    By default, plot size is 30inches (width) x 40inches (heigth)
    """
    slice_noon = pd.date_range(start="2014-09-01 12:00:00", end="2020-12-31 12:00:00", freq="D", tz="US/Eastern")
    slice_8am = pd.date_range(start="2014-09-01 08:00:00", end="2020-12-31 08:00:00", freq="D", tz="US/Eastern")
    slice_full_us = pd.date_range(start="2014-09-01 00:00:00", end="2020-12-31 23:00:00", freq="h", tz="US/Eastern")
    
    fig3 = plt.figure(figsize=(width, height))
    gs = fig3.add_gridspec(28, 8)
    
    rows_interval = 4  
    row0 = 0
    
    # First row will be the scatter yearly_hour and quantile plots
    fig3.add_subplot(gs[row0:rows_interval, :4])
    yearly_hour_plot(full_df)
    
    fig3.add_subplot(gs[row0:rows_interval, 4:])
    quantiles_plot(full_df)
    
    row0+=rows_interval
            
    rowF = row0 + rows_interval
    fig3.add_subplot(gs[row0:rowF, :4])
    weather_plot(full_df,slice_8am)
    plt.title('Weather vs Pickups at 8am')

    fig3.add_subplot(gs[row0:rowF, 4:])
    sun_angle_plot(full_df) 
    
    row0+=rows_interval
    rowF = row0 + 2*rows_interval
    fig3.add_subplot(gs[row0:rowF, :4])
    weather_plot(full_df,slice_noon)
    plt.title("Weather vs Pickups at noon")

    fig3.add_subplot(gs[row0:(rowF-rows_interval), 4:])
    temperature_plot(full_df,slice_8am)
    plt.title('Temperature vs Picups at 8am') 

    fig3.add_subplot(gs[(rowF-rows_interval):rowF, 4:])
    temperature_plot(full_df,slice_noon)
    plt.title('Temperature vs Pickups at noon')
    row0 = rowF
    
    rowF = row0 + rows_interval

    fig3.add_subplot(gs[row0:rowF, :4])
    weather_plot(full_df,slice_full_us)
    plt.title('Weather vs All Pickups')

    fig3.add_subplot(gs[row0:rowF, 4:])
    temperature_plot(full_df,slice_full_us)
    plt.title('Temperature vs All Pickups')
    
    
    
    
    
    print(row0)
    fig3.add_subplot(gs[20:24, :])
    sns.violinplot(**configure_violinplot_timewise(full_df.loc[slice_full_us], by="dayofweek"))
    
    fig3.add_subplot(gs[24:28, :])
    sns.violinplot(**configure_violinplot_timewise(full_df.loc[slice_full_us], by="month", rushhour=False));

    # Add some padding
    fig3.subplots_adjust(hspace=60, wspace=5)




