"""
author: MichaelFeil
"""

import pandas as pd
import numpy as np
import glob
import multiprocessing
import os
import re
from itertools import repeat


# constants
tripduration = "tripduration"
starttime = "starttime"
stoptime = "stoptime"
startstationid = "startstationid"
startstationname = "startstationname"
startstationlatitude = "startstationlatitude"
startstationlongitude = "startstationlongitude"
endstationid = "endstationid"
endstationname = "endstationname"
endstationlatitude = "endstationlatitude"
endstationlongitude = "endstationlongitude"
endstationlongitude = "endstationlongitude"
bikeid = "bikeid"
usertype = "usertype"
birthyear = "birthyear"
gender = "gender"

dtypes_initial = {
    tripduration: pd.Int32Dtype(),
    #"starttime": "string", # is date_columns
    #"stoptime": "string", # is date_columns
    startstationid: pd.Int32Dtype(),
    startstationname: "string",
    startstationlatitude: pd.Float32Dtype(),
    startstationlongitude: pd.Float32Dtype(),
    endstationid: pd.Int32Dtype(),
    endstationname: "string",
    endstationlatitude: pd.Float32Dtype(),
    endstationlongitude: pd.Float32Dtype(),
    bikeid: pd.Int32Dtype(),
    usertype: "category",
    birthyear: pd.Int32Dtype(),
    gender: pd.Int32Dtype(),
}

translation_dict = {
    tripduration: tripduration,
    starttime: starttime, 
    stoptime: stoptime, 
    startstationid: startstationid,
    startstationname: startstationname,
    startstationlatitude: startstationlatitude,
    "start_lat": startstationlatitude,
    startstationlongitude: startstationlongitude,
    "start_lng": startstationlongitude,
    endstationid: endstationid,
    endstationname: endstationname,
    endstationlatitude: endstationlatitude,
    "endlat": endstationlatitude,
    endstationlongitude: endstationlongitude,
    "endlng": endstationlongitude,
    bikeid: bikeid,
    "ride_id": bikeid,
    usertype: usertype,
    "membercasual": usertype,
    birthyear: birthyear,
    gender: gender,
}


# function definitions

def rename_convention(string):
    """rename columns from every csv to same naming convention
    
    string: column name
    author: Michael Feil
    """
    new_str = string.lower().replace(" ","").replace("_","")
    
    # look up for translation of this string
    if new_str in translation_dict:
        return translation_dict[new_str]
    else:
        return new_str
    
def list_working_csvs(folder = "./data_raw/", regex = r'^20((14(09|10|11|12))|((15|16|17|18|19|20)[0-1][0-9]))-citibike-tripdata(.)csv$'):
    """
    list convertable csv paths.
    
    Currently working 201409 until 202012
    author: Michael Feil
    """
    csvs = [
        os.path.realpath(os.path.join(folder, f)) \
        for f in os.listdir(folder) \
        if re.search(regex, f)
    ]
    print(f"listed paths to {len(csvs)} csvs")
    return list(sorted(csvs))

def typecast_csv_to_pd_dataframe(filename: str, allow_buffer: bool = True) -> pd.DataFrame:
    """
    conversion of a raw dataframe to pandas. just typecasting and reindexing, input not modified
    
    filename: name of file to open
    author: Michael Feil
    """    
    try:
        _, file_extension = os.path.splitext(filename)
        
        if file_extension != ".csv":
            raise BaseException(f"filename provided is {filename} ext {file_extension} not a csv file")
                
        if allow_buffer:
            filename_basename = os.path.basename(filename)
            path = f"./data_feature_saved/raw/{filename_basename}.pickle"
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if os.path.exists(path):
                # print("loading dataframe from ", path)
                try:
                    df = pd.read_pickle(path)
                    return df
                except Exception as e:
                    print(f"Exception: loading dataframe from {path} failed due to {e}")
        
        # print("loading unbuffered dataframe from ", filename)
        # read the csv
        df = pd.read_csv(filename, dtype=dtypes_initial)
        
        # rename the csv
        df.rename(rename_convention, axis='columns', inplace=True)
        
        # check everything is included
        columns_required = list(dtypes_initial.keys()) + ["starttime", "stoptime"]
        assert(
            sorted(columns_required) == sorted(list(df.columns))
        ), f"expected  required columns \n {sorted(columns_required)} but got \n {sorted(list(df.columns))}"
        
        # cast the datetypes
        df = df.astype(dtypes_initial)
        
        # change dates format
        df["starttime"] = pd.to_datetime(df["starttime"], infer_datetime_format=True).dt.tz_localize('America/New_York', ambiguous =True).dt.tz_convert('UTC')
        df["stoptime"] = pd.to_datetime(df["stoptime"], infer_datetime_format=True).dt.tz_localize('America/New_York', ambiguous =True).dt.tz_convert('UTC')
        df.set_index('starttime', inplace=True, drop=False)
        
        if allow_buffer:
            # print("saving dataframe to ", path)
            df.to_pickle(path)
        return df

    except Exception as ex:
        raise Exception(f"Unknown error when parsing filename {filename}: {ex}")

def load_concat_multiple_csvs(csvs_list = ["./data_raw/201903-citibike-tripdata.csv", "./data_raw/201702-citibike-tripdata.csv"], allow_buffer: bool = True) -> pd.DataFrame:
    """
    load and concat multiple csvs
    
    csvs: list of csv file locations
    
    author: Michael Feil
    """
    def single_thread():
        dfs = []
        print(f"attemp loading {len(csvs_list)} csvs on {1} cpu process")
        for csv in csvs_list:
            print(f"loading {csv}")
            dfs.append(typecast_csv_to_pd_dataframe(csv, allow_buffer=allow_buffer))
        return dfs
    
    def multi_thread():
        print(f"attemp loading {len(csvs_list)} csvs on {multiprocessing.cpu_count()} cpu processes")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: 
            # multiprocessing
            
            args = zip(csvs_list, repeat(allow_buffer))
            dfs = pool.starmap(
                typecast_csv_to_pd_dataframe,
                args,
            )
        return dfs
    
    if len(csvs_list) < 2:
        dfs = single_thread()
    else:
        try:
            dfs = multi_thread()

        except Exception as e: 
            print("multithread failed. fallback single thread", e)
            dfs = single_thread()
    
    unified_df = pd.concat(dfs, axis=0)
    unified_df.set_index('starttime', inplace=True, drop=False)
    print(f"loaded dataframe of shape {unified_df.shape} from csvs {len(csvs_list)} csvs")
    return unified_df
    
def test_typecast_csv_to_pd_dataframe(csvs: list = [], check_assert=True) -> None:
    """
    verify implementation of typecast_csv_to_pd_dataframe
    author: Michael Feil
    """
        
    for i, csv in enumerate(csvs):
        print(f"{'-'*10}checking {csv}. [ {i+1} / {len(csvs)} ] {'-'*10}")
        
        df = typecast_csv_to_pd_dataframe(csv, allow_buffer=False)
        if not check_assert:
            # test passed
            print(f"checking {csv} done")
            continue
            
        df.sort_index(axis=1, inplace=True)
        df.reset_index(inplace=True, drop=True)
        # new df
        df2 = pd.read_csv(csv)
        df2.sort_index(axis=1, inplace=True)
        df2.reset_index(inplace=True, drop=True)
        
        for column2, column in zip(df2.columns,df.columns):
            if column in ["usertype", "starttime", "stoptime"]:
                continue
            print(f"checking columns {column2} vs {column}")
            pd.testing.assert_series_equal(
                df2[column2], 
                df[column], 
                check_dtype=False, check_names=False, check_index_type=False
            )
        print(f"checking {csv} done")
        continue