# Citi Bike Dataset
Utilities for the Analyis of the Citi Bike Dataset.

## 1.1 Installation
create new conda enviroment `dsfm` using anaconda
```
conda env create --file ./environment.yml
```

## 1.2 initial download of files from s3: 
download from https://s3.amazonaws.com/tripdata/index.html i.e. s3//tripdata

in linux commandline:
```
# download all zips
conda activate dsfm
python -m awscli s3 sync s3://tripdata ./data_raw_zip --no-sign-request

# unzip all files starting with ./data_raw_zip/20[.*].zip to /data_raw
# in linux use this command:
unzip -u './data_raw_zip/20*-citibike-tripdata.*zip' -d './data_raw'

```
## 1.3 keeping up to date
periodically updated the dependencies on changes:
```
conda activate dsfm
conda env update --file environment.yml --prune
```

## 2. Usage
Start Jupyter Lab
```
conda activate dsfm
jupyter lab
```

## Licence for students outside the Project Group:
Copy or reuse of any parts of code for Submissions at Technical University of Denmark is strictly not allowed.
