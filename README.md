# World Bank Data Forecasting
This is a Python project for downloading and forecasting World Bank data using ARIMA and linear regression models.

## Installation
To install the required packages, run the following command:
```
pip install -r requirements.txt
```

This will install all the necessary packages listed in the requirements.txt file.

## Usage
To run the program, use the following command:
```
python python_test_spgi.py --indicator=<indicator> --country=<country> [--ignore_missing_start_values=<True/False>]
```

Replace <indicator> with the World Bank indicator code (e.g. "NY.GDP.MKTP.CN") and <country> with the ISO2 code of the country (e.g. "afg"). You can also optionally set the --ignore_missing_start_values flag to True or False to ignore missing values at the beginning of the time series.

This will generate forecasts for the specified indicator and country, and save the result as a JSON file named output.json.

## Output
The output data is a list of dictionaries, where each dictionary represents a data point for a specific year. Each dictionary contains the following fields:

- `date`: an integer representing the year of the data point
- `value`: a float representing the value of the data point
- `indicator`: a dictionary containing the `id` and `value` of the World Bank indicator for the data point
- `country`: a dictionary containing the `id` and `value` of the country for the data point
- `is_forecast`: a boolean indicating whether the data point is a forecast or not. If True, the value field represents a forecasted value for the corresponding date and country.
