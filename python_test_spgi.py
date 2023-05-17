import json

import fire
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA


def download_data(indicator: str, country: str):
    # Construct the URL with the specified country and indicator
    url = f"http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json"

    # Make the initial request and check the status code
    response = requests.get(url)
    if response.status_code != 200:
        # Raise an exception with the status code and reason
        raise Exception(
            f"Request failed with status code {response.status_code}: {response.reason}"
        )

    # Parse the JSON response and add the data from the first page to the list
    data = response.json()[1]

    # Check if there are more pages of data
    while response.json()[0]["pages"] > response.json()[0]["page"]:
        # Make a request for the next page of data and check the status code
        response = requests.get(url + f"&page={response.json()[0]['page']+1}")
        if response.status_code != 200:
            # Raise an exception with the status code and reason
            raise Exception(
                f"Request failed with status code {response.status_code}:"
                f" {response.reason}"
            )

        # Add the data from the next page to the list
        data.extend(response.json()[1])

    # Raise an exception if no data was found
    if not data:
        raise ValueError("No data found for the specified country and indicator")

    # Return the complete list of data
    return data


def prepare_data_for_forecasting(
    data: list, ignore_missing_start_values: bool
) -> pd.DataFrame:
    df = pd.DataFrame(data)[["date", "value", "indicator", "country"]]

    # Sort the df by date in ascending order
    df.sort_values("date", inplace=True)

    # Drop NaNs at the beginning of the DataFrame
    if ignore_missing_start_values:
        first_valid_index = df["value"].first_valid_index()
        if first_valid_index is not None:
            df = df.loc[first_valid_index:]

    # Fill NaN values
    df["value"].interpolate(method="linear", inplace=True)
    df["value"].fillna(method="bfill", inplace=True)
    df["value"].fillna(method="ffill", inplace=True)

    # Convert the date column to int
    df["date"] = df["date"].astype("int")
    # Add a forecast indicator column
    df["is_forecast"] = False
    return df


def get_arima_forecasts(df: pd.DataFrame, forecast_dates: list) -> np.ndarray:
    # Fit an ARIMA model to the full data
    arima_df = df.copy()
    arima_df.set_index("date", inplace=True)
    arima_model = ARIMA(arima_df[["value"]], order=(1, 1, 1))
    arima_model_fit = arima_model.fit()

    arima_forecasts = arima_model_fit.forecast(steps=len(forecast_dates))
    return arima_forecasts


def get_linear_regression_forecasts(
    df: pd.DataFrame, forecast_dates: list
) -> np.ndarray:
    # Fit a linear regression model to the full data
    x = df["date"].values.reshape(-1, 1)
    y = df["value"].values
    linear_model = LinearRegression().fit(x, y)

    linear_forecasts = linear_model.predict(np.array(forecast_dates).reshape(-1, 1))
    return linear_forecasts


def get_forecasts_ensemble(
    arima_forecasts: np.ndarray, linear_forecasts: np.ndarray
) -> np.ndarray:
    # Return the average of the two forecasts
    return (arima_forecasts + linear_forecasts) / 2


def main(indicator: str, country: str, ignore_missing_start_values: bool = True):
    """
    Downloads the data from the World Bank API, generates forecasts for the
    specified indicator and country, and saves the result as a json file.

    Args:
        indicator (str): The indicator to download data for.
        country (str): The country to download data for.
        ignore_missing_start_values (bool): Whether to ignore missing values at the
            beginning of the time series (by default it's set to True).

    Example:
        python python_test_spgi.py --indicator="NY.GDP.MKTP.CN" --country="afg" --ignore_missing_start_values=False

    """
    # Download the data from API
    data = download_data(indicator=indicator, country=country)

    # Prepare the data for forecasting
    df = prepare_data_for_forecasting(data, ignore_missing_start_values)

    # Generate forecasts for every year between max date from the input data and year 2030
    forecast_dates = [i for i in range(max(df["date"]) + 1, 2031)]

    # Generate the forecasts
    arima_forecasts_values = get_arima_forecasts(df, forecast_dates)
    ln_forecasts_values = get_linear_regression_forecasts(df, forecast_dates)
    ensemble_forecasts_values = get_forecasts_ensemble(
        arima_forecasts_values, ln_forecasts_values
    )

    # Create a dataframe with the forecast dates and values
    forecast_df = pd.DataFrame(
        {
            "date": forecast_dates,
            "value": ensemble_forecasts_values,
            "indicator": np.nan,
            "country": np.nan,
            "is_forecast": True,
        }
    )

    # Append the forecast dataframe to the dataframe with history data and save the result as a json
    history_with_forecast_df = pd.concat([df, forecast_df])
    history_with_forecast_df[["indicator", "country"]].fillna(method="ffill", inplace=True)
    history_with_forecast_df.to_json("output.json", orient="records", indent=4)


if __name__ == "__main__":
    fire.Fire(main)
