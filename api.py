import os
import math
import joblib
import requests
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "rainfall_ensemble.pkl"
)

GEOCODING_URL = (
    "https://geocoding-api.open-meteo.com/v1/search"
)

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


package = joblib.load(MODEL_PATH)


rf_model = package["rf_model"]
xgb_model = package["xgb_model"]
lgbm_model = package["lgbm_model"]
lr_model = package["lr_model"]

FEATURES = list(package["features"])
TARGET = package["target"]


app = Flask(__name__)


def safe_value(value, default=0.0):

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return default


def geocode_city(city):

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        GEOCODING_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if (
        "results" not in data
        or len(data["results"]) == 0
    ):
        raise ValueError(
            f"City not found: {city}"
        )

    result = data["results"][0]

    return {
        "name": result.get("name"),
        "latitude": result.get("latitude"),
        "longitude": result.get("longitude"),
        "country": result.get("country"),
        "state": result.get("admin1")
    }


def get_weather(latitude, longitude):

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "timezone": "auto",

        "forecast_days": 2,

        "daily": ",".join([

            "temperature_2m_min",

            "temperature_2m_max",

            "precipitation_sum",

            "rain_sum",

            "precipitation_probability_max"

        ]),

        "hourly": ",".join([

            "temperature_2m",

            "relative_humidity_2m",

            "surface_pressure",

            "wind_speed_10m",

            "wind_direction_10m",

            "wind_gusts_10m",

            "cloud_cover",

            "precipitation",

            "et0_fao_evapotranspiration",

            "sunshine_duration"

        ])
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def build_model_input(weather_data):

    hourly = weather_data["hourly"]

    hourly_df = pd.DataFrame(hourly)

    hourly_df["time"] = pd.to_datetime(
        hourly_df["time"]
    )

    today_date = (
        hourly_df["time"]
        .dt.date
        .min()
    )

    today_df = hourly_df[
        hourly_df["time"].dt.date == today_date
    ].copy()

    if len(today_df) == 0:
        raise ValueError(
            "No weather data available."
        )

    today_df["hour"] = (
        today_df["time"].dt.hour
    )

    min_temp = safe_value(
        today_df["temperature_2m"].min()
    )

    max_temp = safe_value(
        today_df["temperature_2m"].max()
    )

    rainfall = safe_value(
        today_df["precipitation"].sum()
    )

    evaporation = safe_value(
        today_df[
            "et0_fao_evapotranspiration"
        ].sum()
    )

    sunshine = safe_value(
        today_df[
            "sunshine_duration"
        ].sum() / 3600
    )

    wind_gust_speed = safe_value(
        today_df[
            "wind_gusts_10m"
        ].max()
    )

    morning_candidates = today_df[
        today_df["hour"] >= 9
    ]

    if len(morning_candidates) > 0:
        morning = morning_candidates.iloc[0]
    else:
        morning = today_df.iloc[0]

    afternoon_candidates = today_df[
        today_df["hour"] >= 15
    ]

    if len(afternoon_candidates) > 0:
        afternoon = afternoon_candidates.iloc[0]
    else:
        afternoon = today_df.iloc[-1]

    wind_speed_9am = safe_value(
        morning["wind_speed_10m"]
    )

    humidity_9am = safe_value(
        morning["relative_humidity_2m"]
    )

    pressure_9am = safe_value(
        morning["surface_pressure"]
    )

    cloud_9am = (
        safe_value(
            morning["cloud_cover"]
        ) / 100
    )

    temp_9am = safe_value(
        morning["temperature_2m"]
    )

    wind_speed_3pm = safe_value(
        afternoon["wind_speed_10m"]
    )

    humidity_3pm = safe_value(
        afternoon["relative_humidity_2m"]
    )

    pressure_3pm = safe_value(
        afternoon["surface_pressure"]
    )

    cloud_3pm = (
        safe_value(
            afternoon["cloud_cover"]
        ) / 100
    )

    temp_3pm = safe_value(
        afternoon["temperature_2m"]
    )

    month = today_date.month

    month_sin = math.sin(
        2 * math.pi * month / 12
    )

    month_cos = math.cos(
        2 * math.pi * month / 12
    )

    temp_diff = (
        max_temp - min_temp
    )

    pressure_diff = (
        pressure_9am - pressure_3pm
    )

    humidity_diff = (
        humidity_9am - humidity_3pm
    )

    model_input = {

        "MinTemp": min_temp,

        "MaxTemp": max_temp,

        "Rainfall": rainfall,

        "Evaporation": evaporation,

        "Sunshine": sunshine,

        "WindGustSpeed": wind_gust_speed,

        "WindSpeed9am": wind_speed_9am,

        "WindSpeed3pm": wind_speed_3pm,

        "Humidity9am": humidity_9am,

        "Humidity3pm": humidity_3pm,

        "Pressure9am": pressure_9am,

        "Pressure3pm": pressure_3pm,

        "Cloud9am": cloud_9am,

        "Cloud3pm": cloud_3pm,

        "Temp9am": temp_9am,

        "Temp3pm": temp_3pm,

        "Month_Sin": month_sin,

        "Month_Cos": month_cos,

        "TempDiff": temp_diff,

        "PressureDiff": pressure_diff,

        "HumidityDiff": humidity_diff
    }

    input_df = pd.DataFrame(
        [model_input]
    )

    input_df = input_df[FEATURES]

    return input_df


def get_rain_probability(
    model,
    input_df
):

    probabilities = model.predict_proba(
        input_df
    )

    classes = list(
        model.classes_
    )

    if 1 not in classes:
        return 0.0

    rain_index = classes.index(1)

    return float(
        probabilities[
            0,
            rain_index
        ]
    )


def get_category(probability):

    percentage = probability * 100

    if percentage < 20:
        return "NO RAIN"

    elif percentage < 40:
        return "LIGHT"

    elif percentage < 60:
        return "MODERATE"

    elif percentage < 80:
        return "HEAVY"

    else:
        return "VERY HEAVY"


def estimate_rainfall_mm(probability):

    percentage = probability * 100

    if percentage < 20:

        return 0.0

    elif percentage < 40:

        rainfall = (
            1 +
            ((percentage - 20) / 20) * 4
        )

    elif percentage < 60:

        rainfall = (
            5 +
            ((percentage - 40) / 20) * 15
        )

    elif percentage < 80:

        rainfall = (
            20 +
            ((percentage - 60) / 20) * 30
        )

    else:

        rainfall = (
            50 +
            ((percentage - 80) / 20) * 50
        )

    return round(rainfall, 2)


def get_tomorrow_forecast(weather_data):

    daily = weather_data.get(
        "daily",
        {}
    )

    dates = daily.get(
        "time",
        []
    )

    if len(dates) < 2:
        return None

    index = 1

    precipitation = safe_value(
        daily.get(
            "precipitation_sum",
            [0, 0]
        )[index]
    )

    rain = safe_value(
        daily.get(
            "rain_sum",
            [0, 0]
        )[index]
    )

    probability = safe_value(
        daily.get(
            "precipitation_probability_max",
            [0, 0]
        )[index]
    )

    min_temp = safe_value(
        daily.get(
            "temperature_2m_min",
            [0, 0]
        )[index]
    )

    max_temp = safe_value(
        daily.get(
            "temperature_2m_max",
            [0, 0]
        )[index]
    )

    return {

        "date": dates[index],

        "precipitation_mm": round(
            precipitation,
            2
        ),

        "rain_mm": round(
            rain,
            2
        ),

        "precipitation_probability_percent":
            round(
                probability,
                2
            ),

        "min_temperature_c":
            round(
                min_temp,
                2
            ),

        "max_temperature_c":
            round(
                max_temp,
                2
            )
    }


@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "message":
            "City Rainfall Prediction API",

        "status":
            "running",

        "target":
            TARGET,

        "endpoint":
            "POST /predict",

        "example":
            {
                "city":
                    "Deoghar"
            }
    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status":
            "healthy",

        "models":
            [

                "Random Forest",

                "XGBoost",

                "LightGBM",

                "Logistic Regression"

            ]
    })


@app.route("/model-info", methods=["GET"])
def model_info():

    return jsonify({

        "target":
            TARGET,

        "features":
            FEATURES,

        "feature_count":
            len(FEATURES),

        "models":
            {

                "random_forest":
                    type(
                        rf_model
                    ).__name__,

                "xgboost":
                    type(
                        xgb_model
                    ).__name__,

                "lightgbm":
                    type(
                        lgbm_model
                    ).__name__,

                "logistic_regression":
                    type(
                        lr_model
                    ).__name__

            }
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if data is None:

            return jsonify({

                "error":
                    "Request body must contain JSON."

            }), 400

        if "city" not in data:

            return jsonify({

                "error":
                    "city is required.",

                "example":
                    {
                        "city":
                            "Deoghar"
                    }

            }), 400

        city = str(
            data["city"]
        ).strip()

        if not city:

            return jsonify({

                "error":
                    "city cannot be empty."

            }), 400

        location = geocode_city(
            city
        )

        weather_data = get_weather(

            location["latitude"],

            location["longitude"]

        )

        input_df = build_model_input(
            weather_data
        )

        rf_probability = (
            get_rain_probability(
                rf_model,
                input_df
            )
        )

        xgb_probability = (
            get_rain_probability(
                xgb_model,
                input_df
            )
        )

        lgbm_probability = (
            get_rain_probability(
                lgbm_model,
                input_df
            )
        )

        lr_probability = (
            get_rain_probability(
                lr_model,
                input_df
            )
        )

        ensemble_probability = np.mean([

            rf_probability,

            xgb_probability,

            lgbm_probability,

            lr_probability

        ])

        rain_percentage = (
            ensemble_probability * 100
        )

        category = get_category(
            ensemble_probability
        )

        estimated_rainfall = (
            estimate_rainfall_mm(
                ensemble_probability
            )
        )

        tomorrow = (
            get_tomorrow_forecast(
                weather_data
            )
        )

        return jsonify({

            "location": {

                "city":
                    location["name"],

                "state":
                    location["state"],

                "country":
                    location["country"],

                "latitude":
                    location["latitude"],

                "longitude":
                    location["longitude"]

            },

            "prediction": {

                "date":
                    tomorrow["date"]
                    if tomorrow
                    else None,

                "rain_probability_percent":
                    round(
                        rain_percentage,
                        2
                    ),

                "category":
                    category,

                "estimated_rainfall_mm":
                    estimated_rainfall

            },

            "weather_api_forecast": {

                "precipitation_mm":
                    tomorrow[
                        "precipitation_mm"
                    ]
                    if tomorrow
                    else None,

                "rain_mm":
                    tomorrow[
                        "rain_mm"
                    ]
                    if tomorrow
                    else None,

                "precipitation_probability_percent":
                    tomorrow[
                        "precipitation_probability_percent"
                    ]
                    if tomorrow
                    else None,

                "minimum_temperature_c":
                    tomorrow[
                        "min_temperature_c"
                    ]
                    if tomorrow
                    else None,

                "maximum_temperature_c":
                    tomorrow[
                        "max_temperature_c"
                    ]
                    if tomorrow
                    else None
            },

            "model_predictions": {

                "random_forest":
                    round(
                        rf_probability * 100,
                        2
                    ),

                "xgboost":
                    round(
                        xgb_probability * 100,
                        2
                    ),

                "lightgbm":
                    round(
                        lgbm_probability * 100,
                        2
                    ),

                "logistic_regression":
                    round(
                        lr_probability * 100,
                        2
                    )
            },

            "model_input":
                input_df.to_dict(
                    orient="records"
                )[0]
        })

    except requests.exceptions.RequestException as e:

        return jsonify({

            "error":
                "External weather API request failed.",

            "details":
                str(e)

        }), 502

    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    print(
        "=" * 70
    )

    print(
        "CITY RAINFALL PREDICTION API"
    )

    print(
        "=" * 70
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        "Random Forest       : loaded"
    )

    print(
        "XGBoost             : loaded"
    )

    print(
        "LightGBM            : loaded"
    )

    print(
        "Logistic Regression : loaded"
    )

    print(
        f"Features            : {len(FEATURES)}"
    )

    print(
        f"Port                : {port}"
    )

    print(
        "=" * 70
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
