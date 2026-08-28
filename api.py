import os
import joblib
import pandas as pd
import numpy as np
import requests

from flask import Flask, request, jsonify


app = Flask(__name__)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "rainfall_ensemble.pkl"
)


HF_REPO = os.environ.get(
    "HF_REPO",
    "rottenPotato6969/rainfall-ensemble"
)

HF_FILENAME = "rainfall_ensemble.pkl"

HF_TOKEN = os.environ.get("HF_TOKEN")


FEATURES = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustSpeed",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
    "Month_Sin",
    "Month_Cos",
    "TempDiff",
    "PressureDiff",
    "HumidityDiff"
]


def download_model():

    if os.path.exists(MODEL_PATH):

        print(
            "Model already exists locally."
        )

        return


    print("=" * 70)
    print("DOWNLOADING MODEL")
    print("=" * 70)

    url = (
        f"https://huggingface.co/"
        f"{HF_REPO}/resolve/main/"
        f"{HF_FILENAME}"
    )


    headers = {}

    if HF_TOKEN:

        headers[
            "Authorization"
        ] = f"Bearer {HF_TOKEN}"


    print(
        f"Repository: {HF_REPO}"
    )

    print(
        f"File: {HF_FILENAME}"
    )


    response = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=1800
    )

    response.raise_for_status()


    total_size = int(
        response.headers.get(
            "content-length",
            0
        )
    )


    downloaded = 0


    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if not chunk:
                continue

            file.write(chunk)

            downloaded += len(chunk)


            if total_size:

                percent = (
                    downloaded /
                    total_size
                ) * 100

                print(
                    f"\rDownloading model: "
                    f"{percent:.1f}%",
                    end="",
                    flush=True
                )


    print()
    print(
        "Model downloaded successfully."
    )

    print("=" * 70)


def load_model():

    download_model()

    print(
        "Loading rainfall ensemble..."
    )


    package = joblib.load(
        MODEL_PATH
    )


    rf_model = package[
        "rf_model"
    ]

    xgb_model = package[
        "xgb_model"
    ]

    lgbm_model = package[
        "lgbm_model"
    ]

    lr_model = package[
        "lr_model"
    ]

    preprocessor = package[
        "preprocessor"
    ]


    features = package.get(
        "features",
        FEATURES
    )

    target = package.get(
        "target",
        "RainTomorrow"
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
        f"Features            : "
        f"{len(features)}"
    )

    print(
        f"Target              : "
        f"{target}"
    )


    return (
        rf_model,
        xgb_model,
        lgbm_model,
        lr_model,
        preprocessor,
        features,
        target
    )


(
    rf_model,
    xgb_model,
    lgbm_model,
    lr_model,
    preprocessor,
    FEATURES,
    TARGET
) = load_model()


def get_rain_probability(
    model,
    X
):

    probabilities = (
        model.predict_proba(X)
    )

    classes = model.classes_


    if 1 in classes:

        index = list(
            classes
        ).index(1)


        return float(
            probabilities[
                :,
                index
            ][0]
        )


    return 0.0


def get_category(
    probability
):

    percentage = (
        probability * 100
    )


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


@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "message":
        "Deoghar Rainfall Prediction API",

        "status":
        "running"

    })


@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
        "healthy",

        "models": [

            "Random Forest",

            "XGBoost",

            "LightGBM",

            "Logistic Regression"

        ]

    })


@app.route(
    "/model-info",
    methods=["GET"]
)
def model_info():

    return jsonify({

        "features":
        list(FEATURES),

        "target":
        TARGET,

        "models": {

            "random_forest":
            list(
                rf_model.classes_
            ),

            "xgboost":
            list(
                xgb_model.classes_
            ),

            "lightgbm":
            list(
                lgbm_model.classes_
            ),

            "logistic_regression":
            list(
                lr_model.classes_
            )

        }

    })


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = (
            request.get_json()
        )


        if data is None:

            return jsonify({

                "error":
                "Request must contain JSON"

            }), 400


        missing_features = [

            feature

            for feature in FEATURES

            if feature not in data

        ]


        if missing_features:

            return jsonify({

                "error":
                "Missing features",

                "missing":
                missing_features

            }), 400


        input_data = {

            feature:
            data[feature]

            for feature in FEATURES

        }


        input_df = pd.DataFrame(
            [input_data]
        )


        X_processed = (
            preprocessor.transform(
                input_df
            )
        )


        rf_probability = (
            get_rain_probability(
                rf_model,
                X_processed
            )
        )


        xgb_probability = (
            get_rain_probability(
                xgb_model,
                X_processed
            )
        )


        lgbm_probability = (
            get_rain_probability(
                lgbm_model,
                X_processed
            )
        )


        lr_probability = (
            get_rain_probability(
                lr_model,
                X_processed
            )
        )


        ensemble_probability = np.mean([

            rf_probability,

            xgb_probability,

            lgbm_probability,

            lr_probability

        ])


        rain_probability = (
            ensemble_probability * 100
        )


        category = (
            get_category(
                ensemble_probability
            )
        )


        return jsonify({

            "rain_probability":
            round(
                rain_probability,
                2
            ),

            "prediction":
            category,

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

            }

        })


    except Exception as e:

        print(
            f"Prediction error: {e}"
        )


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


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )

