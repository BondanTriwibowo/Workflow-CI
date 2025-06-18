import pandas as pd
import mlflow
import os
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
import joblib  # ✅ Tambahan penting

def main(data_path):
    with mlflow.start_run():
        # Baca data
        df = pd.read_csv(data_path)
        X = df.drop("traffic_volume", axis=1)
        y = df["traffic_volume"]

        # Encode kolom kategorikal
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        X = pd.get_dummies(X, columns=categorical_cols)
<<<<<<< HEAD

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Prediksi dan log metrik
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)

        # Logging model dengan signature dan input_example yang valid
        input_example = X_test.iloc[:2].astype(float)  # konversi ke float untuk keamanan
        signature = infer_signature(X_test, y_pred)

        print("🚀 Logging model ke MLflow:")
        print(input_example)

        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        signature=signature
)

=======

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Prediksi dan log metrik
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)

        # Logging model dengan signature dan input_example yang valid
        input_example = X_test.iloc[:2].astype(float)  # konversi ke float untuk keamanan
        signature = infer_signature(X_test, y_pred)

        print("🚀 Logging model ke MLflow:")
        print(input_example)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        # ✅ Simpan juga ke artifacts/ agar bisa diupload sebagai artefak
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, "artifacts/model.pkl")
>>>>>>> 3b949d1 (Revisi CI workflow: install joblib & upload artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
