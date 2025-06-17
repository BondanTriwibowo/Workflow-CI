import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse

def main(data_path):
    mlflow.start_run()
    
    df = pd.read_csv(data_path)
    X = df.drop("traffic_volume", axis=1)
    y = df["traffic_volume"]

    # 🔧 Encode kolom kategorikal
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    X = pd.get_dummies(X, columns=categorical_cols)

    # ✅ Baru split data setelah encoding
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    
    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
