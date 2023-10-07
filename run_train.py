from churn_library import (
    import_data,
    perform_feature_engineering,
    train_models,
    perform_eda,
)
import logging
import os
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")
    train_models(X_train, X_test, y_train, y_test)
    logging.info("Training and testing models SUCCESS")
