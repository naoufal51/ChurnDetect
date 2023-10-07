import os
import pandas as pd
import logging

from churn_library import (
    import_data,
    perform_eda,
    encoder_helper,
    perform_feature_engineering,
    train_models,
)


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_eda(perform_eda):
    """
    test perform eda function
    """
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    assert os.path.exists("images/churn_distribution.png")
    assert os.path.exists("images/customer_age_distribution.png")
    assert os.path.exists("images/marital_status_distribution.png")
    assert os.path.exists("images/total_transaction_distribution.png")
    assert os.path.exists("images/correlation_matrix.png")
    assert os.path.exists("images/churn_by_gender.png")
    logging.info("Testing perform_eda: Plot generation SUCCESS")


def test_encoder_helper(encoder_helper):
    """
    test encoder helper
    """
    df = import_data("./data/bank_data.csv")
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    orig_cols = df.columns
    df_encoded = encoder_helper(df, ["Gender"], "Churn")

    assert set(orig_cols).issubset(df_encoded.columns)
    assert "Gender_Churn" in df_encoded.columns
    assert not df_encoded.isna().any().any()
    logging.info("Testing encoder_helper: Encoding SUCCESS")


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """
    df = import_data("./data/bank_data.csv")
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()

    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    logging.info(
        "Testing perform_feature_engineering: Feature Engineering SUCCESS")


def test_train_models(train_models):
    """
    test train_models
    """
    df = import_data("./data/bank_data.csv")
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")
    train_models(X_train, X_test, y_train, y_test)

    assert os.path.exists("./models/rfc_model.pkl")
    assert os.path.exists("./models/logistic_model.pkl")

    assert os.path.exists("images/roc_curve.png")

    assert os.path.exists("images/random_forest_importance.png")

    assert os.path.exists("images/logistic_regression_shap_lr.png")
    assert os.path.exists("images/random_forest_shap_rf.png")

    assert os.path.exists("images/random_forest_report.png")
    assert os.path.exists("images/logistic_regression_report.png")

    logging.info("Testing train_models: Model training and saving SUCCESS")


if __name__ == "__main__":
    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)
