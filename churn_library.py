# library doc string


# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
import shap
import joblib
import logging

logging.basicConfig(
    filename="./logs/churn_library_main.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    try:
        df = pd.read_csv(pth)
        logging.info(f"Success in importing data from {pth}")
    except Exception as e:
        logging.error(f"Error in importing data from {pth}")
        raise
    return df


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """

    def save_plot(plot_fn, filename, **kwargs):
        if not os.path.exists("images"):
            os.makedirs("images")
        plt.figure(figsize=(20, 10))
        plot_fn(**kwargs)
        plt.savefig(f"images/{filename}.png")
        plt.close()
        logging.info(f"Saved {filename} plot to images folder")

    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    save_plot(
        df["Churn"].hist,
        "churn_distribution",
    )
    save_plot(
        df["Customer_Age"].hist,
        "customer_age_distribution",
    )
    save_plot(
        df.Marital_Status.value_counts(normalize=True).plot,
        "marital_status_distribution",
        kind="bar",
    )
    save_plot(
        sns.histplot,
        "total_transaction_distribution",
        data=df["Total_Trans_Ct"],
        stat="density",
        kde=True,
    )
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    save_plot(
        sns.heatmap,
        "correlation_matrix",
        data=correlation_matrix,
        annot=False,
        cmap="Dark2_r",
        linewidths=2,
    )


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for category in category_lst:
        category_groups = df.groupby(category)[response].mean()
        df[f"{category}_{response}"] = df[category].map(category_groups)

    logging.info("Encoding SUCCESS")

    return df


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    # Columns to encode and keep
    category_lst = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
    ] + [f"{cat}_{response}" for cat in category_lst]

    df = encoder_helper(df, category_lst, response)
    X = pd.DataFrame()

    X[keep_cols] = df[keep_cols]
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    logging.info("Feature Engineering SUCCESS")
    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    if not os.path.exists("images"):
        os.makedirs("images")

    models = {
        "Logistic Regression": {"train": y_train_preds_lr, "test": y_test_preds_lr},
        "Random Forest": {"train": y_train_preds_rf, "test": y_test_preds_rf},
    }

    for model_name, data in models.items():
        plt.rc("figure", figsize=(8, 6))

        plt.text(
            0.01,
            1.25,
            f"{model_name} Train",
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.05,
            str(classification_report(y_train, data["train"])),
            {"fontsize": 10},
            fontproperties="monospace",
        )

        plt.text(
            0.01,
            0.6,
            f"{model_name} Test",
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.7,
            str(classification_report(y_test, data["test"])),
            {"fontsize": 10},
            fontproperties="monospace",
        )

        plt.axis("off")
        plt.savefig(f"images/{model_name.replace(' ', '_').lower()}_report.png")
        plt.close()
        logging.info(f"Saved {model_name} classification report to images folder")


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model

    # SHAP summary plot for both models
    if isinstance(estimator, RandomForestClassifier):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_data)
        shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
        plt.savefig(output_pth.replace("importance.png", "shap_rf.png"))
        plt.close()

        # Feature Importance plot only for Random Forest
        importances = estimator.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [X_data.columns[i] for i in indices]
        plt.figure(figsize=(20, 5))
        plt.title("Feature Importance")
        plt.ylabel("Importance")
        plt.bar(range(X_data.shape[1]), importances[indices])
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        plt.savefig(output_pth)
        plt.close()
        logging.info(f"Saved feature importance plot to {output_pth}")

    elif isinstance(estimator, LogisticRegression):
        explainer = shap.LinearExplainer(estimator, X_data)
        shap_values = explainer.shap_values(X_data)
        shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
        plt.savefig(output_pth.replace("importance.png", "shap_lr.png"))
        plt.close()
        logging.info(f"Saved feature importance plot to {output_pth}")

    else:
        logging.error("Model not supported")
        raise ValueError("Model not supported")


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    logging.info("GridSearchCV for Random Forest SUCCESS")

    lrc.fit(X_train, y_train)
    logging.info("Training for LogisticRegression SUCCESS")

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig("images/roc_curve.png")
    plt.close()
    logging.info("ROC Curve plot generation SUCCESS")

    # save models
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")
    logging.info("Models saved SUCCESS")

    # plot feature importance
    models = {"Random Forest": cv_rfc, "Logistic Regression": lrc}

    for model_name, model in models.items():
        output_path = f"images/{model_name.replace(' ', '_').lower()}_importance.png"
        feature_importance_plot(model, X_test, output_path)
