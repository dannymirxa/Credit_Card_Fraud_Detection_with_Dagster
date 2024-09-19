import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from dagster import asset, AssetExecutionContext, AssetIn, MetadataValue, Output, AssetMaterialization

@asset
def create_dataframe(context) -> Output[pd.DataFrame]:
    data = pd.read_csv('./dataset/creditcard_2023.csv')
    # return data
    metadata = {"sample": MetadataValue.md(data.head().to_markdown())}
    return Output(
        value=data, 
        metadata=metadata
    )

@asset(ins={"data": AssetIn(key="create_dataframe")})
def standardize_dataframe(data: pd.DataFrame) -> Output[pd.DataFrame]:
    sc = StandardScaler()
    data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))
    data = data.drop(['id'], axis = 1)
    data = data.drop_duplicates()
    # return data
    return Output(
        value=data, 
        metadata={"sample": MetadataValue.md(data.head().to_markdown())}
    )

# df = standardize_dataframe(create_dataframe('./dataset/creditcard_2023.csv'))

@asset(ins={"data": AssetIn(key="standardize_dataframe")})
def split_dataset(data: pd.DataFrame) -> Output:
    X = data.drop('Class', axis = 1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    # return X_train, X_test, y_train, y_test
    return Output(
        value=(X_train, X_test, y_train, y_test),
        metadata={"X_train": MetadataValue.md(X_train.head().to_markdown()),
                  "X_test": MetadataValue.md(X_test.head().to_markdown()),
                  "y_train": MetadataValue.md(y_train.head().to_markdown()),
                  "y_test": MetadataValue.md(y_test.head().to_markdown())}
    )

# X_train, X_test, y_train, y_test = split_dataset(df)

@asset
def classifiers() -> dict:
    return {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Support Vector Classifier": SVC(),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier()
    }


@asset(ins={"data": AssetIn(key="split_dataset"), "classifiers": AssetIn(key="classifiers")})
def create_prediction_report(context: AssetExecutionContext, data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], classifiers: dict) -> Output[pd.DataFrame]:
    X_train, X_test, y_train, y_test = data
    reports = []

    for model_name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df_result = pd.DataFrame(report).transpose()
        df_result['model'] = model_name
        df_result['metrics'] = df_result.index
        df_result.reset_index(drop=True, inplace=True)
        reports.append(df_result)

    combined_report = pd.concat(reports)
    metadata = {"sample": MetadataValue.md(combined_report.head().to_markdown())}
    return Output(
        value=combined_report,
        metadata=metadata
    )

@asset(ins={"prediction_report": AssetIn(key="create_prediction_report")})
def create_whole_prediction_report(prediction_report: pd.DataFrame) -> Output[pd.DataFrame]:
    # In this case, the prediction report is already combined in the previous step
    metadata = {"sample": MetadataValue.md(prediction_report.head().to_markdown())}
    return Output(
        value=prediction_report,
        metadata=metadata
    )

# @asset(ins={"X_train": AssetIn(key="split_dataset"), "y_train": AssetIn(key="split_dataset"), "X_test": AssetIn(key="split_dataset"), "y_test": AssetIn(key="split_dataset"), "model_name": AssetIn(key="classifiers"), "model": AssetIn(key="classifiers")})
# def create_prediction_report(X_train, X_test, y_train, y_test, model_name: str, model: Any) -> Output[pd.DataFrame]:
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     report = classification_report(y_test, y_pred, output_dict=True)
#     df_result = pd.DataFrame(report).transpose()
#     df_result['model'] = model_name
#     df_result['metrics'] = df_result.index
#     df_result.reset_index(drop=True, inplace=True)
#     return Output(
#         value=df_result, 
#         metadata={"sample": MetadataValue.md(df_result.head().to_markdown())}
#     )

# @asset(ins={"X_train": AssetIn(key="split_dataset"), "y_train": AssetIn(key="split_dataset"), "X_test": AssetIn(key="split_dataset"), "y_test": AssetIn(key="split_dataset")})
# def create_whole_prediction_report(X_train, X_test, y_train, y_test) -> Output[pd.DataFrame]:
#     df_result = pd.concat([create_prediction_report(X_train, X_test, y_train, y_test, name, clf) for name, clf in classifier.items()])
#     return Output(
#         value=df_result, 
#         metadata={"sample": MetadataValue.md(df_result.head().to_markdown())}
#     )