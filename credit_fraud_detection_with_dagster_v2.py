import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Any, Tuple, Dict
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from dagster import asset, AssetExecutionContext, AssetIn, MetadataValue, Output

@asset
def create_dataframe(context: AssetExecutionContext) -> Output[pd.DataFrame]:
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv('./dataset/creditcard_2023.csv')
        metadata = {
            "sample": MetadataValue.md(data.head().to_markdown()),
            "shape": MetadataValue.int(len(data))
        }
        return Output(value=data, metadata=metadata)
    except Exception as e:
        context.log.error(f"Error loading data: {e}")
        raise

@asset(ins={"data": AssetIn(key="create_dataframe")})
def standardize_dataframe(context: AssetExecutionContext, data: pd.DataFrame) -> Output[pd.DataFrame]:
    """Standardize the 'Amount' column and remove duplicates."""
    try:
        sc = StandardScaler()
        data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))
        data = data.drop(['id'], axis=1)
        data = data.drop_duplicates()
        metadata = {
            "sample": MetadataValue.md(data.head().to_markdown()),
            "shape": MetadataValue.int(len(data))
        }
        return Output(value=data, metadata=metadata)
    except Exception as e:
        context.log.error(f"Error standardizing data: {e}")
        raise

@asset(ins={"data": AssetIn(key="standardize_dataframe")})
def split_dataset(context: AssetExecutionContext, data: pd.DataFrame) -> Output[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Split the dataset into training and testing sets."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        metadata = {
            "X_train": MetadataValue.md(X_train.head().to_markdown()),
            "X_test": MetadataValue.md(X_test.head().to_markdown()),
            "y_train": MetadataValue.md(y_train.head().to_markdown()),
            "y_test": MetadataValue.md(y_test.head().to_markdown())
        }
        return Output(value=(X_train, X_test, y_train, y_test), metadata=metadata)
    except Exception as e:
        context.log.error(f"Error splitting dataset: {e}")
        raise

@asset
def classifiers() -> Dict[str, Any]:
    """Return a dictionary of classifiers."""
    return {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Support Vector Classifier": SVC(),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier()
    }

@asset(ins={"data": AssetIn(key="split_dataset"), "classifiers": AssetIn(key="classifiers")})
def create_prediction_report(context: AssetExecutionContext, data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], classifiers: Dict[str, Any]) -> Output[pd.DataFrame]:
    """Train classifiers and create a prediction report."""
    try:
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
        metadata = {
            "sample": MetadataValue.md(combined_report.head().to_markdown()),
            "shape": MetadataValue.int(len(combined_report))
        }
        return Output(value=combined_report, metadata=metadata)
    except Exception as e:
        context.log.error(f"Error creating prediction report: {e}")
        raise

@asset(ins={"prediction_report": AssetIn(key="create_prediction_report")})
def create_whole_prediction_report(context: AssetExecutionContext, prediction_report: pd.DataFrame) -> Output[pd.DataFrame]:
    """Return the combined prediction report."""
    try:
        metadata = {
            "sample": MetadataValue.md(prediction_report.head().to_markdown()),
            "shape": MetadataValue.int(len(prediction_report))
        }
        return Output(value=prediction_report, metadata=metadata)
    except Exception as e:
        context.log.error(f"Error creating whole prediction report: {e}")
        raise
