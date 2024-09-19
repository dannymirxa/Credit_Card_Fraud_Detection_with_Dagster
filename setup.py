from setuptools import find_packages, setup

setup(
    name="credit_card_fraud_detection_with_dagster",
    packages=find_packages(exclude=["credit_card_fraud_detection_with_dagster_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        "kaggle",
        "pandas",
        "scikit-learn"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
