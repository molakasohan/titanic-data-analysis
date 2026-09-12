"""
analysis.py
-----------
Exploratory data analysis on the real Titanic passenger dataset
(source: https://github.com/datasciencedojo/datasets).

Run with:
    python src/analysis.py

What it does:
    1. Loads data/titanic.csv
    2. Cleans it (missing values, dtypes, derived columns)
    3. Computes summary statistics and survival-related insights
    4. Saves 5 charts to the output/ folder
    5. Prints a short text summary to the console
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path="data/titanic.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def clean_data(df):
    print("\n--- Missing values before cleaning ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Age: fill missing with the median age per passenger class (more accurate
    # than a single global median, since class correlates with age)
    df["Age"] = df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))

    # Embarked: fill the 2 missing values with the most common port
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Cabin has too many missing values (~77%) to impute meaningfully —
    # convert it into a simple "has_cabin" flag instead of dropping the column
    df["has_cabin"] = df["Cabin"].notna().astype(int)
    df = df.drop(columns=["Cabin"])

    # Derived columns useful for analysis
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")

    return df


def summarize(df):
    print("\n--- Summary Statistics ---")
    print(df[["Age", "Fare", "family_size"]].describe())

    overall_survival = df["Survived"].mean() * 100
    survival_by_sex = df.groupby("Sex")["Survived"].mean() * 100
    survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100

    print(f"\nOverall survival rate: {overall_survival:.1f}%")
    print(f"Survival rate by sex:\n{survival_by_sex.round(1)}")
    print(f"\nSurvival rate by class:\n{survival_by_class.round(1)}")

    return {
        "overall_survival": overall_survival,
        "survival_by_sex": survival_by_sex,
        "survival_by_class": survival_by_class,
    }


def make_charts(df):
    # 1. Survival rate by passenger class
    plt.figure(figsize=(7, 5))
    class_surv = df.groupby("Pclass")["Survived"].mean() * 100
    sns.barplot(x=class_surv.index, y=class_surv.values, hue=class_surv.index,
                palette="viridis", legend=False)
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate (%)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/survival_by_class.png", dpi=150)
    plt.close()

    # 2. Survival rate by sex
    plt.figure(figsize=(6, 5))
    sex_surv = df.groupby("Sex")["Survived"].mean() * 100
    sns.barplot(x=sex_surv.index, y=sex_surv.values, hue=sex_surv.index,
                palette="magma", legend=False)
    plt.title("Survival Rate by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Survival Rate (%)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/survival_by_sex.png", dpi=150)
    plt.close()

    # 3. Age distribution: survivors vs non-survivors
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", bins=30, palette="Set2")
    plt.title("Age Distribution: Survived vs Did Not Survive")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/age_distribution_survival.png", dpi=150)
    plt.close()

    # 4. Fare distribution by class
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Pclass", y="Fare", hue="Pclass", palette="crest", legend=False)
    plt.title("Fare Distribution by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Fare ($)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fare_by_class.png", dpi=150)
    plt.close()

    # 5. Survival rate by family size
    plt.figure(figsize=(8, 5))
    family_surv = df.groupby("family_size")["Survived"].mean() * 100
    sns.barplot(x=family_surv.index, y=family_surv.values, hue=family_surv.index,
                palette="flare", legend=False)
    plt.title("Survival Rate by Family Size (incl. self)")
    plt.xlabel("Family Size")
    plt.ylabel("Survival Rate (%)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/survival_by_family_size.png", dpi=150)
    plt.close()

    print(f"\nSaved 5 charts to {OUTPUT_DIR}/")


def main():
    df = load_data()
    df = clean_data(df)
    summarize(df)
    make_charts(df)


if __name__ == "__main__":
    main()
