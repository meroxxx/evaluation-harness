import pandas as pd
from sklearn.model_selection import train_test_split
//I add changes to test te super-linter
class DataSplitter:
    def __init__(self, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):

        if round(train_size + val_size + test_size, 5) != 1.0:
            raise ValueError("train + val + test must equal 1")

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

    def remove_duplicates(self, df: pd.DataFrame):
        """Удаление дубликатов"""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)

        if removed > 0:
            print(f"Removed duplicates: {removed}")

        return df

    def split(self, df: pd.DataFrame, target_column=None):
        """Разделение на train / val / test"""

        stratify = df[target_column] if target_column else None

        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - self.train_size),   # 40%
            random_state=self.random_state,
            stratify=stratify
        )

        val_ratio = self.val_size / (self.val_size + self.test_size)  # 0.5

        stratify_temp = temp_df[target_column] if target_column else None

        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio),   # 50% от оставшихся
            random_state=self.random_state,
            stratify=stratify_temp
        )

        return train_df, val_df, test_df

    def check_leakage(self, train_df, val_df, test_df):
        """Контроль утечек данных"""

        train_ids = set(train_df.index)
        val_ids = set(val_df.index)
        test_ids = set(test_df.index)

        if train_ids & val_ids:
            raise ValueError("Leakage detected between train and val")

        if train_ids & test_ids:
            raise ValueError("Leakage detected between train and test")

        if val_ids & test_ids:
            raise ValueError("Leakage detected between val and test")

        print("Leakage check passed")

    def split_dataset(self, df: pd.DataFrame, target_column=None):
        """Полный pipeline разделения"""

        df = self.remove_duplicates(df)

        train_df, val_df, test_df = self.split(df, target_column)

        self.check_leakage(train_df, val_df, test_df)

        return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, path="data"):
    """Сохранение файлов"""

    train_df.to_csv(f"{path}/train.csv", index=False)
    val_df.to_csv(f"{path}/val.csv", index=False)
    test_df.to_csv(f"{path}/test.csv", index=False)

    print("Files saved:")
    print("train.csv")
    print("val.csv")
    print("test.csv")
