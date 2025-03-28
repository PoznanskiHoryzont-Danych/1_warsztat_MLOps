import argparse
import os
from typing import Literal

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


# 1. Załadowanie danych z pliku CSV
def load_data(train_csv_file: str, test_csv_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Funkcja do ładowania danych z pliku CSV.
    Zakłada, że plik CSV ma kolumny 'Sentyment' i 'Tekst'.
    """
    ...  # TUTAJ UMIEŚĆ KOD DO ZAŁADOWANIA DANYCH Z CSV PRZY UŻYCIU PANDAS
    print(f"Załadowano dane z pliku: {train_csv_file}. Liczba przykładów: {train_data.shape[0]}")
    print(f"Załadowano dane z pliku: {test_csv_file}. Liczba przykładów: {test_data.shape[0]}")
    return train_data, test_data


# 2. Podział danych na zbiór treningowy i walidacyjny
def split_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) \
        -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Funkcja do podziału danych na zbiór treningowy i testowy.
    """
    ...  # TUTAJ UMIEŚĆ KOD DO PODZIAŁU DANYCH NA ZBIÓR TRENINGOWY I TESTOWY (train_test_split)
    print(f"Podzielono dane na zbiór treningowy ({X_train.shape[0]} przykładów) i walidacyjny ({X_valid.shape[0]} przykladów).")
    return X_train, X_valid, y_train, y_valid


# 3. Baseline
def baseline_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    """
    Funkcja do tworzenia i trenowania Baselinu, czyli modeu zwracajacego naczęstszą klasę
    """
    # ...  # TUTAJ UMIEŚĆ KOD DO TWORZENIA PIPELINU I TRENOWANIA MODELU (pipeline.fit)
    print("Model został wytrenowany.")
    return pipeline


# 4. Ewaluacja modelu
def evaluate_model(pipeline: Pipeline, X_valid: pd.DataFrame, y_valid: pd.DataFrame, mode: Literal["train", "test"]) -> float:
    """
    Funkcja do ewaluacji wytrenowanego modelu.
    Wyświetla metryki: accuracy, raport klasyfikacji i macierz konfuzji.
    """

    predictions = None  # TUTAJ UMIEŚĆ KOD DO PREDYKCJI NA ZBIORZE TESTOWYM
    accuracy = None  # TUTAJ UMIEŚĆ KOD DO mierzenie metryki dokładności

    database_type = "testowy" if mode == "test" else "walidacyjnym"
    print(f"Accuracy na zbiorze {database_type}: {accuracy:.4f}")
    return accuracy


# 5. Funkcja zapisująca model do Joblib
def save_model(pipeline: Pipeline, pipeline_filename: str):  # Zmieniamy domyślną nazwę pliku
    """
    Funkcja zapisująca wytrenowany pipeline do pliku Joblib.
    """
    # TUTAJ UMIEŚĆ KOD DO ZAPISYWANIA MODELU PIPELINE DO PLIKU JOBLIB'A (joblib.dump)
    print(f"Model Pipeline zapisano do: {pipeline_filename}")


# 6. Wczytywanie modelu
def load_model(pipeline_filename: str) -> Pipeline:
    # TUTAJ UMIEŚĆ KOD DO WCZYTYWANIA MODELU PIPELINE Z PLIKU JOBLIB'A
    pipeline = None
    print(f"Model Pipeline wczytano z: {pipeline_filename}")
    return pipeline


# 7. Trenowanie modelu
def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    """
    Funkcja do tworzenia i trenowania Pipeline.
    Używa CountVectorizer do ekstrakcji cech z tekstu i DecisionTreeClassifier jako klasyfikatora.
    """
    ...  # TUTAJ UMIEŚĆ KOD DO TWORZENIA PIPELINU I TRENOWANIA MODELU (pipeline.fit)
    print("Model został wytrenowany.")
    return pipeline


# Funkcja do sprawdzania, czy pipeline jest nauczony
def is_pipeline_fitted(pipeline: Pipeline):
    try:
        check_is_fitted(pipeline)
        return True
    except NotFittedError:
        return False


# Główna funkcja
def main(args: argparse.Namespace):
    """
    Główna funkcja skryptu, która łączy wszystkie kroki.
    !!!!! NIE ZMIENIAJ KODU FUNKCJI !!!!!
    """
    train_csv_file = 'data/train.csv'  # Nazwa pliku CSV z danymi
    test_csv_file = 'data/test.csv'  # Nazwa pliku CSV z danymi

    # Load data
    train_data, test_data = load_data(train_csv_file, test_csv_file)
    assert isinstance(train_data, pd.DataFrame), f"Train data nie jest typu pd.DataFrame, jest {type(train_data)}"
    assert train_data.shape[0] == 432, f"Żle wczytane dane, dane treningowe powinny mieć 432 wiersze, a ma {train_data.shape[0]}"
    assert test_data.shape[0] == 296, f"Żle wczytane dane, dane testowe powinny mieć 296 wiersze, a ma {test_data.shape[0]}"

    print("\nZadanie NR 1 wykonane prawidłowo!\n")
    if args.mode == "train":
        # Training mode
        X_train, X_valid, y_train, y_valid = split_data(train_data)
        assert X_train is not None and X_valid is not None and  y_train is not None and  y_valid is not None, \
            "Funkcja split_data zwróciła None"
        assert (X_train.shape[0] == y_train.shape[0]) and (X_valid.shape[0] == y_valid.shape[0]) , \
            "Zły podział danych, dane i oznaczenia mają inne rozmiary"
        print("\nZadanie NR 2 wykonane prawidłowo!\n")
        if args.model_type == "baseline":
            print("Trenowanie modelu baseline (DummyClassifier)")
            pipeline = baseline_model(X_train, y_train)
            assert isinstance(pipeline, Pipeline), f"Funkcja baseline_model zwróciła {type(pipeline)} a powinna Pipeline"
            assert is_pipeline_fitted(pipeline), "Model z baseline_model został niewytrenowany"
            print("\nZadanie NR 3 wykonane prawidłowo!\n")
        elif args.model_type == "decision-tree":
            print("Trenowanie modelu DecisionTreeClassifier")
            pipeline = train_model(X_train, y_train)
            assert isinstance(pipeline, Pipeline), f"funkcja train_model zwróciła {type(pipeline)} a powinna Pipeline"
            assert is_pipeline_fitted(pipeline), "Model z train_model został niewytrenowany"
            print("\nZadanie NR 7 wykonane prawidłowo!\n")
        else:
            raise ValueError(f"{args.model_type} nie jest obsługiwany")

        accuracy = evaluate_model(pipeline, X_valid, y_valid, mode=args.mode)
        assert isinstance(accuracy, float) and 0 < accuracy < 1, "Evaluate_model zwróciło nieprawidłową wartość"
        print("\nZadanie NR 4 wykonane prawidłowo!\n")
        save_model(pipeline, args.save_path)
        assert os.path.isfile(args.save_path), "Plik z modelem nie został zapisany"
        print("\nZadanie NR 5 wykonane prawidłowo!\n")

    elif args.mode == "test":
        # Testing mode
        pipeline = load_model(args.model_path)
        assert isinstance(pipeline, Pipeline), f"funkcja load_model zwróciła {type(pipeline)} a powinna Pipeline"
        print("\nZadanie NR 6 wykonane prawidłowo!\n")
        X_test = test_data['Tekst']
        y_test = test_data['Sentyment']
        evaluate_model(pipeline, X_test, y_test, mode=args.mode)


if __name__ == "__main__":
    """
    !!!!! NIE ZMIENIAJ KODU FUNKCJI !!!!!
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Model analizy sentymentu: Trening i tesowanie")

    # Define subcommands for training and testing
    subparsers = parser.add_subparsers(dest="mode", help="Mode: train lub test")

    # Training mode arguments
    parser_train = subparsers.add_parser("train", help="Naucz model przewidywania sentymentu")
    parser_train.add_argument(
        "--model-type",
        type=str,
        choices=["baseline", "decision-tree"],
        required=True,
        help="Naucz model 'baseline' (DummyClassifier) albo 'decision-tree' (DecisionTreeClassifier)"
    )
    parser_train.add_argument(
        "--save-path",
        type=str,
        default="sentiment_model.joblib",
        help="ścieżka zapisu nauczonego modelu"
    )

    # Testing mode arguments
    parser_test = subparsers.add_parser("test", help="Test wytreowanego modelu na zbiorze testowym")
    parser_test.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file"
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if a mode was specified
    if not args.mode:
        parser.print_help()
        exit(1)

    main(args)