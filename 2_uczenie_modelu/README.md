# Część 2: Uczenie modelu do przewidywania sentymentu  
Druga część warsztatu służy nauczeniu modelu, który później zostanie zdeployowany. Twoim zadaniem jest wypełnienie funkcji, które służą do wczytania danych oraz nauki modeli.  

# Zadania do wykonania w 2. części warsztatu  
Zadania w tym segmencie zostały podzielone na 3 części („Nauka Baselinu”, „Przetestowanie Baselinu na zbiorze treningowym” oraz „Nauka Drzewa Decyzyjnego i Przetestowanie Baselinu na zbiorze treningowym”). W każdym module główne zadania będą polegały na wypełnieniu brakującego kodu funkcji.  
W celu uproszczenia pracy została stworzona funkcja `main`, która uruchamia każdą z funkcji, twoim zadaniem jest wypełnienie innych funkcji.

## Nauka Baselinu  
Wywołanie kodu poprzez:  
`python 2_uczenie_modelu/train_model.py train --model-type baseline --save-path model_baseline.joblib`  

Do wypełnienia funkcje nr:  
1. Wczytanie danych za pomocą biblioteki pandas ([pd.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html))  
   - Sprawdź format danych i separatory kolumn `sep`.  
2. Podział danych treningowych na dane treningowe i walidacyjne za pomocą biblioteki sklearn ([train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split))  
   - Pamiętaj o podziale z uwzględnieniem oznaczeń `stratify`.  
   - Zapewnij powtarzalność za każdym uruchomieniem za pomocą `random_state`.  
3. Naucz Baseline:  
   - Użyj modelu BoW do zliczania słów w zdaniach ([CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html))  
     - Zwróć uwagę na parametry klasy, takie jak maksymalna liczba cech/słów `max_features`, co robić z rzadkimi słowami `min_df` oraz częstymi słowami `max_df`.  
   - Naucz model, który zawsze zwraca jedną odpowiedź ([DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)).
   - Całość opakuj w [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) 
4. Implementacja metryki dokładności ([accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)).  
5. Zapisanie modelu za pomocą [joblib](https://joblib.readthedocs.io/en/latest/persistence.html#use-case).  

## Przetestowanie Baselinu na zbiorze testowym  
Wywołanie kodu poprzez:  
`python 2_uczenie_modelu/train_model.py test --model-path model_baseline.joblib`  

Do wypełnienia funkcja nr:  
6. Wczytanie modelu za pomocą [joblib](https://joblib.readthedocs.io/en/latest/persistence.html#use-case).  

Po wykonaniu zadania zapisz wartość uzyskanej dokładności. Wynik powinien wynosić ~32%.  

## Nauka Drzewa Decyzyjnego i przetestowanie na zbiorze testowym  
Wywołanie kodu poprzez:  
`python 2_uczenie_modelu/train_model.py train --model-type decision-tree --save-path model_df.joblib`  

Do wykonania funkcja nr:  
7. Naucz model Drzewa Decyzyjnego ([DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html))  
   - Zwróć uwagę na parametry, takie jak `max_features`, `criterion` czy `max_depth`.  

Po wykonaniu zadania zapisz wartość uzyskanej dokładności na zbiorze walidacyjnym.  
Na koniec przetestuj model za pomocą:  
`python 2_uczenie_modelu/train_model.py test --model-path model_df.joblib`  
Wynik na zbiorze testowym powinien być większy niż 65%. Porównaj tę wartość z dokładnością na zbiorze walidacyjnym.  

# Gratulacje!  
Przeszedłeś wszystkie zadania! Twój model jest gotowy do wdrożenia!
Jeśli zostało Ci czasu, możesz poeksperymentować z parametrami `CountVectorizer` i `DecisionTreeClassifier` w celu uzyskania lepszej jakości modelu.