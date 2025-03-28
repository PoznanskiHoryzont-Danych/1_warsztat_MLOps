# obraz bazowy z pythonem
FROM python:3.12.8-slim-bullseye

# argumenty z domyślną wartością
#   na wystawienie aplikacji na wszystkich interfejsach sieciowych
#   i na porcie 8080
ARG HOST=0.0.0.0
ARG PORT=8080

# ustawienie powyższych argumentów jako zmiennych środowiskowych
ENV HOST=$HOST
ENV PORT=$PORT

# ustawienie katalogu roboczego na /app
WORKDIR /app

# przekopiowanie listy requirementsów
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# kopiowanie pliku modelu i servingu w FastAPI
COPY model_df.joblib model_serving_4.py ./

# odpalenie servingu
CMD ["sh", "-c", "exec fastapi run model_serving_4.py --host $HOST --port $PORT"]
