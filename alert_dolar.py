import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta
import numpy as np

from datetime import datetime, timedelta, date

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#obtendo dados
start_date = date(2010, 1, 1)
today = date.today()
end_date = today.strftime('%Y-%m-%d')
data = yf.download("USDBRL=X", start=start_date, end=end_date, interval="1d")

#tratando
data.reset_index(inplace=True)
data.columns = data.columns.droplevel(1)
data.set_index('Date', inplace=True)
data.dropna()

#RSI
data['RSI'] = ta.rsi(data['Close'], length=14)

#Bandas de Bollinger
bb = ta.bbands(data['Close'], length=20)
data['Bollinger_Upper'] = bb['BBU_20_2.0']
data['Bollinger_Lower'] = bb['BBL_20_2.0']

# Identificando possíveis pontos de venda
data['Sell_Signal'] = (data['RSI'] > 70) & (data['Close'] >= data['Bollinger_Upper'])

#criando um modelo
dataset = data.dropna()
#[['RSI', 'Bollinger_Upper', 'Close', 'Sell_Signal']]
# Separando variáveis independentes (X) e dependentes (y)
X = dataset[['RSI', 'Bollinger_Upper', 'Close']]
y = dataset['Sell_Signal'].astype(int)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Avaliando
y_pred = model.predict(X_test)

#função de alerta via telegram
def send_telegram_message(message):
    token = "8059277119:AAEGcOtpnyxjQMCRPf-0Brl7FTsQpIAnVQs"
    chat_id = "SEU_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    response = requests.get(url, params=params)
    return response.status_code

# Configurando a interface Streamlit
st.title("Alerta de Venda de Dólar com IA")

# Exibindo os dados mais recentes
latest_data = data.iloc[-1]
st.write(f"Último Preço do Dólar: R${latest_data['Close']:.2f}")
st.write(f"RSI Atual: {latest_data['RSI']:.2f}")

# Previsão do modelo
X_latest = latest_data[['RSI', 'Bollinger_Upper', 'Close']].values.reshape(1, -1)
prediction = model.predict(X_latest)

if prediction == 1:
    st.warning(f"**ALERTA**: O modelo identificou um momento de venda para o dólar!")
else:
    st.success("Nenhum alerta no momento.")

# Gráfico do preço do dólar
st.line_chart(data[['Close']])

# Configuração para enviar alerta
if prediction == 1:
    message = f"ALERTA: O modelo identificou um momento de venda para o dólar. RSI: {latest_data['RSI']:.2f}, Preço: {latest_data['Close']:.2f}"
    send_telegram_message(message)