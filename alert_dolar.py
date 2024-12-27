import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas_ta as ta

# Função para enviar mensagem via Telegram
def send_telegram_message(message):
    token = "7654351807:AAHJ-Pve956eUVbRDMpF17iDG6Qv-V8zovM"
    chat_id = "-1002473077704"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    response = requests.get(url, params=params)
    return response.status_code

# Carregando os dados
@st.cache_data(ttl=24*60*60)  # Cache de 1 dia
def get_data():
    start_date = '2020-01-01'
    today = pd.to_datetime("today").strftime('%Y-%m-%d')
    data = yf.download("USDBRL=X", start=start_date, end=today, interval="1d")
    data.reset_index(inplace=True)
    data.columns = data.columns.droplevel(1)
    data.set_index('Date', inplace=True)
    data.dropna(inplace=True)
    return data

# Função para processar os dados
def preprocess_data(data):
    # Cálculo do Estocástico Lento
    stochastic = ta.stoch(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowk_matype=1)
    data['Stochastic_K'] = stochastic['STOCHk_14_3_3']

    # Calculando RSI
    data['RSI'] = ta.rsi(data['Close'], length=14)

    # Calculando Bandas de Bollinger
    bb = ta.bbands(data['Close'], length=20)
    data['Bollinger_Upper'] = bb['BBU_20_2.0']
    data['Bollinger_Lower'] = bb['BBL_20_2.0']

    # Identificando pontos de venda
    data['Sell_Signal'] = (data['Stochastic_K'] > 80)
    return data.dropna()

# Função para treinar o modelo
@st.cache_data
def train_model(data):
    X = data[['RSI', 'Bollinger_Upper', 'Close', 'Stochastic_K']]
    y = data['Sell_Signal'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinando o modelo
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Interface do Streamlit
st.title("Alerta Diário de Vendas - Mercado de Dólar")
st.write("Este aplicativo verifica diariamente os indicadores de venda e envia um alerta ao Telegram.")

# Botão para atualizar dados e realizar previsão
if st.button("Executar análise diária"):
    with st.spinner("Carregando dados e realizando análise..."):
        data = get_data()
        data = preprocess_data(data)
        model = train_model(data)

        # Obter o último dado
        latest_data = data.iloc[-1]
        X_latest = latest_data[['RSI', 'Bollinger_Upper', 'Close', 'Stochastic_K']].values.reshape(1, -1)
        X_latest = pd.DataFrame(X_latest, columns=['RSI', 'Bollinger_Upper', 'Close', 'Stochastic_K'])

        # Previsão
        prediction = model.predict(X_latest)

        # Geração de mensagem
        if prediction == 1:
            message = (f"**ALERTA**: O modelo identificou um momento de venda para o dólar!  "
                       f"Último Preço do Dólar: R${latest_data['Close']:.2f}, "
                       f"RSI: {latest_data['RSI']:.2f}, "
                       f"Estocástico Lento: {latest_data['Stochastic_K']:.2f}")
        else:
            message = (f"Nenhum alerta no momento.  Último Preço do Dólar: R${latest_data['Close']:.2f}, "
                       f"RSI: {latest_data['RSI']:.2f}, "
                       f"Estocástico Lento: {latest_data['Stochastic_K']:.2f}")

        # Exibir mensagem na interface
        st.success("Análise concluída!")
        st.write(message)

        # Enviar mensagem para o Telegram
        if send_telegram_message(message) == 200:
            st.success("Mensagem enviada ao Telegram com sucesso!")
        else:
            st.error("Erro ao enviar mensagem ao Telegram.")
