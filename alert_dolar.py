import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Função para enviar mensagem via Telegram
def send_telegram_message(message):
    token = "teutoker"
    chat_id = "oteu"
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
    # Definir a janela para cálculo do Estocástico Lento
    fastk_period = 14
    slowk_period = 3
    
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Cálculo do Estocástico Lento %K
    lowest_low = low.rolling(window=fastk_period).min()
    highest_high = high.rolling(window=fastk_period).max() 
    # Cálculo do %K (Estocástico Lento)
    data['Stochastic_K'] = 100 * (close - lowest_low) / (highest_high - lowest_low)
    # Cálculo do %D (média móvel suavizada de %K com uma janela de 3 dias)
    data['Stochastic_D'] = data['Stochastic_K'].rolling(window=slowk_period).mean(

    # Calculando RSI 
    delta = close.diff()  # Diferença de preço entre o atual e anterior
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Média dos ganhos
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Média das perdas
    rs = gain / loss  # RS (Relative Strength)
    data['RSI'] = 100 - (100 / (1 + rs))

    # Calculando as Bandas de Bollinger
    sma = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std()
    data['Bollinger_Upper'] = sma + (2 * rolling_std)
    data['Bollinger_Lower'] = sma - (2 * rolling_std)

    #sinais de venda
    data['Sell_Signal'] = data['Stochastic_K'] > 80
    
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
