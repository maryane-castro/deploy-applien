import streamlit as st

# Função para carregar dinamicamente o código do arquivo selecionado
def load_app(selected_app):
    with open(selected_app, 'r') as file:
        exec(file.read(), globals())

# Lista de opções para a barra lateral
options = {
    "App V1": "app_v1.py",
    "App V2": "app_v2.py"
}

# Barra lateral para seleção do aplicativo
selected_app = st.sidebar.radio("Selecione o aplicativo", list(options.keys()))

# Carrega o aplicativo selecionado
load_app(options[selected_app])
