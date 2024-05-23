import json
import nltk
import spacy
import string
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from classes.RetirarCartNegacao import RetirarCartNegacao

#!python -m spacy download pt_core_news_lg
nltk.download('stopwords')
nltk.download('punkt')

cartoes_e_fibonacci = 'jsons/cartoes_fibonacci.json'
with open(cartoes_e_fibonacci, 'r') as file:
    cartoes_e_fibonacci = json.load(file)

# versao 3 - att 7 de maio
users_json = 'jsons/base-dados-aleatória-alternativas.json'
with open(users_json, 'r') as file:
    users_read = json.load(file)

users = users_read
frases = cartoes_e_fibonacci["Juntos"]
originais = cartoes_e_fibonacci["Juntos"].copy()
pesos_fibonacci = [34, 21, 13, 8, 5, 3, 2, 1, 1]

def vetorizar_respostas_usuario(respostas_usuario, max_features=7):
    vectorizer = TfidfVectorizer(max_features=max_features)
    respostas_usuario_vetorizadas = vectorizer.fit_transform(respostas_usuario) # tf-idf
    return respostas_usuario_vetorizadas

def plot_grafico_dispersao_teste1(X, aux, respostas_vetorizadas, vetor_medio_respostas, save_vetores):
    try:
        tsne = TSNE(n_components=2, random_state=42, init='random')
        X_tsne = tsne.fit_transform(X.toarray())
    except ValueError as e:
        if "perplexity must be less than n_samples" in str(e):
            st.error(f"{e}. \nO gráfico não será mostrado por conta que a perplexidade deve ser menor que n_samples, o que não ocorre nessa opção.")
            return
        else:
            raise

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:len(aux), 0], X_tsne[:len(aux), 1], color='blue', label='Frases')

    stacked_save_vetores = np.vstack([vetor.toarray().flatten() for vetor in save_vetores])

    offsets = np.linspace(-0.05, 0.05, len(stacked_save_vetores))
    for i, vetor_cartao in enumerate(stacked_save_vetores):
        plt.scatter(vetor_cartao[0] + offsets[i], vetor_cartao[1] + offsets[i], color='green', label='Cartões de Fibonacci' if i == 0 else "")

    plt.scatter(vetor_medio_respostas[0, 0], vetor_medio_respostas[0, 1], color='red', label='Vetor Médio das Respostas')
    plt.title('Dispersão das Frases, Respostas Vetorizadas e Média dos Vetores')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend()
    st.pyplot(plt)

def plot_grafico_dispersao_teste2(X, aux, respostas_vetorizadas, vetor_medio_respostas):
    try:
        tsne = TSNE(n_components=2, random_state=42, init='random')
        vetores_concatenados = np.concatenate((X.toarray(), respostas_vetorizadas), axis=0)
        X_tsne = tsne.fit_transform(vetores_concatenados)
    except ValueError as e:
        if "perplexity must be less than n_samples" in str(e):
            st.error(f"Erro: {e}. Certifique-se de que a perplexidade seja menor que o número de amostras.")
            return
        else:
            raise

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:len(aux), 0], X_tsne[:len(aux), 1], color='blue', label='Frases')
    plt.scatter(X_tsne[len(aux):, 0], X_tsne[len(aux):, 1], color='green', label='Respostas Vetorizadas')
    plt.scatter(vetor_medio_respostas[0, 0], vetor_medio_respostas[0, 1], color='red', label='Vetor Médio das Respostas')
    plt.title('Dispersão das Frases e Respostas Vetorizadas')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend()
    st.pyplot(plt)

def alternativa(escolha_teste, respostas_usuarios):
    posicoes_fibonacci = [1, 2, 3, 5, 8] # 5 num
    sequencia_fibonacci = [1, 2, 3, 5, 8, 13, 21, 34, 55] # 9 num
    lista_cartas_posicoes_fibonacci = []
    cartoes_top = []
    novo_cartoes = []
    save_vetores = []

    for usuario in respostas_usuarios:
        respostas = usuario['respostas'].values()
        frases = cartoes_e_fibonacci["Juntos"]
        retirar_cartoes_negacao = RetirarCartNegacao(cartoes_e_fibonacci)
        X, aux, novas_respostas, cartoes_usuario, pesos_fibonacci_novo = retirar_cartoes_negacao.retirar_cartoes(respostas, frases)

        st.write('Respostas do Usuário')
        st.write(' '.join(list(usuario['respostas'].values())))

        if escolha_teste == 1: 
            print('aqui um')
            respostas_vetorizadas = vetorizar_respostas_usuario(novas_respostas)
            vetor_medio = np.mean(respostas_vetorizadas.toarray(), axis=0)
            vetor_medio = vetor_medio.reshape(1, -1)
            similaridades = cosine_similarity(vetor_medio, X)
            cartoes_usuario_ordem = np.argsort(similaridades[0])
            indices_top5 = np.argsort(similaridades[0])[::-1]

            for indice in range(len(indices_top5)):
                cartoes_top.append(cartoes_usuario[indices_top5[indice]])

            for i in range(len(posicoes_fibonacci)):
                auxiliar = cartoes_top[posicoes_fibonacci[i]]
                novo_cartoes.append(auxiliar)

            vectorizer = TfidfVectorizer(max_features=7)
            novo_cartoes_vetorizados = vectorizer.fit_transform(novo_cartoes) # tf-idf
            novo_cartoes_vetorizados = novo_cartoes_vetorizados * (1.618)

            similaridades_nova = cosine_similarity(vetor_medio, novo_cartoes_vetorizados)
            indices_top5 = np.argsort(similaridades_nova[0])[:5][::-1]

            #lista_retorno = []
            for indice in range(len(indices_top5)):
                save_vetores.append(novo_cartoes_vetorizados[indices_top5[indice]])
                st.write(f"{indice+1} - {novo_cartoes[indices_top5[indice]]}")
                #resultado = f"{indice + 1} - {novo_cartoes[indices_top5[indice]]} --> {similaridades_nova[0][indices_top5[indice]]}"
                #lista_retorno.append(resultado)

            plot_grafico_dispersao_teste1(X, aux, respostas_vetorizadas, vetor_medio, save_vetores)
            #return lista_retorno
        
        elif escolha_teste == 2:
            respostas_vetorizadas = vetorizar_respostas_usuario(novas_respostas)
            respostas_vetorizadas = respostas_vetorizadas.toarray()
            novas_multiplicadoFI_respostas_vetorizadas = []
            for i, vetor in enumerate(respostas_vetorizadas):
                numero_fibonacci = sequencia_fibonacci[i]
                vetor_multiplicado = [valor * numero_fibonacci for valor in vetor]
                novas_multiplicadoFI_respostas_vetorizadas.append(vetor_multiplicado)

            vetor_medio = np.mean(novas_multiplicadoFI_respostas_vetorizadas, axis=0)
            vetor_medio = vetor_medio.reshape(1, -1)  # transformando em uni-dimensional

            similaridades = cosine_similarity(vetor_medio, X)
            indices_top5 = np.argsort(similaridades[0])[:5][::-1]

            st.write("Usuário:", usuario['user'])
            n = 0
            for indice in indices_top5:
                st.write(f"{n+1} - {cartoes_usuario[indice]}")
                n += 1

            plot_grafico_dispersao_teste2(X, aux, respostas_vetorizadas, vetor_medio)

def main():
    st.image("assets/logo_nuven.png", width=200)
    st.title("NUVEN")
    with open("jsons/json-dic-9perg-v3.json", "r") as file:
        data = json.load(file)

    perguntas = data["perguntas"]
    alternativas = data["alternativas"]

    respostas_usuarios = []
    st.header("Responda as perguntas")

    nome = st.text_input("Por favor, digite seu nome:")
    escolha_teste = 1

    if nome:
        respostas = {}
        for i in range(len(perguntas)):
            pergunta_escolhida = perguntas[i]
            alternativas_escolhidas = alternativas[i]
            resposta = st.radio(pergunta_escolhida, alternativas_escolhidas, key=i)
            respostas[pergunta_escolhida] = resposta

        if st.button("Enviar respostas"):
            respostas_usuario = {'user': nome, 'respostas': respostas}
            respostas_usuarios.append(respostas_usuario)
            st.write('Respostas Enviadas')
            # st.write("Respostas dos usuários armazenadas com sucesso:", respostas_usuarios)
            alternativa(escolha_teste, respostas_usuarios)

if __name__ == "__main__":
    main()

