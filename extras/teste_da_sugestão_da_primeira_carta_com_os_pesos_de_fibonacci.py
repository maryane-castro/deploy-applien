

"""# Setup e Definição de Funções"""



# setup
import json
import nltk
import spacy
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')

cartoes_e_fibonacci = 'jsons/cartoes_fibonacci.json'
with open(cartoes_e_fibonacci, 'r') as file:
    cartoes_e_fibonacci = json.load(file)



# versao 3 - att 7 de maio
users_json = 'jsons/base-dados-aleatória-alternativas.json'
with open(users_json, 'r') as file:
    users_read = json.load(file)



users = users_read
frases = cartoes_e_fibonacci["Juntos"]
originais = cartoes_e_fibonacci["Juntos"].copy()
pesos_fibonacci = [34, 21, 13, 8, 5, 3, 2, 1, 1]

# pré processamento dos cartões

import string

# -=-=-=-=-
def preprocessamento(lista_dos_cartoes):
  texts_preprocessed = []
  for text in lista_dos_cartoes:
      text_sem_pontuacao = text.replace('.', '').replace(',', '').replace('(','').replace(')','')
      texto_minusculo = text_sem_pontuacao.lower()

      palavras = texto_minusculo.split()
      palavras_separadas = []
      for palavra in palavras:
          if '-' in palavra:
              palavras_separadas.extend(palavra.split('-'))
          else:
              palavras_separadas.append(palavra)


      palavras_a_remover = ['ordem', 'semana', 'nessa','nesta','aspecto', 'aspectos','intensidade', 'cada','coloque', 'quais',
                            'resposta', 'tipo', 'enumere', 'diante', 'resto', 'só', 'coisas', 'mais', 'níveis', 'já', 'justifique',
                            'mais', 'dê', 'justifiquem','cite', 'três','hierarquize', 'identifique', 'de', 'e' ,'se', 'a', 'com',
                            'para', 'por', 'em', 'no', 'na', 'ao', 'à', 'sobre', 'sob', 'entre', 'sob', 'ante', 'após', 'até',
                            'contra', 'desde', 'durante', 'até', 'sem', 'sob', 'trás', 'para', 'perante', 'além', 'junto', 'longe',
                            'acima', 'abaixo', 'fora', 'através', 'dentro', 'fora', 'longe', 'próximo', 'atrás', 'depois', 'antes',
                            'quando', 'enquanto', 'assim', 'também', 'porque', 'mas', 'porém', 'portanto', 'contudo', 'embora', 'mesmo',
                            'senão', 'então', 'além', 'ainda', 'logo', 'pois', 'entretanto', 'porque', 'entanto', 'site', 'primeira',
                            'opções', 'pare', 'tire', 'nesse', 'acha', 'pouco', 'respostas', 'tomar', 'nesta', 'relação', 'precisa',
                            'bastante', 'todos', 'ligue', 'vá', 'desta', 'local', 'tipos', 'alguma', 'partes', 'onde', 'coluna', 'fundo',
                            'responda', 'nesta', 'quais', 'final', 'fazer', 'partir', 'órgãos', 'nele', 'nessa', 'favorito', 'principais',
                            'tempinho', 'tardinha', 'maior', 'vídeo', 'dando', 'possibilidades', 'moribundo', 'feita']


      palavras_filtradas = [palavra.strip(string.punctuation) for palavra in palavras_separadas if palavra not in palavras_a_remover]

      texto_preprocessado = ' '.join(palavras_filtradas)
      texto_preprocessado = ' '.join(texto_preprocessado.split())  # Remover espaços duplicados
      texts_preprocessed.append(texto_preprocessado)


  return texts_preprocessed



# -=-=-=-=-
"""
  Se for utilizar os pesos, não use lematização
"""
def lemmatization(textos_preprocessados, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("pt_core_news_lg") #, disable=["parser", "ner"]
    texts_out = []
    for text in textos_preprocessados:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text).lower()
        texts_out.append(final)
    return (texts_out)


# -=-=-=-=-
def remover_stopwords_digitos(textos_preprocessados):
    stop_words = set(stopwords.words('portuguese'))
    filtered_texts = []
    for text in textos_preprocessados:
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words and not word.isdigit()]
        filtered_texts.append(' '.join(filtered_text))
    return filtered_texts



# -=-=-=-=-
def vetorizacao(frases, max_features=None, is_pesos=False):
    vectorizer = TfidfVectorizer(max_features=max_features)
    vetores_tf_idf = vectorizer.fit_transform(frases)
    X = vetores_tf_idf
    pesos = np.array([5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 6, 5, 4, 3, 2, 1])
    if is_pesos:
        for i in range(len(frases)):
            X[i] *= pesos[i]
            #print(i, pesos[i])
    return X



# # def pesos
# def sem_pesos(lista_de_textos):
#   lista_de_textos = preprocessamento(lista_de_textos)
#   lista_de_textos = lemmatization(lista_de_textos)
#   print(lista_de_textos)
#   lista_de_textos_vetorizado = vetorizacao(lista_de_textos, 7, False)
#   aux = lista_de_textos
#   return lista_de_textos_vetorizado, lista_de_textos


def pesos(lista_de_textos):
  lista_de_textos = preprocessamento(lista_de_textos)
  lista_de_textos= remover_stopwords_digitos(lista_de_textos)
  #print(lista_de_textos)
  lista_de_textos_vetorizado = vetorizacao(lista_de_textos, 7, True)
  aux = lista_de_textos
  return lista_de_textos_vetorizado, lista_de_textos

# retirar alternativas que se referem a negação e cartões relacionados

def retirar_cartoes(respostas_usuario, frases_cartoes, tipo):
    originais = cartoes_e_fibonacci["Juntos"].copy()
    pesos_fibonacci = [34, 21, 13, 8, 5, 3, 2, 1, 1]

    alternativas_retirar = ['Minha vida é mais individual.',
                            'Não sinto nada em particular.',
                            'Opiniões alheias não afetam minha jornada.',
                            'Não tenho planejamento financeiro.',
                            'Não tenho filhos.',
                            'Não sou empresário(a) e/ou não tenho parceiros.',
                            'A aceitação do meu físico afeta minha satisfação pessoal.'
                            'Não tenho parceiro(a).',
                            'Não influenciam minha jornada.'
                            ]

    palavras_chave_alternativas = [
                          ['amigo', 'família', 'pais', 'amigo(a)', 'amigos', 'familiares', 'familia'],
                          ['morte', 'universo','reflita', 'espiritualmente'],
                          ['elogios', 'afetam', 'acha', 'pensam'],
                          ['planejamento', 'financeiro', 'luxo'],
                          ['filhos', 'filho', 'filhos,'],
                          ['empresário', 'parceiros', 'parcerias', 'subordinados', 'empresarial', 'empresario', 'empresarios', 'empresários', 'empresa', 'parceria'],
                          ['físico', 'fisicamente','espelho', 'roupas', 'fisico'],
                          ['parceiro(a)', 'conjugue'],
                          ['história', 'historia']]

    indices_a_remover = []
    novas_respostas = []
    indice_alternativa = []

    for resposta in respostas_usuario:
        for index, alternativa in enumerate(alternativas_retirar):
            if alternativa == resposta:
                indice_alternativa.append(index)
                for i, frase in enumerate(frases_cartoes):
                    for palavra_chave in palavras_chave_alternativas[index]:
                        if palavra_chave in frase:
                            indices_a_remover.append(i)
                            break
                break
        else:
            novas_respostas.append(resposta)

    frases_novo = [frases_cartoes[i] for i in range(len(frases_cartoes)) if i not in indices_a_remover]
    originais = [originais[i] for i in range(len(originais)) if i not in indices_a_remover]
    pesos_fibonacci_novo = [pesos_fibonacci[i] for i in range(len(pesos_fibonacci)) if i not in indice_alternativa]
    #print(pesos_fibonacci_novo)
    #print(frases_novo)

    if tipo == 1: # sem pesos
        #X_novo, aux_novo = sem_pesos(frases_novo)
        pass
    else:         # com pesos
        X_novo, aux_novo = pesos(frases_novo)

    return X_novo, aux_novo, novas_respostas, originais, pesos_fibonacci_novo,

def multiplicacao_fibonacci(respostas_usuario, pesos_fibonacci_alternativas):
    for i in range(respostas_usuario.shape[0]):
        respostas_usuario[i] = respostas_usuario[i].multiply(pesos_fibonacci_alternativas[i])
    return respostas_usuario


def vetorizar_respostas_usuario(respostas_usuario, max_features=7):
    vectorizer = TfidfVectorizer(max_features=max_features)
    respostas_usuario_vetorizadas = vectorizer.fit_transform(respostas_usuario) # tf-idf

    return respostas_usuario_vetorizadas

"""# Menu de Respostas Teste"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

def plot_grafico_dispersao(X, aux, respostas_vetorizadas, vetor_medio_respostas):
    tsne = TSNE(n_components=2, random_state=42, init='random')
    X_tsne = tsne.fit_transform(X.toarray())
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:len(aux), 0], X_tsne[:len(aux), 1], color='blue', label='Frases')

    for i, resposta_vectorized in enumerate(respostas_vetorizadas):
        similaridades = cosine_similarity(resposta_vectorized, X)
        indice_cartao_similar = np.argmax(similaridades)
        plt.scatter(X_tsne[indice_cartao_similar, 0], X_tsne[indice_cartao_similar, 1], color='purple')

    plt.scatter(vetor_medio_respostas[0, 0], vetor_medio_respostas[0, 1], color='green', label='Vetor Médio das Respostas')
    plt.title('Dispersão das Frases, Respostas Vetorizadas e Média dos Vetores')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend()
    plt.show()


def menu(respostas_usuarios, escolha):
    for usuario in respostas_usuarios:
        respostas = usuario['respostas'].values()
        frases = cartoes_e_fibonacci["Juntos"]
        X, aux, novas_respostas, originais, pesos_fibonacci_novo = retirar_cartoes(respostas, frases, 2)

        # print('Respostas do Usuário')
        # print(' '.join(list(usuario['respostas'].values())))

        respostas_vetorizadas = vetorizar_respostas_usuario(novas_respostas)


        if escolha == 1: # média aritmética
          print('Esolha 1')
          vetor_medio = np.mean(respostas_vetorizadas.toarray(), axis=0)
          media_type = "Aritmética"

        elif escolha == 2: # média ponderada fibonacci
          print('Escolha 2')
          vetor_medio = np.average(respostas_vetorizadas.toarray(), axis=0, weights=pesos_fibonacci_novo)
          media_type = "Ponderada"

        else:
          print('Escolha inválida')
          break




        vetor_medio = vetor_medio.reshape(1, -1)
        similaridades = cosine_similarity(vetor_medio, X)
        indices_top5 = np.argsort(similaridades[0])[-1:][::-1]

        for indice in indices_top5:
            print("\nUSUÁRIO:", usuario['user'])
            print(f"Cartão (Média {media_type}):", originais[indice])

        plot_grafico_dispersao(X, aux, respostas_vetorizadas, vetor_medio)

        print('-'*30)
        print('\n\n')

import json

def exibir_pergunta_e_alternativas(pergunta, alternativas):
    print(pergunta)
    for i, alternativa in enumerate(alternativas):
        print(f"{i+1}. {alternativa}")

def main():
    # datajlm = '/content/drive/MyDrive/[applien] IA NLP/Colab/BaseDeDados/json-dic-9perg-v3.json'
    # with open(datajlm, 'r') as file:
    #     data = json.load(file)
    with open("jsons/json-dic-9perg-v3.json", "r") as file:
       data = json.load(file)

    perguntas = data["perguntas"]
    alternativas = data["alternativas"]

    respostas_usuarios = []

    for _ in range(2):
        respostas_usuario = {}
        nome = input("Por favor, digite seu nome: ")
        respostas = {}

        for i in range(len(perguntas)):
            while True:
                pergunta_escolhida = perguntas[i]
                alternativas_escolhidas = alternativas[i]
                exibir_pergunta_e_alternativas(pergunta_escolhida, alternativas_escolhidas)

                resposta = input("Sua resposta (digite o número correspondente de 1 a 5): ")
                if resposta.isdigit() and 1 <= int(resposta) <= 5:
                    resposta_index = int(resposta) - 1
                    respostas[pergunta_escolhida] = alternativas_escolhidas[resposta_index]
                    break
                    print("Essa resposta não existe. Por favor, tente de 1 a 5.")

        respostas_usuario['user'] = nome
        respostas_usuario['respostas'] = respostas
        respostas_usuarios.append(respostas_usuario)
        print('\n')


    print("Respostas dos usuários armazenadas com sucesso:")
    for resposta_usuario in respostas_usuarios:
        print(resposta_usuario)

    while True:
        escolha = int(input('''Qual opção de respostas quer visualizar?
        [1] Média
        [2] Média Ponderada com os Pesos de Fibonacci
        [3] Sair
        Digite aqui: '''))


        if escolha == 3:
          break
        # resultados
        menu(respostas_usuarios, escolha)
        print('\n')


if __name__ == "__main__":
    main()


