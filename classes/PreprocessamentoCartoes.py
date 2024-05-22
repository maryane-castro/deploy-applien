# setup
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

#!python -m spacy download pt_core_news_lg
nltk.download('stopwords')
nltk.download('punkt')

class PreprocessamentoCartoes:
    def __init__(self):
        pass

    def preprocessamento(self, lista_dos_cartoes):
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

    def remover_stopwords_digitos(self, textos_preprocessados):
        stop_words = set(stopwords.words('portuguese'))
        filtered_texts = []
        for text in textos_preprocessados:
            word_tokens = word_tokenize(text)
            filtered_text = [word for word in word_tokens if word.lower() not in stop_words and not word.isdigit()]
            filtered_texts.append(' '.join(filtered_text))
        return filtered_texts

    def vetorizacao(self, frases, max_features=None, is_pesos=False):
        vectorizer = TfidfVectorizer(max_features=max_features)
        vetores_tf_idf = vectorizer.fit_transform(frases)
        X = vetores_tf_idf
        pesos = np.array([5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 6, 5, 4, 3, 2, 1])
        if is_pesos:
            for i in range(len(frases)):
                X[i] *= pesos[i]
        return X

    def preprocessar_cartoes(self, lista_de_textos):
        lista_de_textos_preproc = self.preprocessamento(lista_de_textos)
        lista_de_textos_filtrados = self.remover_stopwords_digitos(lista_de_textos_preproc)
        lista_de_textos_vetorizado = self.vetorizacao(lista_de_textos_filtrados, 7, True)
        return lista_de_textos_vetorizado, lista_de_textos_filtrados # X e Aux
