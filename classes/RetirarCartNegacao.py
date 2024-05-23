from .PreprocessamentoCartoes import PreprocessamentoCartoes


class RetirarCartNegacao:
    def __init__(self, cartoes_e_fibonacci):
        self.cartoes_e_fibonacci = cartoes_e_fibonacci

    def retirar_cartoes(self, respostas_usuario, frases_cartoes):
        originais = self.cartoes_e_fibonacci["Juntos"].copy()
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
                              ['parceiro(a)', 'conjugue', 'companheiro', 'companheiro(a)'],
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
        preproc = PreprocessamentoCartoes()


        X_novo, aux_novo = preproc.preprocessar_cartoes(frases_novo)

        return X_novo, aux_novo, novas_respostas, originais, pesos_fibonacci_novo