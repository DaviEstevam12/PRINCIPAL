#JOGO 2048
#SERVIÇO#

def gera_celula(tabuleiro):
    'Sorteia uma célula vazia do tabuleiro e preenche com 2 ou 4. list -->list'
    '''RESUMO: A função:

    Encontra as posições vazias no tabuleiro.

    Sorteia uma dessas posições.

    Preenche essa posição com o número 2 ou 4.

    Retorna True se conseguiu inserir ou False se o tabuleiro estiver cheio.'''
    
    from random import choice

    m = len(tabuleiro)
    n = len(tabuleiro[0])
    empty_cells = []
    for i in range(m):
        for j in range(n):
            if tabuleiro[i][j] == 0:
                empty_cells.append((i,j))
    if empty_cells == []:
        return False
    chosen_cell = choice(empty_cells)
    i = chosen_cell[0]
    j = chosen_cell[1]

    tabuleiro[i][j] = choice([2,4])
    return True

def vitoria(tabuleiro,n=11):
    'Recebe o tabuleiro e um critério de vitória(expoente inteiro de dois) e indica se o tabuleiro é vitorioso.list,int --> bool'
    '''RESUMO: A função vitoria(tabuleiro, n=11) verifica se existe alguma célula no tabuleiro com o valor 2ⁿ.
    Se existir, retorna True (vitória). Caso contrário, retorna False.'''
    
    for i in tabuleiro:
        for j in i:
            if j == 2**n:
                return True
    return False

def game_over(tabuleiro):
    'Indica se há movimentos possíveis(True) ou não(False). list --> bool'
    '''RESUMO: A função verifica se ainda há movimentos possíveis no tabuleiro.
    Se algum movimento (esquerda, direita, cima ou baixo) modificar o tabuleiro, retorna True (ainda é possível jogar).
    Se nenhum movimento for possível, retorna False (fim de jogo).'''
    
    movimentos = [move_esquerda, move_direita, move_cima, move_baixo]
    for f in movimentos:
        copia = [linha[:] for linha in tabuleiro]
        f(copia)
        if copia != tabuleiro:
            return True
    return False

def move_esquerda(tabuleiro):
    'Recebe o tabuleiro e realiza o movimento para a esquerda.'
    '''RESUMO: A função realiza o movimento das peças para a esquerda em cada linha do tabuleiro, conforme as regras do jogo 2048:

    Remove os zeros da linha.

    Soma pares de números iguais consecutivos.

    Insere zeros à direita para manter o comprimento da linha.

    Atualiza o tabuleiro com a nova linha resultante.'''
    m = len(tabuleiro)
    #Remove os zeros
    for i in range(m):
        lista = [x for x in tabuleiro[i] if x != 0]
        lista1 = []
        j = 0
        while j < len(lista):
            if j < len(lista) -1 and lista[j] == lista[j+1]:
                lista1.append(2*lista[j])
                j += 2
            else:
                lista1.append(lista[j])
                j += 1
        while len(lista1) < len(tabuleiro[i]):
            lista1.append(0)
        tabuleiro[i] = lista1

        
def move_direita(tabuleiro):
    'Recebe o tabuleiro e realiza o movimento para a direita.'
    '''RESUMO: A função move todas as peças de cada linha do tabuleiro para a direita, aplicando as regras do 2048:

    Remove os zeros,

    Soma os pares adjacentes iguais (da direita para a esquerda),

    Preenche zeros à esquerda,

    Atualiza a linha com o novo estado.'''

    m = len(tabuleiro)
    #Remove os zeros
    for i in range(m):
        lista = [x for x in tabuleiro[i] if x!= 0]

    #Efetuando as somas
        lista1 = []
        j = len(lista)-1
        while j > -1:
            if j > 0 and lista[j] == lista[j-1]:
                lista1.insert(0,2*lista[j])
                j-=2
            else:
                lista1.insert(0,lista[j])
                j-=1
    #Posicionando os zeros à esquerda
        while len(lista1) < len(tabuleiro[i]):
            lista1.insert(0,0)
        tabuleiro[i] = lista1


def transposta(tabuleiro):
    'Recebe o tabuleiro e o transforma em sua transposta.'
    '''RESUMO: Essa função modifica o tabuleiro original, trocando as linhas pelas colunas (fazendo sua transposta).
    Isso é útil, por exemplo, para aplicar movimentos verticais (cima e baixo) reutilizando os movimentos horizontais (esquerda e direita).'''


    m = len(tabuleiro)
    n = len(tabuleiro[0])
    nova = [[tabuleiro[j][i] for j in range(m)] for i in range(n)]
    for i in range(m):
        tabuleiro[i] = nova[i]
        

def move_baixo(tabuleiro):
    'Recebe o tabuleiro e realiza o movimento para baixo.'
    '''RESUMO: Essa função move as peças do tabuleiro para baixo aplicando os seguintes passos:

    Transpõe o tabuleiro (linhas viram colunas).

    Aplica o movimento para a direita (agora atuando nas “novas linhas”).

    O efeito final é um movimento para baixo no tabuleiro original.'''

    transposta(tabuleiro)
    move_direita(tabuleiro)

def move_cima(tabuleiro):
    'Recebe o tabuleiro e realiza o movimento para cima.'
    '''A função move as peças do tabuleiro para cima da seguinte forma:

    Transpõe o tabuleiro (linhas viram colunas).

    Aplica o movimento para a esquerda nas linhas transpostas.

    O efeito final é um movimento vertical para cima no tabuleiro original.'''

    transposta(tabuleiro)
    move_esquerda(tabuleiro)

#FUNÇÃO EXTRA!
def celulas(tabuleiro):
    'Função que recebe e valida as celulas'
    '''RESUMO: Essa função:

    Solicita ao usuário que digite uma coordenada no formato linha,coluna (por exemplo: 2,3);
    Converte os valores para inteiros;
    Valida se estão dentro dos limites do tabuleiro (0 a 3);
    Se válidos, retorna (linha, coluna);
    Caso contrário, exibe uma mensagem de erro e repete o pedido.'''


    tab = [0,1,2,3]
    while True:
        celula = input('Insira a linha e a coluna da célula para troca:')
        linha_str , coluna_str = celula.split(',')
        linha = int(linha_str)
        coluna = int(coluna_str)
        if linha in tab and coluna in tab:
            return linha, coluna
        else:
            print("Erro:Coordenadas fora do tabuleiro.A linha e a coluna devem estar entre 0 e 3.")

def habilidade_troca(tabuleiro):
    'Função que ativa a habilidade de troca'
    '''RESUMO : Ativa uma habilidade especial onde o jogador pode trocar o valor de duas posições do tabuleiro.
    Solicita duas coordenadas válidas.
    Efetua a troca dos valores entre essas posições.
    Exibe mensagens explicativas.'''

    print("---Habilidade de troca ativada----\nVocê pode trocar o valor de duas células de posição.")
    l1, c1 = celulas(tabuleiro)
    l2, c2 = celulas(tabuleiro)
    if (l1,c1) == (l2,c2):
        print("As células escolhidas são iguais. Não houve troca.")
        return
    #Realiza a troca
    
    tabuleiro[l1][c1], tabuleiro[l2][c2] = tabuleiro[l2][c2] , tabuleiro[l1][c1]
    print(f"Troca realizada({l1},{c1}) <--> ({l2},{c2})")


#APRESENTAÇÃO
def mostra_tabuleiro(tabuleiro):
    'Recebe o tabuleiro e exibe ao usuário.'
    '''Exibe o tabuleiro (matriz) na tela de forma organizada e espaçada.

    Cada número ocupa uma largura fixa (6^), para que a visualização fique alinhada em colunas.

    Após cada linha, imprime duas quebras de linha ('|') para espaçamento vertical.'''


    for linha in tabuleiro:
        for num in linha:
            print(f"{num:6^}",end='|')
        print("\n" + "-"*29)

def pede_jogada():
    'Lê do usuário a jogada desejada e a retorna.'
    '''RESUMO: Solicita ao usuário que pressione uma tecla para indicar a jogada:

    W = cima

    A = esquerda

    S = baixo

    D = direita

    Converte a entrada para maiúscula (.upper()), permitindo aceitar letras minúsculas também.

    Valida se a tecla está entre as opções permitidas.

    Retorna a tecla válida.'''
    while True:
        tecla = input('Pressione W,A,S ou D:').upper()
        if tecla in ['W','A','S','D']:
            return tecla

def checa_habilidade():
    'Pergunta se o jogador vai querer usar a habilidade de troca.'
    resposta = input('Deseja utilizar a habilidade de troca de células? Sim ou Não?:').upper()
    if resposta in ['SIM','NÃO']:
        return resposta

#MAIN
def main():
    'Função principal do programa 2048.'
    tabuleiro = [4*[0] for _ in range(4)]
    gera_celula(tabuleiro)
    habilidade_troca_usada = False
    while not vitoria(tabuleiro,n=11):
        print()
        mostra_tabuleiro(tabuleiro)
        print()
        if not game_over(tabuleiro):
            if not habilidade_troca_usada:
                quer_usar = checa_habilidade()
                if quer_usar == 'SIM':
                    habilidade_troca(tabuleiro)
                    habilidade_troca_usada = True
                    continue
            print('GAME OVER')
            break
                    
        jogada = pede_jogada()
        if jogada == 'W':
            move_cima(tabuleiro)
        elif jogada == 'A':
            move_esquerda(tabuleiro)
        elif jogada == 'S':
            move_baixo(tabuleiro)
        elif jogada == 'D':
            move_direita(tabuleiro)
        gera_celula(tabuleiro)

    if vitoria(tabuleiro, n=11) == True:
        print('Você venceu, parabéns')

main()

        
        
    
            
        
