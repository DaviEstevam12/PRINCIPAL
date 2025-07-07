##################### SERVICO #####################
def gera_celula(tabuleiro):
    'Sorteia uma célula vazia do tabuleiro e preenche com 2 ou 4. list --> list'
    from random import choice

    m=len(tabuleiro)
    n=len(tabuleiro[0])
    empty_cells=[]
    for i in range(m):
      for j in range(n):
        if tabuleiro[i][j]==0:
          empty_cells.append((i,j))

    if empty_cells==[]:
      return False

    chosen_cell = choice(empty_cells)
    i = chosen_cell[0]
    j = chosen_cell[1]

    tabuleiro[i][j] = choice([2,4])
    return True

def vitoria(tabuleiro,n=11):
  'Recebe o tabuleiro e um critério de vitória (expoente inteiro de dois) e indica se o tabuleiro é vitorioso. list,int-->bool'
  for i in tabuleiro:
    for j in i:
      if j==2**n:
        return True
  return False

def game_over(tabuleiro):
  '''Indica se há movimentos possíveis (True) ou não (False). list-->bool'''
  movimentos = [move_esquerda, move_direita, move_cima, move_baixo]
  for f in movimentos:
    copia = [linha[:] for linha in tabuleiro]
    f(copia)
    if copia != tabuleiro:
      return True
  return False

def move_esquerda(tabuleiro):
  'Recebe o tabuleiro e realiza o movimento para a esquerda. list-->'

  m = len(tabuleiro)
  # Removendo os zeros
  for i in range(m):
    lista = [x for x in tabuleiro[i] if x!=0]

  # Efetuando as somas
    lista1 = []
    j = 0
    while j < len(lista):
      if j<len(lista)-1 and lista[j]==lista[j+1]:
        lista1.append(2*lista[j])
        j = j+2
      else:
        lista1.append(lista[j])
        j = j+1

  # Posicionando os zeros (à direita)
    while len(lista1)<len(tabuleiro[i]):
      lista1.append(0)
    tabuleiro[i]=lista1

def move_direita(tabuleiro):
  'Recebe o tabuleiro e realiza o movimento para a direita. list-->'

  m = len(tabuleiro)
  # Removendo os zeros
  for i in range(m):
    lista = [x for x in tabuleiro[i] if x!=0]

  # Efetuando as somas
    lista1 = []
    j = len(lista)-1
    while j > -1:
      if j>0 and lista[j]==lista[j-1]:
        lista1.insert(0,2*lista[j])
        j = j-2
      else:
        lista1.insert(0,lista[j])
        j = j-1

  # Posicionando os zeros (à esquerda)
    while len(lista1)<len(tabuleiro[i]):
      lista1.insert(0,0)
    tabuleiro[i]=lista1

def transposta(tabuleiro):
  'Recebe o tabuleiro e o transforma na sua transposta. list-->'
  m = len(tabuleiro)
  n = len(tabuleiro[0])
  matriz=[]
  for i in range(m):
    lista=[]
    for j in range(n):
      lista.append(tabuleiro[j][i])
    matriz.append(lista)

  for i in range(m):
    tabuleiro[i]=matriz[i]

def move_baixo(tabuleiro):
  'Recebe o tabuleiro e realiza o movimento para baixo. list-->'
  transposta(tabuleiro)
  move_direita(tabuleiro)
  transposta(tabuleiro)

def move_cima(tabuleiro):
  'Recebe o tabuleiro e realiza o movimento para cima. list-->'
  transposta(tabuleiro)
  move_esquerda(tabuleiro)
  transposta(tabuleiro)

  
  ############################ Função Extra #############################
def celulas(tabuleiro,celula):
    'Função que recebe e valida as celulas'
    tab = [0,1,2,3]
    while True:
        celula = input('Insira a linha e coluna da célula para troca:')


        linha_str , coluna_str = celula.split(',')
        linha = int(linha_str)
        coluna = int(coluna_str)

        if linha in tab and coluna  in tab:
          return linha , coluna
        else:
            print("Erro: Coordenadas fora do tabuleiro. A linha e a coluna devem estar entre 0 e 3.")

def habilidade_troca(tabuleiro):
    'Função que ativa a habilidade de troca'
    print("--- Habilidade de Troca Ativada ---\nVocê pode trocar o valor de duas células de posição.")
    
    l1, c1 = celulas(tabuleiro, "primeira célula")
    l2, c2 = celulas(tabuleiro, "segunda célula")

    valor1 = tabuleiro[l1][c1]
    valor2 = tabuleiro[l2][c2]
    tabuleiro[l1][c1], tabuleiro[l2][c2] = tabuleiro[l2][c2], tabuleiro[l1][c1]



##################### APRESENTACAO #####################


def mostra_tabuleiro(tabuleiro):
  'Recebe o tabuleiro e exibe ao usuário. list-->'
  for linha in tabuleiro:
    for num in linha:
       print(f'{num:<7}',end = '')
    print("\n\n")

def pede_jogada():
  'Lê do usuário a jogada desejada e a retorna.'
  while True:
    tecla = input('Pressione W,A,S ou D:').upper()
    if tecla in ['W','A','S','D']:
      return tecla

def checa_habilidade():
    'pergunta se o jogador vai querer usar a habilidade de troca'
    resposta = input('Deseja utilizar a habilidade de troca de células? S ou N:').upper()
    if resposta in ['S','N']:
        return resposta

##################### MAIN #####################


def main():
  'Função principal do programa 2048.'
  tabuleiro = [4*[0] for _ in range(4)]
  gera_celula(tabuleiro)
  gera_celula(tabuleiro)

  habilidade_troca_usada = False
  
  while vitoria(tabuleiro,n=11)==False:
    print()
    mostra_tabuleiro(tabuleiro)
    print()
    if game_over(tabuleiro) == False:
        if not habilidade_troca_usada:
            quer_usar = checa_habilidade()
            if quer_usar == 'S':
                habilidade_troca(tabuleiro)
                habilidade_troca_usada = True
                continue 
            else:
                print('GAME OVER! Você optou por não usar a habilidade.')
                break
        else:
            print('GAME OVER! Sem mais movimentos possíveis.')
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

    
  if vitoria(tabuleiro,n=11)==True:
        print('Parabéns, você venceu!')

main()
