def encontrar_indice_matriz(matriz,x):
    '''Recebe uma matriz(lista) e um inteiro e retorna um inteiro dessa matriz.
    list, int --> int'''
    indice_matriz = []
    for i in range(len(matriz)):
        for j in range(len(matriz[0])):
            if matriz[i][j] == x:
                indice_matriz.append((i,j))
    return indice_matriz

def multiplica_matriz(matriz,k):
    '''Recebe uma matriz(lista) e um inteiro e retorn uma outra lista com a multiplicação do inteiro na lista.
    list, int --> list'''
    nova_matriz = []
    for i in range(len(matriz)):
        linha = []
        for j in range(len(matriz[0])):
            linha.append(matriz[i][j] * k )
        nova_matriz.append(linha)
    return nova_matriz

def matriz_soma(A,B):
    '''Recebe duas matrizes(listas) e retorna a soma dela resultando em outra lista
    list,list --> list'''
    if len(A) == len(B) and len(A[0]) == len(B[0]):
        nova_matriz = []
        for i in range(len(A)):
            linha = []
            for j in range(len(A[0])):
                linha.append(A[i][j] + B[i][j])
        nova_matriz.append(linha)
    return nova_matriz

def deu_match(afinidade):
    '''recebe um dicionário e retorna uma lista
    dict --> list'''
    nova_matriz = []
    for pessoa, lista in afinidade.get():
        for outra_pessoa in lista:
            if pessoa in afinidade.get(outra_pessoa,[]):
                nova_matriz.append((pessoa,outra_pessoa))
    return nova_matriz                      
