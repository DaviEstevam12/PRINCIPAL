#TRABALHO 7
def numero_de_vogais(palavra):
    '''recebe uma string e retorna um inteiro indicando o número de vogais.
    str --> int'''
    i = 0
    cont = 0                                       
    while i < len(palavra):
        if palavra[i] in 'aeiouAEIOU':
            cont += 1
        i += 1
    return cont

import random
def jogo_dados(x,y):
    '''recebe dois inteiros(numero de jogadas) e retorna outro inteiro indicando
    o numero de repetições até a igualdade desses números.
    int, int --> int, int'''
    cont = 0
    dado_1 = random.randint(1,x)
    dado_2 = random.randint(1,y)

    while dado_1 != dado_2:
        dado_1 = random.randint(1,x)
        dado_2 = random.randint(1,y)
        cont += 1
        print(f" Número de jogadas foi {cont}: 1º Dado {dado_1}, 2º Dado {dado_2}")  #O print foi usado para mostrar ao usuário o número de jogadas até vencer

    return cont

def n_esima_ocorrencia(lista,x,n):
    ''' l é uma lista, x é um inteiro, n é um inteiro. Função que remove n vezes x de l
    (lista, int, int --> lista)'''
    cont = 0
    while cont < n:
        if x in lista:                       #Essa questão foi fácil fazer pois tem um exemplo igual no pdf
            lista.remove(x)
            cont += 1
        return cont

def int_aleatorios(n,k):
    '''recebe dois inteiros (n,k) e retorna uma lista de números distintos aleatórios
    int(n),int(k) --> lista'''
    if k > n:
        return 'Erro'
    lista = []
    while len(lista) < k:
        num = random.randint(1,n)
        if num not in lista:
            lista.append(num)
    return lista


def divisao(x,y):
    '''recebe inteiros(x,y) sendo eles divisores e retora o quociente e o resto
    da divisão.
    int(x),int(y) --> int(x), int(y)'''
    
    resto = x #Pode-se fazer um exemplo pensando em x = 20 e y = 3. Na primeira ocorrência 20 = 3*6 + 2 r(x) = 2, Q(x) = 6 --> 20-6 = 14. Na segunda ocorrÊncia 14-6 = 8. Na terceira ocorrencia 8-6=2.
    quoc = 0                        
    while resto >= y:
        resto -= y
        quoc += 1
    return quoc, resto
        
            
        
        
    



        
            
