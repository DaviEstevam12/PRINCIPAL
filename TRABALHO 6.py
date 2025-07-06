def pares(n):
    '''...'''
    i = 0
    lista = []
    while i <= n:
        if n % 2 ==0:
            lista.append(i)
        i += 1
    return lista

def vogais(palavra):
    '''...'''
    i = 0
    cont = 0
    while i < len(palavra):
        if palavra[i] in 'aeiou':
            cont += 1
        i += 1
    return cont

def filtra_vogais(palavra):
    '''...'''
    i = 0
    cont = []
    while i < len(palavra):
        if palavra[i] in 'aeiou':
            cont.append(palavra[i])
        i += 1
    return cont

def ultima_vogal(palavra):
    '''...'''
    i = 0
    cont = []
    while i < len(palavra):
        if palavra[i] in 'aeiou':
            cont.append(palavra[i])
        i += 1
    return cont[-1]

def elementos_pares(palavra):
    '''recebe uma lista de inteiros e retorna apenas os pares dessa lista
    list --> list'''
    i = 0
    pares = []
    while i < len(palavra):
        if palavra[i] % 2 == 0:
            pares.append(palavra[i])
        i+= 1
    return pares

def n_esima_ocorrencia(l,x,n):
    ''' l é uma lista, x é um inteiro, n é um inteiro. Função que remove n vezes x de l
    (lista, int, int --> lista)'''
    n = 0
    while n < l:
        if x in l:
            l.remove(x)
        i += 1
    return l

import random
def jogo(n):
    '''...'''
    jogador_1 = random.randint(pedra,papel,tesoura)
    jogador_2 = random.randint(pedra,papel,tesoura)
    if jogador_1 == jogador_2:
        return "Repita a jogada"

    while n <= 3:
        if jogador_1 == 'papel' and jogador_2 == 'pedra':
            return "jogador 1 é o vencedor"
        elif jogador_1 == 'pedra' and jogador_2 == 'tesoura':
            return "jogador 1 é o vencedor"
        elif jogador_1 == 'tesoura' and jogador_2 == 'papel':
            return "jogador 1 é o vencedor"
        
            
        
    
        
