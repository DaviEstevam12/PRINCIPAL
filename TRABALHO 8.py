def qntd_elementos(palavra):
    '''Função que recebe uma string e devolve um inteiro indicando o número de letras nessa string.
    str --> int'''
    cont = 0
    for i in palavra:
        if ('a' <= i <= 'z') or ('A' <= i <= 'Z'): #NESSE TRECHO, PODE-SE USAR O ISALPHA() TAMBÉM. SEU USO É PARA VERIFICAR SE TODAS AS LETRAS PERTENCEM AO ALFABETO
            cont+= 1
    return cont

def tem_par(lista):
    '''Função que recebe uma lista de inteiros e retorna um booleano.
    list --> bool'''
    for n in lista:
        if n % 2 == 0:
            return True
    return False

def atualiza_mascara(palavra,lista,letra):
    '''Função que recebe uma string(palavra), uma lista e uma letra(string) e retorna um booleano.
    str,list,str --> bool'''
    #A função é basicamente um jogo de forca, onde tentaremos advinhar se a letra existe ou não na palavra. Bom, ao fazer isso, teremos a seguinte construção.
    palavra_certa = False
    for i in range(len(palavra)):
        if palavra[i] == letra:   #Se a letra que você escolheu estiver na palavra, colocaremos esta letra na palavra escondida e retornará um True
            lista[i] = letra
            palavra_certa = True 
    return palavra_certa   #Perguntei ao Gemini por que não poderia retornar um false fora do for. Isso faz com que o último valor só dependa do último caractere, ou seja, sempre retornará False.

def serie_convergente(termo):
    '''Recebe um inteiro e retorna a soma de inteiros
    int --> float'''
    soma = 0
    for n in range(termo+1):
        s = ((-1)**n) / (2*n+1)
        soma += s
    return soma

import math
def calculo_erro(ε):
    '''Recebe um inteiro que calculará a soma menor que ε que será o erro em relação π/4 e retorna uma tupla(n,soma) que é os termos da soma.
    int --> tupla'''
    soma = 0
    for n in range(1000000):  #Só consegui fazer esta função colocando um valor no range(100) por exemplo, perguntei ao Gemini por que funciona só assim, ele me respondeu que é necessário tomar um ''limite articficial'' por conta do ε ser muito pequeno. 
        s = ((-1)**n) / (2*n+1)
        soma += s         #Pode-se forçar um break aqui que seria um limite. Prefiro fazer desta forma e talvez mudar a função.
        erro = abs(soma - math.pi/4)
        if erro < ε:
            return n, soma
