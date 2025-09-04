import numpy as np
import matplotlib.pyplot as plt

#Define a função
def f(x):
    return (x**3) / (x**2 + 1)

#Gera os valores de x
x = np.linspace(-10, 10, 400)
y = f(x)

#Plota o gráfico da função
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f(x) = \frac{x^3}{x^2 + 1}$", color='blue')

#Adiciona o eixo x e y
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color= 'black', linewidth=0.5)

#Plota a assíntota oblíqua y = x
plt.plot(x,x, linestyle='--', color='red', label='Assíntota oblíquoa: $y = x$')

#Personaliza o gráfico
plt.title("Gráfico de $f(x) = \\frac{x^3}{x^2 + 1}$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.ylim(-10,10)

#Mostra o gráfico
plt.show()

import numpy as np
import matplotlib.pyplot as plt

#Define a função
def f(x):
    return x / (x**3 - 1)

# Intervalos separados para evitar x = 1
x1 = np.linspace(-10, 0.99, 500)
x2 = np.linspace(1.01, 10, 500)

# Calcula y
y1 = f(x1)
y2 = f(x2)

# Cria o gráfico
plt.figure(figsize=(10, 6))
plt.plot(x1, y1, label=r"$y = \frac{x}{x^3 - 1}$", color='blue')
plt.plot(x2,y2, color='blue')

# Assíntotas
plt.axvline(x=1, color='red', linestyle='--', label='Assíntota vertical em x=1')
plt.axhline(y=0, color='gray', linestyle=':', label='Assíntota horizontal em y=0')

# Destaques
plt.title('Gráfico da função $y = \\frac{x}{x^3 - 1}$')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-2,2)
plt.grid(True)
plt.legend()

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# definindo a função
def f(x):
    return x*np.exp(x)

#criando valores de x
x = np.linspace(-3,3,500)
y = f(x)

#criando o gráfico
plt.figure(figsize=(8,5))
plt.plot(x,y, label=r'$f(x) = x e^x$', color='purple')

#Ajustes do gráfico
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth= 0.8)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gráfico de f(x) = x.e^x")
plt.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define a função
def f(x):
    return x / (x**2 - 9)

#Cria intervalos separados para evitar os pontos de descontinuidade(x = -3 e x = 3)
x1 = np.linspace(-10, -3.01, 500)
x2 = np.linspace(-2.99, 2.99, 500)
x3 = np.linspace(3.01, 10, 500)

#Calculo os valores de y
y1 = f(x1)
y2 = f(x2)
y3 = f(x3)

# Cria o gráfico
plt.figure(figsize=(10, 6))
plt.plot(x1, y1, label="y = x / (x²- 9 )", color='blue')
plt.plot(x2,y2, color='blue')
plt.plot(x3,y3, color='blue')

#Assíntotas horizontais em y = 0
plt.axhline(y = 0, color='gray', linestyle=':')

#Destaque e labels

plt.title(r'Gráfico de $y = \frac{x}{x^2 - 9}$')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-2,2)
plt.grid(True)
plt.legend()

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define a função original
def f(x):
    return (x**2 - 4) / (x**2-2*x)

# Gera o valoresde x, evitando os pontos de descontinuidade
x1 = np.linspace(-10,-0.01,500)
x2 = np.linspace(0.01,1.99,500)
x3 = np.linspace(2.01,10,500)

# Ponto removível em x = 2
x_hole = 2
y_hole = (2+2) / 2 # Função simplificada: y (x + 2) / x

#Plota o gráfico
plt.figure(figsize=(10, 6 ))
plt.plot(x1,f(x1), 'b', label=r"$y=\frac{x^2-4}{x^2-2x}$")
plt.plot(x2,f(x2), 'b')
plt.plot(x3,f(x3), 'b')
plt.plot(x_hole, y_hole, 'ro', label='Buraco em x=2')

#Assíntotas
plt.axvline(x = 0, color='r', linestyle='--', label='Assíntota vertical: x=0')
plt.axhline(y=1, color='g', linestyle='--', label='Assíntota horizontal: y=1')

#Eixos e título
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title("Gráfico de $y=\\frac{x^2-4}{x^2-2x}$")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-10,10)
plt.legend()
plt.grid()

plt.show()

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define a variável simbólica
x = sp.symbols('x')

# Define a função
y = x* sp.tan(x)

# Deriva a função 1º e 2º vez
y_prime = sp.diff(y, x)
y_double_prime = sp.diff(y_prime, x)

# Mostra as derivadas
print("y(x) = ",y)
print("y'(x)=", y_prime)
print("y''(x) = ", sp.simplify(y_double_prime))

# Converte a função numérica para plotar
f = sp.lambdify(x, y, 'numpy')
f_prime = sp.lambdify(x, y_prime, 'numpy')
f_double_prime = sp.lambdify(x, y_double_prime, 'numpy')

# Cria vetor de valores de x, evitando os pontos de descontinuidade da tangente
x_vals = np.linspace(-1.4,1.4,1000) # Intervalo menor que +-pi/2 para evitar assíntotas
x_vals = x_vals[np.abs(x_vals - np.pi/2) > 0.1] # Exclui pontos perto de pi/2

# Avalia as funções
y_vals = f(x_vals)
y1_vals = f_prime(x_vals)
y2_vals = f_double_prime(x_vals)

# Plota os gráficos
plt.figure(figsize=(10,6))
plt.plot(x_vals, y_vals, label="y(x) = x*tan(x)")
plt.plot(x_vals, y1_vals, '--', label="y'(x)")
plt.plot(x_vals, y2_vals, ':', label="y''(x)")
plt.axhline(0, color='black', linewidth=0.5)
plt.title("Função, primeira e segunda derivadas de y = x*tan(x)")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define a função
def f(x):
    return x**5 - 5*x

#Gera os valores de x
x = np.linspace(-3, 3, 100)
y = f(x)

# Plota o gráfico
plt.figure(figsize=(8,6))
plt.plot(x, y, label=r"$y = x^5 - 5x$", color='blue')
plt.axhline(0, color='black', linewidth= 0.8)
plt.axvline(0, color='black', linewidth= 0.8)
plt.title("Gráfico da função $y = x^5 - 5x$")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()

from scipy.optimize import minimize_scalar

# Receita e custo

R = lambda x: 100 * x - 0.5 * x**2
C = lambda x: 20 * x + 100
L = lambda x: (R(x) - C(x)) # Negativo para minimizar

res = minimize_scalar(L, bounds=(0, 100), method='bounded')
x_opt = res.x
lucro_max = -res.fun

print(f"Produção ótima: {x_opt:.2f} unidades")
print(f"Lucro máximo: {lucro_max:.2f}")

import numpy as np
import matplotlib.pyplot as plt

#Define a função
def f(x):
    return 8*x**2 - x**4
# Cria os valores de x
x= np.linspace(-5,5,400)
y = f(x)

#Plota o gráfico
plt.figure(figsize=(8,6))
plt.plot(x, y, color='darkblue', label=r'$f(x) = 8x^2 - x^4$')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth= 0.5)
plt.title("Gráfico de $f(x) = 8x^2 - x^4$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# Parâmetros
r = 0.5 # taxa de crescimento
K = 1000 # capacidade de suporte

# Modelo logístico
def logistic(t,P):
    return r * P * (1 - P / K)

# Solução numérica
sol = solve_ivp(logistic, [0, 20], [100], t_eval=np.linspace(0, 20, 100))

# Plotar

plt.plot(sol.t, sol.y[0])
plt.title("Crescimento logístico da população")
plt.xlabel("Tempo")
plt.ylabel("População P(t)")
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define a função
def f(x):
    return (x**2 + 7*x + 3) / x**2

# Cria um intervalo de valores de x, evitando x = 0 para não dividir por zero
x = np.linspace(-10, 10, 1000)
x = x[x != 0] # Remove x = 0

# Calcula os falores de f(x)
y = f(x)

# Cria o gráfico
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f(x) = \frac{x^2 + 7x + 3}{x^2}$")
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.axvline(x=0, color='red', linestyle='--', label='Assíntota vertical x=0')
plt.ylim(-10,50)
plt.legend()
plt.title("Gráfico de uma função racional")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Intervalo de x
x = np.linspace(0,2,200)
y = x**3

# Criando o gráfico
plt.figure(figsize=(6,4))
plt.plot(x,y, color='blue', linewidth=2, label=r'$y = x^3$')

# Configurações do gráfico
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curva y = x^3')
plt.grid(True)
plt.legend()
plt.show()

def somar(v1,v2):
    if len(v1) != len(v2):
        raise ValueError("Os vetores devem ter o mesmo tamanho.")
    res = []
    for i in range(len(v1)):
        res.append(v1[i] + v2[i])
    return tuple(res)


def inverter(v):
    res = []
    for i in range(len(v)):
        res.append(-v[i])
    return tuple(res)


def subtrair(v1,v2):
    return somar(v1, inverter(v2))


def multiplicar(vet,num):
    return (num * vet[0], num * vet[1])


def produto_escalar(v1,v2):
    for i in range(len(v1)):
        if len(v1) != len(v2):
            raise ValueError("Os vetores devem ter o mesmo tamanho.")
    soma = 0
    for i in range(len(v1)):
        soma += v1[i] * v2[i]
    return soma


if __name__ == '__main__':
    a = (1,5)
    b = (1.5,7.0)
    num = 4

    print(f'Soma dos vetores {a} e {b}', somar(a,b))
    print(f'Diferença dos vetores {a} e {b}', subtrair(a,b))
    print(f'Multiplicação de {a} por {num}', multiplicar(a,b))
    print(f'Produto escalar entre {a} e {b}', produto_escalar(a,b))


import numpy as np
from numba import jit
import time

# Função normal em Python
def soma_quadrados(n):
    total = 0
    for i in range(n):
        total += i**2
    return total

# Função acelerada com Numba
@jit(nopython=True)  # nopython=True gera a versão mais rápida
def soma_quadrados_numba(n):
    total = 0
    for i in range(n):
        total += i**2
    return total

# Teste de tempo
N = 100_000_000

start = time.time()
soma_quadrados(N)
print("Python puro:", time.time() - start, "segundos")

start = time.time()
soma_quadrados_numba(N)
print("Numba JIT:", time.time() - start, "segundos")

import numpy as np
import plotly.express as px

# Dados
x = np.random.rand(50)
y = np.random.rand(50)
cores = np.random.rand(50)

# Criando gráfico
fig = px.scatter(x=x, y=y, color=cores, size=cores,
                 labels={'x':'X aleatório', 'y':'Y aleatório'},
                 title='Gráfico de pontos colorido')

fig.show()

#Diagrams for the region bounded by y = sqrt(x) and y = x^2, rotated about the y-axis.
import numpy as np
import matplotlib.pyplot as plt

# Curves
x = np.linspace(0,1,400)
y1 = np.sqrt(x)  # upper curve
y2 = x**2        # upper curve

# -- Diagram 1: Washers (integration in y)
plt.figure(figsize=(6,5))
plt.plot(x,y1, label='y = sqrt(x)')
plt.plot(x,y2, label='y = x^2')
plt.fill_between(x,y2,y1, alpha=0.2)

# draw a horizontal slice to indicate washers method

y_slice = 0.6
x_left = y_slice**2
x_right = np.sqrt(y_slice)
plt.hlines(y_slice, x_left, x_right, linewidth=3)
plt.vlines([x_left, x_right], y_slice-0.03, y_slice+0.03)

#axis of rotation
plt.vlines(0,0,1, linestyles='--', label='eixo y')
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('fatiamento por anéis(washers): integrar em y')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# -- Diagram 2: Cylindrical shells (integration in x)
plt.figure(figsize=(6,5))
plt.plot(x,y1, label='y = sqrt(X)')
plt.plot(x,y2, label='y = x^2')
plt.fill_between(x,y2,y1, alpha=0.2)

#draw a vertical slice to indicate shells method
x_slice = 0.6
plt.vlines(x_slice, x_slice**2, np.sqrt(x_slice), linewidth=3)
plt.hlines([x_slice**2, np.sqrt(x_slice)], x_slice-0.02, x_slice+0.02)

#axis of rotation
plt.vlines(0,0,1, linestyles='--', label='eixo y')
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('cascas cilíndricas: integrar em x')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Animar a rotação (2,5 voltas em 1,25 s) e gerar um GIF + gráfico θ(t)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------ Dados do problema ------------------
voltas = 2.5
tempo_total = 1.25  # s
omega = (voltas * 2 * np.pi) / tempo_total  # rad/s

# ------------------ Discretização temporal ------------------
fps = 60
frames = int(np.ceil(fps * tempo_total))
t = np.linspace(0, tempo_total, frames)
theta = omega * t

# ------------------ Animação: ponto girando ------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title("Rotação da Ginasta (2,5 voltas em 1,25 s)")

# Círculo de referência (borda sem preenchimento)
circle = plt.Circle((0, 0), 1, fill=False)
ax.add_patch(circle)

# Ponto que gira
point, = ax.plot([], [], 'o', markersize=10)

# Texto para mostrar tempo e ângulo
time_text = ax.text(-1.45, 1.35, '', fontsize=12)

def init():
    point.set_data([], [])
    time_text.set_text('')
    return point, time_text

def update(i):
    x = np.cos(theta[i])
    y = np.sin(theta[i])
    point.set_data(x, y)
    time_text.set_text(f"t = {t[i]:.2f} s | θ = {theta[i]:.2f} rad")
    return point, time_text

ani = FuncAnimation(fig, update, init_func=init, frames=len(t), interval=1000/fps, blit=False)

# Salvar GIF usando PillowWriter (dispensa ffmpeg)
gif_path = "/mnt/data/rotacao_ginasta.gif"
ani.save(gif_path, writer=PillowWriter(fps=fps))

plt.close(fig)  # fecha a figura da animação para evitar render duplicado

# ------------------ Gráfico estático θ(t) ------------------
plt.figure(figsize=(8, 5))
plt.plot(t, theta, linewidth=2)
plt.xlabel("Tempo (s)")
plt.ylabel("Ângulo acumulado, θ (rad)")
plt.title("Ângulo em função do tempo (ω constante)")
plt.grid(True)


plt.savefig(png_path, dpi=150)
plt.show()

import sympy as sp
# Definir a variável
x = sp.Symbol('x')
# Função a ser rotacionada
f = sp.sqrt(25 - x**2)

# Limites de integração
a = 2
b = 4

# Volume pelo método dos discos
volume = sp.pi * sp.integrate(f**2, (x, a, b))

# Mostrar resultado simbólico e numérico
print("Volume simbólico:", volume)
print("Volume numérico aproximado:", volume.evalf())

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Criação da malha
theta = np.linspace(0, 2 * np.pi, 100)
x = np.linspace(2, 4, 100)
X, Theta = np.meshgrid(x, theta)
Y = np.sqrt(25 - X**2)
Z = Y * np.cos(Theta)
W = Y * np.sin(Theta)

# Plotagem 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Z, W, color='lightcoral', alpha=0.8)

# Rótulos e título
ax.set_title("Sólido de revolução gerado por $y = \sqrt{25 - x^2}$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

import numpy as np
import pandas as pd

df = pd.DataFrame({
    'produto': ['notebook', 'smartphone', 'livro', 'câmera'],
    'preco': [3000, 2000, 100, 2500],
    'categoria': ['eletronico', 'eletronico', 'educacao', 'fotografia']
})

# Cálculo do imposto
def calcular_imposto(row):
    if row['categoria'] == 'eletronico':
        return row['preco'] * 1.3
    elif row['categoria'] == 'fotografia':
        return row['preco'] * 1.25
    else:
        return row['preco'] * 1.05

df['preco_com_imposto'] = df.apply(calcular_imposto, axis=1)

print(df)


condicoes = [
df['categoria'] == 'eletronico',
df['categoria'] == 'fotografia']
fatores = [1.3, 1.25]
default = 1.05

df['preco_com_imposto_v2'] = df['preco'] * np.select(condicoes, fatores, default = default)

import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

ativos = ['BBDC4.SA', 'IVVB11.SA', 'WEGE3.SA', 'HGLG11.SA', 'SMAL11.SA']
pesos = np.array([0.20, 0.10, 0.20, 0.30, 0.20])

# Usando 'Close' pois auto_adjust=True por padrão
carteira = yf.download(ativos, start='2021-01-01', end='2023-12-31')['Close']
carteira = carteira.dropna()  # Remove linhas com valores faltantes

retornos = carteira.pct_change().dropna()
retorno_carteira = (retornos * pesos).sum(axis=1)

plt.figure(figsize=(8,6))
sns.heatmap(retornos.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlação de retornos diários entre ativos')
plt.show()

cov_matriz = retornos.cov()
vol_carteira = np.sqrt(np.dot(pesos.T, np.dot(cov_matriz, pesos)))
print("Volatilidade da carteira:", vol_carteira)

import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np

# TESTE 1 - COMP_FIANCE_QUANT -  Statsmodels


# >>> Estatísitica Descritiva <<< #

# Captura de dados:

ticker = 'PETR4.SA'
data = yf.download(ticker,start='2020-01-01', end='2023-01-01')

# Calculando retornos diários
data['Return'] = data['Adj Close'].pct_change()

# Removendo NanS
returns = data['Return'].dropna()

# Estatística Descritiva
mean = np.mean(data['Return'])
std_dev = np.std(data['Return'])
skewness = sm.stats.stattools.skew(data['Return'])
kurtosis = sm.stats.stattools.kurtosis(data['Return'])

print(f"Mean: {mean:.4f}")
print(f"Standard Deviation: {std_dev:.4f}")
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")


# >>> Corralação entre duas ações <<< #

# Capturando Dados
stock1 = 'BBDC4.SA'
stock2 = 'BBAS3.SA'
data1 = yf.download(stock1, start='2020-01-01', end='2023-01-01')
data2 = yf.download(stock2, start='2020-01-01', end='2023-01-01')

# Calculando retornos diários
data1['Return'] = data1['Adj Close'].pct_change()
data2['Return'] = data2['Adj Close'].pct_change()

# Combinando os dois dataframes
returns = pd.DataFrame({stock1: data1['Return'], stock2: data2['Return']}).dropna()

# Calculando a correlação
# Calculate correlation using statsmodels
correlation = returns.corr().iloc[0, 1]
print(f"Correlation between {stock1} and {stock2}: {correlation:.4f}")

# >>> Fatores <<< #
import yfinance as yf
import pandas as pd
import statsmodels.api as sm

# Dados de exemplo (normalmente você obteriaisso de uma fonte financeira ou banco de dados)
# Fatores de Fama-French e retornos do ativo
factors = pd.DataFrame({
    'Rm-Rf': [0.01, 0.02, -0.01, 0.03, 0.01],
    'SMB': [0.02, 0.01, -0.02, 0.01, 0.03],
    'HML': [-0.01, 0.01, 0.02, 0.00, -0.02]
})
returns = pd.Series([0.015, 0.025, -0.005, 0.035, 0.015], name='Return')

# Adiciona a constante (intercepto) aos fatores
X = sm.add_constant(factors)
y = returns

# Ajuste do modelo de regressão múltipla
model = sm.OLS(y, X).fit()

# Sumário do modelo
print(model.summary())


gimport numpy as np
import matplotlib.pyplot as plt
# definindo a função
def f(x):
    return x**2 / np.sqrt(x + 1)
# criando valores de x (domínio da função x > -1)
x = np.linspace(-0.99,10,500)
y = f(x)
#criando o gráfico
plt.figure(figsize=(8,5))
plt.plot(x,y, label = r'$f(x) = \frac{x^2}{\sqrt{x+1}}$', color='blue')

#ajuste do gráfico
plt.axhline(0, color='black',linewidth= 0.8)
plt.axvline(0, color='black',linewidth= 0.8)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gráfico de f(x) = x² /   √(x + 1)")
plt.legend()
plt.grid(True)
plt.show()
