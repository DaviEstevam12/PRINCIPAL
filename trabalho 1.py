import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# TRABALHO REGRESSÃO LINEAR 1 MAT. FINANCES

# Dados fictícios de vendas
np.random.seed(42)
data = {
    'Investimento_Marketing': np.random.normal(2000, 5000, 100),  # Investimento em Marketing
    'Descontos_Oferecidos': np.random.normal(1500, 300, 100),     # Descontos oferecidos aos clientes
    'Número_de_vendedores': np.random.randint(10, 50, 100),       # Número de vendedores
    'Vendas': np.random.normal(50000, 12000, 100)                 # Total de vendas
}

# Criando DataFrame
df = pd.DataFrame(data)

# Ajustando o modelo de regressão linear múltipla usando fórmulas
model = smf.ols('Vendas ~ Investimento_Marketing + Descontos_Oferecidos + Número_de_vendedores', data=df).fit()

# Resumo do modelo com a tabela ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Exibindo os resultados
print("Resumo do Modelo:")
print(model.summary())
print("\nTabela ANOVA:")
print(anova_table)
