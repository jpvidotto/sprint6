import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('dataframe/games.csv')

# Tratamento inicial dos dados
df.columns = df.columns.str.lower() # Colunas em letras minúsculas

df['year_of_release'] = df['year_of_release'].astype('Int64')
# Transformei em Int64 para manter os valores nulos

df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce') 
# Transformei todos os valores em nulos, pois os que estão sem avaliação poderiam atrapalhar a análise

df['rating'] = df['rating'].fillna('Unknown')
# Preencher valores nulos com 'Unknown'

df = df.dropna(subset=['name']) #2 valores removidos 
df = df.dropna(subset=['year_of_release']) # #244 valores removidos

#Realizei todo o tramamento inicial dos dados conforme solicitado. Os valores ausentes e tbd eu transformei em NaN para facilitar a manipulação dos dados, visto que se tivesse preenchido eles, com média ou mediana, poderia influenciar negativamente na análise futura.

total_sales = df[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)
df['total_sales'] = total_sales 
# Nova coluna com a soma das vendas

#Lançamento de Jogos a cada ano:

#Ordena os jogos por ano de lançamento, do mais antigo ao mais novo e remove os duplicados para assim ter somente o ano original que o game lançou
df_unique = df.sort_values('year_of_release').drop_duplicates(subset='name', keep='first')
df_unique_gb = df_unique.groupby('year_of_release')['name'].count()

#print(df_unique_gb)

df_gb_totalsales_platform = df.groupby(['platform', 'year_of_release' ])['total_sales'].sum()

#print(df_gb_totalsales_platform)

df_top10_platforms = df.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
#print(df_top10_platforms.head(10))

top_platforms = df_top10_platforms.head(10).index.tolist()
print("Top 10 plataformas:", top_platforms)

# Filtrar dados apenas para essas plataformas
df_top = df[df['platform'].isin(top_platforms)]

# Criar tabela de vendas por ano e plataforma
vendas_ano_plataforma = df_top.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index().sort_values())
print(vendas_ano_plataforma)