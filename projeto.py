import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

df_gb_totalsales_platform = df.groupby(['platform', 'year_of_release' ])['total_sales'].sum().reset_index()

#print(df_gb_totalsales_platform)

df_top_platforms_list = df.groupby('platform')['total_sales'].sum().sort_values(ascending=False)

top3_platforms = df_top_platforms_list.head(3).index.tolist()
#print("Top 3 plataformas:", top_platforms)

top10_platforms = df_top_platforms_list.head(10).index.tolist()
#print("Top 10 plataformas:", top10_platforms)

# Filtrar dados apenas para essas plataformas
df_top = df[df['platform'].isin(top3_platforms)]

# Criar tabela de vendas por ano e plataforma
vendas_ano_plataforma = df_top.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index().sort_values(by='year_of_release')
#print(vendas_ano_plataforma)

#plt.figure(figsize=(12, 6))
#plt.title('Vendas Totais por Ano para as 3 Principais Plataformas')
#sns.scatterplot(data=vendas_ano_plataforma, x='year_of_release', y='total_sales', hue='platform')
#plt.xlabel('Ano de Lançamento')
#plt.savefig('vendas_totais_ano.png')

#Análise de quantidade de vendas por plataforma ao longo dos anos:
#df.groupby('platform')['total_sales', 'year_of_release'].sum().sort_values(ascending=False)
df_platforms_1995 = df[df['year_of_release'] > 1995 ].sort_values('year_of_release')
df_platforms_1995 = df_platforms_1995.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index()
df_platforms_1995 = df_platforms_1995[df_platforms_1995['total_sales'] > 20]

#plt.title('Vendas Totais por Ano para todas as Plataformas')
#plt.figure(figsize=(12, 6))
#sns.lineplot(data=df_platforms_1995, x='year_of_release', y='total_sales', hue='platform')
#plt.xlabel('Ano de Lançamento')
#plt.savefig('vendas_totais_ano_todas_plataformas.png')

#Antes de 2010 as plataformas tinham uma vida média de 7 anos, após 2010 essa média caiu para 5 anos. Isso pode ser explicado pelo avanço tecnológico e a rápida evolução do mercado de jogos, onde novas plataformas são lançadas com mais frequência para atender às demandas dos consumidores por melhores gráficos, desempenho e funcionalidades. Além disso, a crescente popularidade dos jogos móveis e serviços de streaming de jogos pode ter contribuído para a redução da vida útil das plataformas tradicionais.

#Irei pegar dados de 2013 para frente, visto que é um período mais recente e relevante para análise de mercado atual.
df_recent = df[df['year_of_release'] >= 2013].sort_values('year_of_release')
df_recent = df_recent.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index()

plt.title('Vendas Totais por Ano para todas as Plataformas (2013-2016)')
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_recent, x='year_of_release', y='total_sales', hue='platform')
plt.xlabel('Ano de Lançamento')
plt.savefig('vendas_totais_ano_todas_plataformas_recente.png')

#O ps3, xbox360 e 3DS começaram liderando em 2013, porém entraram em declinio rápido, sendo substituídos por PS4 e XOne que dominaram o mercado entre 2015 e 2016.

# Análise comparativa entre PS4 e XOne

df_recent_ps4 = df_recent[df_recent['platform'] == 'PS4']
df_recent_xone = df_recent[df_recent['platform'] == 'XOne'] 

df_recent_ps4_gb = df_recent_ps4.groupby('name')['total_sales'].sum().reset_index()
df_recent_xone_gb = df_recent_xone.groupby('name')['total_sales'].sum().reset_index()

#plt.figure(figsize=(12, 6))
#plt.title('Vendas Totais PS4 vs XOne (2013-2020)')
#plt.boxplot(data=[df_recent_ps4_gb, df_recent_xone_gb], labels=['PS4', 'XOne'])
#plt.ylabel('Vendas Totais (em milhões)')
#plt.savefig('vendas_totais_ps4_xone.png')