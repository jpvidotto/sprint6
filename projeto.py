import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

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
#print("Top 3 plataformas:", top3_platforms)

top10_platforms = df_top_platforms_list.head(10).index.tolist()
print("Top 10 plataformas:", top10_platforms)

print()

# Filtrar dados apenas para essas plataformas
df_top = df[df['platform'].isin(top3_platforms)]

# Criar tabela de vendas por ano e plataforma
vendas_ano_plataforma = df_top.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index().sort_values(by='year_of_release')
print(vendas_ano_plataforma)

print()

plt.figure(figsize=(12, 6))
plt.title('Vendas Totais por Ano para as 3 Principais Plataformas')
sns.scatterplot(data=vendas_ano_plataforma, x='year_of_release', y='total_sales', hue='platform')
plt.xlabel('Ano de Lançamento')
plt.savefig('vendas_totais_ano.png')

#Análise de quantidade de vendas por plataforma ao longo dos anos:
df_platforms_1995 = df[df['year_of_release'] > 1995 ].sort_values('year_of_release')
df_platforms_1995 = df_platforms_1995.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index()
df_platforms_1995 = df_platforms_1995[df_platforms_1995['total_sales'] > 20]

plt.figure(figsize=(12, 6))
plt.title('Vendas Totais por Ano para todas as Plataformas')
sns.lineplot(data=df_platforms_1995, x='year_of_release', y='total_sales', hue='platform')
plt.xlabel('Ano de Lançamento')
plt.savefig('vendas_totais_ano_todas_plataformas.png')

#Antes de 2010 as plataformas tinham uma vida média de 7 anos, após 2010 essa média caiu para 5 anos. Isso pode ser explicado pelo avanço tecnológico e a rápida evolução do mercado de jogos, onde novas plataformas são lançadas com mais frequência para atender às demandas dos consumidores por melhores gráficos, desempenho e funcionalidades. Além disso, a crescente popularidade dos jogos móveis e serviços de streaming de jogos pode ter contribuído para a redução da vida útil das plataformas tradicionais.

#Irei pegar dados de 2013 para frente, visto que é um período mais recente e relevante para análise de mercado atual.
df_recent = df[df['year_of_release'] >= 2013].sort_values('year_of_release')
df_recent = df_recent.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.title('Vendas Totais por Ano para todas as Plataformas (2013-2016)')
sns.lineplot(data=df_recent, x='year_of_release', y='total_sales', hue='platform')
plt.xlabel('Ano de Lançamento')
plt.savefig('vendas_totais_ano_todas_plataformas_recente.png')

#O ps3, xbox360 e 3DS começaram liderando em 2013, porém entraram em declinio rápido, sendo substituídos por PS4 e XOne que dominaram o mercado entre 2015 e 2016.

# Análise comparativa entre PS4 e XOne

df_recent_ps4 = df_recent[df_recent['platform'] == 'PS4']
df_recent_xone = df_recent[df_recent['platform'] == 'XOne'] 

df_recent_ps4_xone = pd.concat([df_recent_ps4, df_recent_xone])

plt.figure(figsize=(12, 6))
plt.title('Vendas Totais PS4 vs XOne (2013-2016)')
sns.boxplot(data = df_recent_ps4_xone, x='platform', y='total_sales')
plt.ylabel('Vendas Totais (em milhões)')
plt.savefig('vendas_totais_ps4_xone.png')

#Conclusão: O PS4 teve vendas totais significativamente maiores do que o XOne no período de 2013 a 2016, indicando uma preferência clara dos consumidores pela plataforma da Sony em comparação com a da Microsoft.

df_nan_count = df[df['critic_score'].isna()].isna().sum()
#Fiz uma contagem para saber se há muitos valores nulos na coluna critic_score e user_score ao mesmo tempo, para assim ver se seria possível fazer uma análise de correlação entre essas duas colunas.

df_wo_nan_ps4 = df.dropna(subset=['critic_score', 'user_score'])
df_wo_nan_ps4 = df_wo_nan_ps4[df_wo_nan_ps4['platform'] == 'PS4']

#Criei um novo dataframe sem os valores nulos para fazer a análise da correlação de vendas entre as notas dos críticos e dos usuários no PS4.

plt.figure(figsize=(12, 6))
plt.title('Correlação entre Notas dos Críticos e Usuários vs Vendas Totais no PS4')
sns.scatterplot(data=df_wo_nan_ps4, x='critic_score', y='user_score', size='total_sales', sizes=(20, 200), alpha=0.6)
plt.xlabel('Nota dos Críticos')
plt.ylabel('Nota dos Usuários')
plt.savefig('correlacao_notas_vendas_ps4.png')

#Os jogos com maiores nota dos críticos costumam ter maiores numero em vendas, já os jogos com maiores notas dos usuários não necessariamente possuem maiores vendas. Portanto, a nota dos críticos parece ter uma correlação mais forte com as vendas totais no PS4 em comparação com a nota dos usuários.


df_wo_nan_ps4 = df_wo_nan_ps4.sort_values('critic_score', ascending=False)

df_wo_nan_ps4_top10 = df_wo_nan_ps4[['name', 'critic_score', 'user_score', 'total_sales']].head(10).reset_index(drop=True)
#Listei os 10 jogos com maiores notas dos críticos no PS4, juntamente com suas notas dos usuários e vendas totais.]

df_wo_nan_xone = df.dropna(subset=['critic_score', 'user_score'])
df_wo_nan_xone = df_wo_nan_xone[df_wo_nan_xone['platform'] == 'XOne']
df_wo_nan_xone = df_wo_nan_xone[df_wo_nan_xone['name'].isin(df_wo_nan_ps4_top10['name'])]

df_wo_nan_ps4_compare = df_wo_nan_ps4[df_wo_nan_ps4['name'].isin(df_wo_nan_ps4_top10['name'])]

df_wo_nan_xone_ps4_concat = pd.concat([df_wo_nan_ps4_compare, df_wo_nan_xone])

plt.figure(figsize=(12, 6))
plt.title('Comparacao entre vendas totais no PS4 e XOne para jogos com notas dos críticos e usuários')
sns.barplot(data=df_wo_nan_xone_ps4_concat, x='critic_score', y='total_sales', hue='platform')
plt.xlabel('Nota dos Críticos')
plt.ylabel('Vendas Totais (em milhões)')
plt.savefig('comparacao_vendas_ps4_xone.png')

#Conclusão sobre a comparação entre PS4 e XOne para jogos com altas notas dos críticos:
#Dos 10 jogos com maiores notas dos críticos no PS4, apenas 6 foram lançados no XOne. Desses 6 jogos, todos tiveram vendas totais significativamente menores no XOne em comparação com o PS4. Isso reforça a ideia de que o PS4 teve um desempenho de vendas superior ao XOne, mesmo quando analisamos jogos com alta qualidade reconhecida pelos críticos.

df_genre_sales = df.groupby('genre')['total_sales'].sum().reset_index().sort_values(by='total_sales', ascending=False)
print(df_genre_sales)

print()

#conclusão sobre as vendas por gênero:
#O gênero Action lidera as vendas totais, seguido por Shooter e Sports. Esses gêneros parecem ser os mais populares entre os consumidores, contribuindo significativamente para as vendas totais de jogos. Por outro lado, gêneros como Strategy e Puzzle apresentam vendas totais mais baixas, indicando uma preferência menor por esses tipos de jogos no mercado. Essa informação pode ser útil para desenvolvedores e publishers ao decidirem quais tipos de jogos investir e desenvolver no futuro. E também mostra uma preferencia por generos onde a jogabilidade é mais dinâmica e envolvente.

# Análise de vendas por plataforma em diferentes regiões

#Filtragem dos dados por plataforma e soma das vendas em cada região
df_na_sales = df.groupby('platform')['na_sales'].sum().reset_index().sort_values(by='na_sales', ascending=False)
df_eu_sales = df.groupby('platform')['eu_sales'].sum().reset_index().sort_values(by='eu_sales', ascending=False)
df_jp_sales = df.groupby('platform')['jp_sales'].sum().reset_index().sort_values(by='jp_sales', ascending=False)

df_top5_na = df_na_sales.head(5)
df_top5_eu = df_eu_sales.head(5)
df_top5_jp = df_jp_sales.head(5)
print("Top 5 plataformas na América do Norte:\n", df_top5_na)
print()
print("Top 5 plataformas na Europa:\n", df_top5_eu)
print()
print("Top 5 plataformas no Japão:\n", df_top5_jp)
print()

#Conclusão sobre as vendas por plataforma regional:
#Na América do Norte e Europa, as plataformas com maiores vendas são PS4, XOne e PS3, indicando uma preferência por consoles da Sony e Microsoft nessas regiões. No Japão, o cenário é diferente, com o 3DS liderando as vendas, seguido pelo PS4 e PS3. Isso sugere que os consumidores japoneses têm uma preferência maior por consoles portáteis, como o 3DS, em comparação com outras regiões. Essas diferenças regionais nas preferências de plataforma podem ser influenciadas por fatores culturais, econômicos e de mercado específicos de cada região.

# Análise de vendas por gênero em diferentes regiões

#Filtragem dos dados por gênero e soma das vendas em cada região
df_na_sales_by_genre = df.groupby('genre')['na_sales'].sum().reset_index().sort_values(by='na_sales', ascending=False)
df_eu_sales_by_genre = df.groupby('genre')['eu_sales'].sum().reset_index().sort_values(by='eu_sales', ascending=False)
df_jp_sales_by_genre = df.groupby('genre')['jp_sales'].sum().reset_index().sort_values(by='jp_sales', ascending=False)

df_top5_na_genre = df_na_sales_by_genre.head(5)
df_top5_eu_genre = df_eu_sales_by_genre.head(5)
df_top5_jp_genre = df_jp_sales_by_genre.head(5)
print("Top 5 gêneros na América do Norte:\n", df_top5_na_genre)
print()
print("Top 5 gêneros na Europa:\n", df_top5_eu_genre)
print()
print("Top 5 gêneros no Japão:\n", df_top5_jp_genre)
print()

#Conclusão sobre as vendas por gênero regional:
#Na América do Norte e Europa, os gêneros com maiores vendas são Action, Shooter e Sports, indicando uma preferência por jogos dinâmicos e competitivos nessas regiões. No Japão, os gêneros mais populares são Role-Playing, Action e Fighting, sugerindo uma preferência por jogos com narrativas envolventes e elementos de combate. Essas diferenças regionais nas preferências de gênero podem ser influenciadas por fatores culturais e de mercado específicos de cada região, visto que no Japão há uma tradição maior em jogos de RPG e luta.

# Análise de vendas por classificação etária em diferentes regiões


df_rating_sales_wo_nan = df[df['rating'] != 'Unknown']
#Filtragem dos dados por classificação etária e soma das vendas em cada região
df_rating_nasales = df_rating_sales_wo_nan.groupby('rating')['na_sales'].sum().reset_index().sort_values(by='na_sales', ascending=False)
df_rating_eusales = df_rating_sales_wo_nan.groupby('rating')['eu_sales'].sum().reset_index().sort_values(by='eu_sales', ascending=False)
df_rating_jpsales = df_rating_sales_wo_nan.groupby('rating')['jp_sales'].sum().reset_index().sort_values(by='jp_sales', ascending=False)

df_top5rating_nasales = df_rating_nasales.head(5)
df_top5rating_eusales = df_rating_eusales.head(5)
df_top5rating_jpsales = df_rating_jpsales.head(5)
print("Top 5 ratings na América do Norte:\n", df_top5rating_nasales)
print()
print("Top 5 ratings na Europa:\n", df_top5rating_eusales)
print()
print("Top 5 ratings no Japão:\n", df_top5rating_jpsales)
print()

#Conclusão sobre as classificações etárias:
#Nos três mercados analisados, a classificação "M" (Mature) lidera as vendas, indicando uma preferência por jogos destinados a um público mais velho. Em seguida, as classificações "E" (Everyone) e "T" (Teen) também apresentam vendas significativas, sugerindo que jogos adequados para todas as idades e adolescentes também são populares. A classificação "E10+" (Everyone 10 and older) tem vendas menores em comparação com as outras classificações, mas ainda assim é relevante. Essas tendências refletem as preferências dos consumidores em diferentes faixas etárias e podem influenciar as decisões de desenvolvimento e marketing dos jogos.


# Análise estatística das notas dos usuários entre plataformas e gêneros
mean_user_score = df.groupby('platform')['user_score'].mean().reset_index().sort_values(by='user_score', ascending=False)

xbox_mean_user_score = mean_user_score[mean_user_score['platform'] == 'XOne']['user_score'].dropna().values[0]
pc_mean_user_score = mean_user_score[mean_user_score['platform'] == 'PC']['user_score'].dropna().values[0]
print(f"Média da nota dos usuários no XOne: {xbox_mean_user_score}")
print(f"Média da nota dos usuários no PC: {pc_mean_user_score}")

print()

mean_user_score_genres = df.groupby('genre')['user_score'].mean().reset_index().sort_values(by='user_score', ascending=False)

action_mean_user_score = mean_user_score_genres[mean_user_score_genres['genre'] == 'Action']['user_score'].dropna().values[0]
sports_mean_user_score = mean_user_score_genres[mean_user_score_genres['genre'] == 'Sports']['user_score'].dropna().values[0]
print(f"Média da nota dos usuários no gênero Action: {action_mean_user_score}")
print(f"Média da nota dos usuários no gênero Sports: {sports_mean_user_score}")

print()

#Conclusão sobre as médias das notas dos usuários:
#A média da nota dos usuários no XOne é ligeiramente maior do que no PC, sugerindo que os jogadores do XOne tendem a avaliar os jogos de forma um pouco mais positiva em comparação com os jogadores de PC. No entanto, a diferença não é muito grande, indicando que ambos os grupos têm opiniões relativamente semelhantes sobre os jogos. Já em relação aos gêneros, a média da nota dos usuários para jogos de Action é significativamente maior do que para jogos de Sports. Isso sugere que os jogos de Action são avaliados de forma mais positiva em comparação com os jogos de Sports, possivelmente devido à natureza mais dinâmica e envolvente dos jogos de Action.

#definir limiar alfa

alfa = 0.05

#utilizei o valor de alfa em 0.05, que é um valor comumente utilizado em testes estatísticos para determinar o nível de significância. Isso significa que estamos dispostos a aceitar uma probabilidade de 5% de cometer um erro tipo I, ou seja, rejeitar a hipótese nula quando ela é verdadeira. Esse valor é amplamente aceito na comunidade científica e estatística como um padrão para avaliar a significância dos resultados dos testes.

# Realizar o teste t de Student para comparar as médias das notas dos usuários no XOne e no PC

xbox_user_score_sales = df[df['platform'] == 'XOne']['user_score'].dropna()
pc_user_score_sales = df[df['platform'] == 'PC']['user_score'].dropna()

action_user_score_sales = df[df['genre'] == 'Action']['user_score'].dropna()
sports_user_score_sales = df[df['genre'] == 'Sports']['user_score'].dropna()

t_stat, p_value = st.ttest_ind(a=xbox_user_score_sales, b=pc_user_score_sales, equal_var=True)
print(f"Estatística t: {t_stat:.4f}")
print(f"Valor p: {p_value:.4f}")
if p_value < alfa:
    print("Rejeitamos a hipótese nula: Há uma diferença significativa entre as médias das notas dos usuários no XOne e no PC.")
else:
    print("Falhamos em rejeitar a hipótese nula: Não há uma diferença significativa entre as médias das notas dos usuários no XOne e no PC.")

print()

t_stat_genres, p_value_genres = st.ttest_ind(a=action_user_score_sales, b=sports_user_score_sales, equal_var=True)
print(f"Estatística t (gêneros): {t_stat_genres:.4f}")
print(f"Valor p (gêneros): {p_value_genres:.4f}")
if p_value_genres < alfa:
    print("Rejeitamos a hipótese nula: Há uma diferença significativa entre as médias das notas dos usuários nos gêneros Action e Sports.")
else:
    print("Falhamos em rejeitar a hipótese nula: Não há uma diferença significativa entre as médias das notas dos usuários nos gêneros Action e Sports.")

#Conclusão sobre o projeto geral e possível investimento para o marketing da empresa:
#Ao longo deste projeto, exploramos diversos aspectos do mercado de jogos, desde o lançamento de jogos ao longo dos anos até a análise de vendas por plataforma, gênero e região. Identificamos tendências importantes, como a preferência por certas plataformas em diferentes regiões e a correlação entre as notas dos críticos e as vendas totais. Além disso, realizamos testes estatísticos para comparar as médias das notas dos usuários em diferentes plataformas e gêneros, fornecendo insights valiosos sobre as preferências dos consumidores. Essas análises podem ajudar na tomada de decisão sobre em qual área o marketing da empresa deve focar para maximizar suas vendas e atender melhor ao público-alvo no ano de 2017. Quando vemos os testes estatísticos, percebemos que não há diferença significativa entre as médias das notas dos usuários no XOne e no PC, sugerindo que ambos os grupos têm opiniões semelhantes sobre os jogos. No entanto, há uma diferença significativa entre as médias das notas dos usuários nos gêneros Action e Sports, indicando que os jogadores avaliam esses gêneros de forma diferente. Esses insights podem ser úteis para direcionar os investimentos do marketing da loja ICE.
