###################
# Sávio Francisco #
# Matheus Tavares #
#                 #
# #################

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

def exploracao_dados(df):
    
    st.write("Análise Exploratória dos Dados")

    # Histograma para "Número de Médicos por 1000 Habitantes"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Número de médicos por 1000 habitantes'], kde=True, ax=ax, color='blue')
    ax.set_title('Número de Médicos por 1000 Habitantes')
    ax.set_xlabel('Número de Médicos por 1000 Habitantes')
    ax.set_ylabel('Frequência')
    plt.tight_layout()
    st.pyplot(fig)

    # Gráfico de Linhas para "Produto Interno Bruto"
    fig, ax = plt.subplots(figsize=(10, 6))

    df['Produto Interno Bruto'] = df['Produto Interno Bruto'].str.replace(',', '.').astype(float)
    df['Produto Interno Bruto'] = df['Produto Interno Bruto'] / 1_000_000

    # Plotando o gráfico de linhas para o PIB ao longo dos anos
    ax.plot(df['Ano'].astype(int), df['Produto Interno Bruto'], marker='o', color='green')

    ax.set_title('Produto Interno Bruto ao Longo dos Anos')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Produto Interno Bruto (Milhões de R$)')
    plt.tight_layout()
    st.pyplot(fig)

    # Histograma para "Gasto per Capita com Atividades de Saúde"
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df['Gasto per capita com atividades de saúde'] = df['Gasto per capita com atividades de saúde'].str.replace(',', '.').astype(float)
    ax.plot(df['Ano'],df['Gasto per capita com atividades de saúde'], marker='o', color='orange')
    ax.set_title('Gasto per Capita com Saúde')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Gasto per Capita com Saúde (R$)')
    plt.tight_layout()
    st.pyplot(fig)

    # Boxplot para "Produto Interno Bruto"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df['Produto Interno Bruto'], ax=ax, color='green')
    ax.set_title('Produto Interno Bruto')
    plt.tight_layout()
    st.pyplot(fig)

    # Boxplot para "Gasto per Capita com Atividades de Saúde"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df['Gasto per capita com atividades de saúde'], ax=ax, color='orange')
    ax.set_title('Gasto per Capita com Saúde')
    plt.tight_layout()
    st.pyplot(fig)

    # Boxplot para "Número de Médicos por 1000 Habitantes"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df['Número de médicos por 1000 habitantes'], ax=ax, color='blue')
    ax.set_title('Número de Médicos por 1000 Habitantes')
    plt.tight_layout()
    st.pyplot(fig)

def plot_cases_by_year(df_agrupado, city):
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_agrupado['Ano'], df_agrupado['Casos'], marker='o', label=f'Casos de Sífilis em {city}')
    plt.title(f'Casos de Sífilis por Ano - {city}')
    plt.xlabel('Ano')
    plt.ylabel('Número de Casos')
    plt.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(plt)

def plot_filtered_heatmap(correlation_matrix, target="Casos", top_n=10):
    """Plota um heatmap com as variáveis mais correlacionadas com o alvo"""
    
    # Selecionar as N variáveis mais fortemente correlacionadas com 'Casos'
    correlations = correlation_matrix[target].drop(target)  # Remover a correlação consigo mesmo
    top_vars = correlations.abs().nlargest(top_n).index  # Selecionar as N mais fortes

    # Criar submatriz de correlação apenas com essas variáveis
    filtered_matrix = correlation_matrix.loc[top_vars, top_vars]

    # Criar heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(filtered_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)

    # Ajustes visuais
    plt.title(f"Heatmap - Top {top_n} Variáveis Mais Correlacionadas com 'Casos'", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    # Exibir no Streamlit
    st.pyplot(fig)

def plot_filtered_correlation_network(correlation_matrix, target="Casos", top_n=8, threshold=0.7):
    """Plota um grafo de rede com apenas as N variáveis mais correlacionadas"""

    # Filtrar as correlações com a variável alvo
    correlations = correlation_matrix[target].drop(target)  # Remover a correlação consigo mesmo
    top_vars = correlations.abs().nlargest(top_n).index  # Selecionar as N mais fortes

    # Criar submatriz de correlação
    filtered_matrix = correlation_matrix.loc[top_vars, top_vars]

    # Criar o grafo
    G = nx.Graph()
    
    for col in filtered_matrix.columns:
        for row in filtered_matrix.index:
            correlation = filtered_matrix.loc[row, col]
            if abs(correlation) >= threshold and row != col:
                G.add_edge(row, col, weight=correlation)

    # Ajustar posições do grafo
    pos = nx.spring_layout(G, seed=42)

    # Tamanhos dos nós baseados no grau de conexões
    node_sizes = [G.degree(n) * 500 for n in G.nodes()]

    # Criar a figura
    fig, ax = plt.subplots(figsize=(10, 6))

    # Desenhar os nós
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.7, ax=ax)

    # Desenhar as conexões (arestas) com espessura baseada na força da correlação
    edges = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edges(G, pos, edgelist=edges.keys(), width=[abs(v) * 2 for v in edges.values()], edge_color="gray", ax=ax)

    # Adicionar rótulos aos nós
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    # Exibir o gráfico
    plt.title(f"Top {top_n} Variáveis Mais Correlacionadas com 'Casos'", fontsize=14)
    plt.axis("off")
    st.pyplot(fig)

def arima(dk):

    # Ordenar os dados por Ano
    df_temporal = dk[["Ano", "Casos"]].sort_values(by="Ano")

    # Definir o modelo ARIMA (parâmetros ajustáveis)
    modelo = ARIMA(df_temporal["Casos"], order=(5,1,0))  # (p,d,q) - ajustar esses valores
    modelo_fit = modelo.fit()

    # Fazer previsões para os anos futuros
    anos_futuros = list(range(2024, 2030))
    previsoes_futuras = modelo_fit.forecast(steps=len(anos_futuros))

    # Criar DataFrame com previsões
    df_previsoes_arima = pd.DataFrame({
        "Ano": anos_futuros,
        "Previsão de Casos": previsoes_futuras
    })

    st.write("Mostrar previsões")
    st.write(df_previsoes_arima)

    # Plotando a série temporal
    plt.figure(figsize=(10, 5))
    plt.plot(df_temporal["Ano"], df_temporal["Casos"], label="Casos Históricos", marker='o')
    plt.plot(df_previsoes_arima["Ano"], df_previsoes_arima["Previsão de Casos"], 
            label="Previsão ARIMA", linestyle="dashed", marker='o', color='red')

    # Definir os ticks do eixo X para serem anuais
    plt.xticks(ticks=range(df_temporal["Ano"].min(), df_previsoes_arima["Ano"].max() + 1, 1))

    # Personalizar o gráfico
    plt.xlabel("Ano")
    plt.ylabel("Casos de Sífilis")
    plt.title("Previsão de Casos usando ARIMA")
    plt.legend()
    plt.grid(True)

    # Exibir o gráfico
    st.pyplot(plt)

def codificacao_bases_merged(df_merged:pd.DataFrame):
    # Criar uma cópia do DataFrame
    dk = df_merged.copy()

    # Garantir que "Ano" seja numérico antes de qualquer transformação
    dk["Ano"] = dk["Ano"].astype(int)  # Mantém os anos corretos

    # Aplicar LabelEncoder apenas às colunas categóricas (excluindo "Ano")
    label_encoders = {}
    for col in dk.select_dtypes(include=['object', 'category']).columns:
        if col != "Ano":  
            le = LabelEncoder()
            
            # 🚨 Substituir valores NaN por "MISSING" antes de codificar 🚨
            dk[col] = dk[col].fillna("MISSING")  

            # Aplicar o LabelEncoder
            dk[col] = le.fit_transform(dk[col])

        # Armazena o encoder caso precise reverter depois
        label_encoders[col] = le
    
    #Correlação de Pearson
    st.write("Correlação de Pearson") 
    correlation_matrix = dk.corr()
    correlations = correlation_matrix["Casos"]
    strong_correlations = correlations[correlations.abs() > 0.5].sort_values(ascending=False)
    st.write(strong_correlations)

    plot_filtered_heatmap(correlation_matrix, top_n=10)
    
    #Correlação de Spearman
    st.write("Correlação de Spearman")
    correlation_matrix = dk.corr(method="spearman")
    correlations = correlation_matrix["Casos"]
    strong_correlations = correlations[correlations.abs() > 0.5].sort_values(ascending=False)
    st.write(strong_correlations)

    plot_filtered_heatmap(correlation_matrix, top_n=10)

    return dk

def merge_bases(df_agrupado, df:pd.DataFrame)->pd.DataFrame:
    df_agrupado['Ano'] = df_agrupado['Ano'].astype(str)
    df['Ano'] = df['Ano'].astype(str)

    df_merged = df_agrupado.merge(df, left_on="Ano", right_on="Ano", how="left")

    return df_merged

def base_siflis(cidade):

    data = pd.read_csv("BH_Juiz_Ipatinga/babf2604-06a5-42b9-8038-0ea1d54051fb.csv")
    dbh = data[data['ID_MUNICIP'] == cidade]
    dbh = dbh[dbh["NU_ANO"] != 2024]
    df_agrupado = dbh['NU_ANO'].value_counts().reset_index()
    df_agrupado.columns = ['Ano', 'Casos']
    
    return df_agrupado

# Função para carregar os dados de cada cidade
def load_data(city):
    # Alterar os caminhos conforme a cidade
    if city == "Belo Horizonte":
        
        df = pd.read_csv("BH_Juiz_Ipatinga/Dados_BeloHorizonte.csv",encoding="latin1", sep=";")
        df = df[df["Ano"] >= 2010]
        df.isnull().sum()
        df = df.drop(columns=df.columns[df.isnull().sum() > 3])
        
        #Número de casos de síflis
        plot_cases_by_year(base_siflis("BELO HORIZONTE"), city)

        #Exploração de Dados
        exploracao_dados(df)
        
        #Merge das bases
        df_merged = merge_bases(base_siflis("BELO HORIZONTE"), df)
        dk = codificacao_bases_merged(df_merged)

        arima(dk)
        
    elif city == "Ipatinga":
         
        df = pd.read_csv("BH_Juiz_Ipatinga/Dados_Ipatinga.csv",encoding="latin1", sep=";")
        df = df[df["Ano"] >= 2010]
        df.isnull().sum()
        df = df.drop(columns=df.columns[df.isnull().sum() > 3])
        
        #Número de casos de síflis
        plot_cases_by_year(base_siflis("IPATINGA"), city)
        
        #Exploração de Dados
        exploracao_dados(df)
        
        #Merge das bases
        df_merged = merge_bases(base_siflis("IPATINGA"), df)
        dk = codificacao_bases_merged(df_merged)

        arima(dk)
    
    elif city == "Juiz de Fora":

        df = pd.read_csv("BH_Juiz_Ipatinga/Dados_Ipatinga.csv",encoding="latin1", sep=";")
        df = df[df["Ano"] >= 2010]
        df.isnull().sum()
        df = df.drop(columns=df.columns[df.isnull().sum() > 3])
        
        #Número de casos de síflis
        plot_cases_by_year(base_siflis("JUIZ DE FORA"), city)
        
        #Exploração de Dados
        exploracao_dados(df)

        #Merge das bases
        df_merged = merge_bases(base_siflis("JUIZ DE FORA"), df)
        dk = codificacao_bases_merged(df_merged)

        arima(dk)
    
    else:
        return "Cidade não encontrada"

# Função para processar os dados
def process_data():

    city = st.selectbox("Escolha a cidade", ["Belo Horizonte", "Ipatinga", "Juiz de Fora"])
    load_data(city)
    

if __name__ == "__main__":
   
    st.title("Análise de Casos de Sífilis nas Cidades")
    st.write("""
    Este aplicativo permite visualizar e fazer previsões de casos de sífilis em diversas cidades. Escolha uma cidade e veja os dados de casos históricos e previsões para os próximos anos.
    """)

    process_data()