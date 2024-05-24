# Importando bibliotecas
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Mensagem para avisar que o modelo esta sendo treinado
print("Treinando o modelo, esse procedimento pode demorar!")

# Carregando seu dataset
seu_dataset = pd.read_csv("Electric_Vehicle_Population_Data.csv")

# Selecionando as características para clustering
X_clustering = seu_dataset[['Model Year', 'Electric Range', 'Base MSRP']]

# Pré-processamento de dados para clustering
scaler = StandardScaler()
X_clustering_scaled = scaler.fit_transform(X_clustering)

# Determinando o número ideal de clusters
kmeans = KMeans(n_clusters=3, random_state=42)  # Você pode ajustar o número de clusters conforme necessário
kmeans.fit(X_clustering_scaled)

# Termino do treinamento do modelo
print("Modelo de clustering treinado e salvo com sucesso!")

# Salvando o modelo treinado de clustering
dump(kmeans, 'kmeans_model.joblib')

# Classificação de uma nova instância (simulada)
new_instance_clustering = [[2022, 250, 35000]]  # Preencha com valores da sua nova instância
new_instance_clustering_scaled = scaler.transform(new_instance_clustering)
predicted_cluster = kmeans.predict(new_instance_clustering_scaled)
print("A nova instância pertence ao cluster:", predicted_cluster)
