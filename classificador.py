# Importando bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# Mensagem para avisar que o modelo esta sendo treinado
print("Treinando o modelo, esse procedimento pode demorar!")

# Carregando seu dataset
seu_dataset = pd.read_csv("Electric_Vehicle_Population_Data.csv")

# Selecionando as características e as labels para classificação
X_classification = seu_dataset[['Model Year', 'Electric Range', 'Base MSRP']]
y_classification = seu_dataset['Clean Alternative Fuel Vehicle (CAFV) Eligibility']

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Treinando o modelo classificador com RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Salvando o modelo treinado de classificação
dump(grid_search, 'rf_classifier_model.joblib')

# Termino do treinamento do modelo
print("Modelo de clustering treinado e salvo com sucesso!")

# Avaliando o modelo
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo de classificação:", accuracy)

# Classificação de uma nova instância (simulada) e apresentação dos scores de probabilidade
new_instance_classification = [[2022, 250, 35000]]  # Preencha com valores da sua nova instância
predicted_class = grid_search.predict(new_instance_classification)
probability_scores = grid_search.predict_proba(new_instance_classification)
print("A nova instância pertence à classe:", predicted_class)
print("Scores de probabilidade:", probability_scores)
