import os
import numpy as np
import cv2
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__)
print("Seaborn version:", sns.__version__)
print("Matplotlib version:", matplotlib.__version__)

def carregar_imagens(caminho, classes):
    print("Entrou na funcao carregar_imagem")
    X = []
    y = []
    
    for classe in classes:
        caminho_classe = os.path.join(caminho, classe)
        for arquivo in os.listdir(caminho_classe):
            if arquivo.endswith('.png'):
                caminho_arquivo = os.path.join(caminho_classe, arquivo)
                imagem = cv2.imread(caminho_arquivo, cv2.IMREAD_GRAYSCALE)
                imagem = cv2.resize(imagem, (100, 100))  # Redimensionar para garantir tamanho consistente
                X.append(imagem.flatten())  # Achatar a imagem para um vetor
                y.append(classe)
    
    return np.array(X), np.array(y)

def plotar_matriz_confusao(y_true, y_pred, classes, title):
    print("Entrou na funcao plotar_matriz_confusao")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()
    print("Terminou de plotar matriz de confusao")

# Caminhos para os dados de treino e teste
caminho_treino = r'C:\Users\joaog\OneDrive\Documentos\GitHub\Processamento_Analise_Imagem_Papanicolau\arquivos_fundamentais\saida_imagens\treino'
caminho_teste = r'C:\Users\joaog\OneDrive\Documentos\GitHub\Processamento_Analise_Imagem_Papanicolau\arquivos_fundamentais\saida_imagens\teste'

# Lista de classes
classes = ['ASC-H', 'ASC-US', 'HSIL', 'LSIL', 'Negative for intraepithelial lesion', 'SCC']

# Carregar dados de treino
X_train, y_train = carregar_imagens(caminho_treino, classes)
print(1)

# Carregar dados de teste
X_test, y_test = carregar_imagens(caminho_teste, classes)
print(2)

# Treinar SVM para classificação multiclasse
svm_multi = SVC(kernel='linear')
svm_multi.fit(X_train, y_train)
print("Iniciando previsoes e avaliacao multiclasse")

# Previsões e avaliação multiclasse
y_pred_multi = svm_multi.predict(X_test)
accuracy_multi = accuracy_score(y_test, y_pred_multi)

print("Acurácia (Classificação Multiclasse):", accuracy_multi)
print("Relatório de Classificação (Classificação Multiclasse):")
print(classification_report(y_test, y_pred_multi))

# Plotar matriz de confusão multiclasse
plotar_matriz_confusao(y_test, y_pred_multi, classes, 'Matriz de Confusão (Classificação Multiclasse)')

