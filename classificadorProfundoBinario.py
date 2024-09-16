import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Configurar crescimento de memória da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Função para preparar os dados
def preparar_dados(caminho_treino, caminho_teste, img_size=(224, 224), batch_size=64):
    train_datagen = ImageDataGenerator(rotation_range=20, 
                                       width_shift_range=0.2, 
                                       height_shift_range=0.2, 
                                       shear_range=0.2, 
                                       zoom_range=0.2, 
                                       horizontal_flip=True, 
                                       fill_mode='nearest')
    
    test_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow_from_directory(
        caminho_treino,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    test_generator = test_datagen.flow_from_directory(
        caminho_teste,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, test_generator

# Função para plotar a matriz de confusão
def plotar_matriz_confusao(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Função para criar o modelo binário
def criar_modelo_efficientnet_binario(img_size=(224, 224, 3), learning_rate=0.0001, train_base_layers=False):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=img_size, pooling='max')
    x = base_model.output
    x = Dense(128, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)  # Saída binária

    model = Model(inputs=base_model.input, outputs=x)
    
    if not train_base_layers:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in base_model.layers[-20:]:
            layer.trainable = True
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Caminhos para os dados de treino e teste
caminho_treino = r'C:\Users\joaog\OneDrive\Documentos\GitHub\Processamento_Analise_Imagem_Papanicolau\arquivos_fundamentais\saida_imagens\treino'
caminho_teste = r'C:\Users\joaog\OneDrive\Documentos\GitHub\Processamento_Analise_Imagem_Papanicolau\arquivos_fundamentais\saida_imagens\teste'

# Preparar os dados
train_generator, test_generator = preparar_dados(caminho_treino, caminho_teste)

# Criar o modelo binário
modelo_binario = criar_modelo_efficientnet_binario(train_base_layers=True)

# Callbacks
checkpoint_binario = ModelCheckpoint('modelo_binario.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop_binario = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
reduce_lr_binario = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, mode='min')

# Treinamento
historico_binario = modelo_binario.fit(
    train_generator,
    epochs=50,  # Aumentar o número de épocas conforme necessário
    validation_data=test_generator,
    callbacks=[checkpoint_binario, early_stop_binario, reduce_lr_binario]
)

# Avaliação
y_pred_bin = (modelo_binario.predict(test_generator) > 0.5).astype("int32").flatten()
y_true_bin = test_generator.classes

# Verificar as classes presentes nos dados
num_classes = len(np.unique(y_true_bin))

if num_classes == 2:
    target_names = ['Negative', 'Others']
else:
    # Ajuste conforme necessário para seu conjunto de dados
    target_names = [str(i) for i in range(num_classes)]

print("Acurácia (Classificação Binária):", accuracy_score(y_true_bin, y_pred_bin))
print("Relatório de Classificação (Classificação Binária):")
print(classification_report(y_true_bin, y_pred_bin, target_names=target_names, labels=range(num_classes)))

# Plotar matriz de confusão binária
plotar_matriz_confusao(y_true_bin, y_pred_bin, target_names, 'Matriz de Confusão (Classificação Binária)')

# Plotar gráficos de aprendizado
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historico_binario.history['accuracy'], label='Acurácia de Treino')
plt.plot(historico_binario.history['val_accuracy'], label='Acurácia de Teste')
plt.legend()
plt.title('Acurácia')

plt.subplot(1, 2, 2)
plt.plot(historico_binario.history['loss'], label='Loss de Treino')
plt.plot(historico_binario.history['val_loss'], label='Loss de Teste')
plt.legend()
plt.title('Loss')
plt.show()
