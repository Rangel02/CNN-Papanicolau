import os
import numpy as np
import pandas as pd
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

# Verificar GPUs disponíveis

def preparar_dados(caminho_treino, caminho_teste, img_size=(224, 224), batch_size=32):
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
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        caminho_teste,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, test_generator

def plotar_matriz_confusao(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def criar_modelo_efficientnet_multiclasse(img_size=(224, 224, 3), num_classes=6, learning_rate=0.0001, train_base_layers=False):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=img_size, pooling='max')
    x = base_model.output
    x = Dense(128, activation='relu')(x)  # Saída multiclasse
    x = Dense(num_classes, activation='softmax')(x)  # Saída multiclasse

    model = Model(inputs=base_model.input, outputs=x)
    
    if not train_base_layers:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        # Descongele algumas camadas do modelo base
        for layer in base_model.layers[-20:]:  # Ajuste conforme necessário
            layer.trainable = True
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Caminhos para os dados de treino e teste
caminho_treino = './arquivos_fundamentais/saida_imagens/treino'
caminho_teste = './arquivos_fundamentais/saida_imagens/teste'

# Preparar os dados
train_generator, test_generator = preparar_dados(caminho_treino, caminho_teste)

# Criar o modelo multiclasse
num_classes = len(train_generator.class_indices)
modelo_multiclasse = criar_modelo_efficientnet_multiclasse(num_classes=num_classes, train_base_layers=True)

# Callbacks
checkpoint_multiclasse = ModelCheckpoint('modelo_multiclasse.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop_multiclasse = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
reduce_lr_multiclasse = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, mode='min')

# Treinamento
historico_multiclasse = modelo_multiclasse.fit(
    train_generator,
    epochs=50,  # Aumentar o número de épocas
    validation_data=test_generator,
    callbacks=[checkpoint_multiclasse, early_stop_multiclasse, reduce_lr_multiclasse]
)

# Avaliação
y_pred_multi = np.argmax(modelo_multiclasse.predict(test_generator), axis=-1)
y_true_multi = test_generator.classes

print("Acurácia (Classificação Multiclasse):", accuracy_score(y_true_multi, y_pred_multi))
print("Relatório de Classificação (Classificação Multiclasse):")
print(classification_report(y_true_multi, y_pred_multi, target_names=list(train_generator.class_indices.keys())))

# Plotar matriz de confusão multiclasse
plotar_matriz_confusao(y_true_multi, y_pred_multi, list(train_generator.class_indices.keys()), 'Matriz de Confusão (Classificação Multiclasse)')

# Plotar gráficos de aprendizado
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historico_multiclasse.history['accuracy'], label='Acurácia de Treino')
plt.plot(historico_multiclasse.history['val_accuracy'], label='Acurácia de Teste')
plt.legend()
plt.title('Acurácia')

plt.subplot(1, 2, 2)
plt.plot(historico_multiclasse.history['loss'], label='Loss de Treino')
plt.plot(historico_multiclasse.history['val_loss'], label='Loss de Teste')
plt.legend()
plt.title('Loss')
plt.show()
