import os
import shutil
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

class ProcessadorDeImagens:
    def __init__(self, caminho_csv, caminho_imagens, caminho_saida):
        self.dados = pd.read_csv(caminho_csv)
        self.caminho_imagens = caminho_imagens
        self.caminho_saida = caminho_saida
    
    def criar_diretorios(self):
        """
        Cria os subdiretórios para armazenar as sub-imagens de acordo com a classe da célula.
        """
        classes = self.dados['bethesda_system'].unique()
        for classe in classes:
            Path(os.path.join(self.caminho_saida, 'treino', classe)).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(self.caminho_saida, 'teste', classe)).mkdir(parents=True, exist_ok=True)
    
    def recortar_e_armazenar_subimagens(self):
        """
        Recorta as sub-imagens de 100x100 e armazena em sub-diretórios de acordo com a classe.
        O nome da imagem será o número da célula na planilha.
        """
        for _, linha in self.dados.iterrows():
            caminho_imagem = os.path.join(self.caminho_imagens, linha['image_filename'])
            imagem = cv2.imread(caminho_imagem)
            
            if imagem is not None:
                x, y = linha['nucleus_x'], linha['nucleus_y']
                
                # Garantir que as coordenadas estejam dentro dos limites da imagem
                altura, largura = imagem.shape[:2]
                x_inicio = max(0, x - 50)
                x_fim = min(largura, x + 50)
                y_inicio = max(0, y - 50)
                y_fim = min(altura, y + 50)
                
                sub_imagem = imagem[y_inicio:y_fim, x_inicio:x_fim]
                
                # Verificar se a sub-imagem tem o tamanho esperado (100x100)
                if sub_imagem.shape[0] == 100 and sub_imagem.shape[1] == 100:
                    caminho_saida = os.path.join(self.caminho_saida, linha['bethesda_system'], f"{linha['cell_id']}.png")
                    cv2.imwrite(caminho_saida, sub_imagem)
                else:
                    print(f"Sub-imagem de tamanho inesperado: {sub_imagem.shape} para a célula {linha['cell_id']}")
    
    def processar(self):
        """
        Executa o processamento completo das imagens.
        """
        #self.criar_diretorios()
        #self.recortar_e_armazenar_subimagens()
        self.organizar_dados_treino_teste()
    
    def organizar_dados_treino_teste(self):
        """
        Organiza as sub-imagens em conjuntos de treino e teste na proporção 80:20 de forma aleatória.
        """
        classes = self.dados['bethesda_system'].unique()
        for classe in classes:
            caminho_classe = os.path.join(self.caminho_saida, classe)
            if not os.path.exists(caminho_classe):
                print(f"Caminho {caminho_classe} não existe. Verifique se as sub-imagens foram criadas corretamente.")
                continue
            
            imagens = [f for f in os.listdir(caminho_classe) if os.path.isfile(os.path.join(caminho_classe, f))]
            
            if not imagens:
                print(f"Nenhuma imagem encontrada em {caminho_classe}.")
                continue
            
            # Embaralhar as imagens
            np.random.shuffle(imagens)
            
            # Separar as imagens em treino (80%) e teste (20%)
            tamanho_treino = int(0.8 * len(imagens))
            imagens_treino = imagens[:tamanho_treino]
            imagens_teste = imagens[tamanho_treino:]
            
            # Mover as imagens para as pastas apropriadas
            for imagem in imagens_treino:
                shutil.move(os.path.join(caminho_classe, imagem), os.path.join(self.caminho_saida, 'treino', classe, imagem))
                print("Treino:   "+self.caminho_saida)

            
            for imagem in imagens_teste:
                shutil.move(os.path.join(caminho_classe, imagem), os.path.join(self.caminho_saida, 'teste', classe, imagem))
                print("Teste:   "+self.caminho_saida)


# Exemplo de uso
caminho_csv = '/home/tiger/2024/PAI/repositorio/Processamento_Analise_Imagem_Papanicolau/arquivos_fundamentais/classifications.csv'  # Caminho para o arquivo CSV
caminho_imagens = '/home/tiger/2024/PAI/repositorio/Processamento_Analise_Imagem_Papanicolau/arquivos_fundamentais/imagens'  # Caminho para o diretório onde estão as imagens
caminho_saida = '/home/tiger/2024/PAI/repositorio/Processamento_Analise_Imagem_Papanicolau/arquivos_fundamentais/saida_imagens'  # Caminho para o diretório onde as sub-imagens serão salvas

processador = ProcessadorDeImagens(caminho_csv, caminho_imagens, caminho_saida)
processador.processar()
