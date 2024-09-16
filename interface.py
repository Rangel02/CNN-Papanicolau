import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QLabel, QVBoxLayout, QWidget, QMenu, QMenuBar, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Image Processing App'
        self.image_path = None
        self.image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(500, 500)  # Define o tamanho inicial da janela para 500x500 pixels
        
        # Menus
        menubar = self.menuBar()
        
        fileMenu = menubar.addMenu('File')
        processMenu = menubar.addMenu('Process')
        analyzeMenu = menubar.addMenu('Analyze')
        
        # File Menu
        openFile = QAction('Open', self)
        openFile.triggered.connect(self.showDialog)
        fileMenu.addAction(openFile)
        
        # Process Menu
        histogramHSV = QAction('Show Image', self)
        histogramHSV.triggered.connect(self.show_image)
        processMenu.addAction(histogramHSV)

        convertGray = QAction('Convert to Grayscale', self)
        convertGray.triggered.connect(self.convert_to_grayscale)
        processMenu.addAction(convertGray)
        
        histogramGray = QAction('Histogram (Grayscale)', self)
        histogramGray.triggered.connect(self.show_histogram_gray)
        processMenu.addAction(histogramGray)
        
        histogramHSV = QAction('Histogram (HSV)', self)
        histogramHSV.triggered.connect(self.show_histogram_hsv)
        processMenu.addAction(histogramHSV)
        
        # Analyze Menu
        haralick = QAction('Haralick Descriptors', self)
        haralick.triggered.connect(self.show_haralick)
        analyzeMenu.addAction(haralick)
        
        huMoments = QAction('Hu Moments', self)
        huMoments.triggered.connect(self.show_hu_moments)
        analyzeMenu.addAction(huMoments)

        classifySubImage = QAction('Classify Sub-Image', self)
        classifySubImage.triggered.connect(self.classify_sub_image)
        analyzeMenu.addAction(classifySubImage)
        
        # Main Label for Image Display
        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        
        self.show()

    def showDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)", options=options)
        if fileName:
            self.image_path = fileName
            self.image = cv2.imread(fileName)
            self.display_image(self.image)

    def display_image(self, img):
        # Redimensionar a imagem para 500x500 pixels
        img_resized = cv2.resize(img, (500, 500))

        qformat = QImage.Format_Indexed8
        if len(img_resized.shape) == 3:  # rows[0], cols[1], channels[2]
            if img_resized.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img_resized, img_resized.shape[1], img_resized.shape[0], img_resized.strides[0], qformat)
        img = img.rgbSwapped()

        self.label.setPixmap(QPixmap.fromImage(img))
        self.resize(img.width(), img.height())

    def show_message(self, title, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

    def check_image_loaded(self):
        if self.image_path is None:
            self.show_message("Error", "No image loaded. Please load an image first.")
            return False
        return True

    def convert_to_grayscale(self):
        if not self.check_image_loaded():
            return
        
        imagem = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        plt.figure(figsize=(10, 10))
        plt.imshow(imagem, cmap='gray')
        plt.axis('off')
        plt.show()

    def show_histogram_gray(self):
        if not self.check_image_loaded():
            return
        
        imagem = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        niveis = 16
        imagem_quantizada = (imagem // (256 // niveis)).astype(np.uint8)
        
        histograma, _ = np.histogram(imagem_quantizada, bins=niveis, range=(0, niveis))
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(niveis), histograma, width=1, edgecolor='black')
        plt.title('Histograma de Tons de Cinza (16 Tons)')
        plt.xlabel('Intensidade')
        plt.ylabel('Frequência')
        plt.show()

    def show_image(self):
        if not self.check_image_loaded():
            return
        imagem = cv2.imread(self.image_path)
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)  # Converter de BGR para RGB
        plt.figure(figsize=(10, 10))
        plt.imshow(imagem_rgb)  # Usar a imagem RGB para exibição
        plt.axis('off')
        plt.show()


    def show_histogram_hsv(self):
        if not self.check_image_loaded():
            return
        
        imagem = cv2.imread(self.image_path)
        imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        
        H, S, V = cv2.split(imagem_hsv)
        H_quantizado = (H // 16) * 16
        V_quantizado = (V // 32) * 32
        
        histograma_2d, _, _ = np.histogram2d(H_quantizado.ravel(), V_quantizado.ravel(), bins=[16, 8], range=[[0, 256], [0, 256]])
        
        plt.figure(figsize=(10, 5))
        plt.imshow(histograma_2d, interpolation='nearest', cmap='hot')
        plt.title('Histograma 2D de Cor (H quantizado em 16, V quantizado em 8)')
        plt.xlabel('H (16 bins)')
        plt.ylabel('V (8 bins)')
        plt.colorbar()
        plt.show()

    def calculate_haralick(self):
        if not self.check_image_loaded():
            return
        
        imagem = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        niveis = 16
        imagem_quantizada = (imagem // (256 // niveis)).astype(np.uint8)
        
        distancias = [1, 2, 4, 8, 16, 32]
        angulos = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        descritores = []
        
        for distancia in distancias:
            matriz_coocorrencia = graycomatrix(imagem_quantizada, distances=[distancia], angles=angulos, levels=niveis, symmetric=True, normed=True)
            
            entropia = graycoprops(matriz_coocorrencia, 'ASM')[0, 0]  # Aproximação de Entropia
            homogeneidade = graycoprops(matriz_coocorrencia, 'homogeneity').mean()
            contraste = graycoprops(matriz_coocorrencia, 'contrast').mean()
            
            descritores.append((entropia, homogeneidade, contraste))
        
        return descritores
    
    def show_haralick(self):
        descritores = self.calculate_haralick()
        if descritores:
            message = f"Haralick Descriptors:\n{descritores}"
            self.show_message("Haralick Descriptors", message)

    def calculate_hu_moments(self):
        if not self.check_image_loaded():
            return
        
        imagem = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        momentos = cv2.moments(imagem)
        momentos_hu = cv2.HuMoments(momentos).flatten()
        
        return momentos_hu

    def show_hu_moments(self):
        momentos_hu = self.calculate_hu_moments()
        if momentos_hu is not None:
            message = f"Hu Moments:\n{momentos_hu}"
            self.show_message("Hu Moments", message)

    def classify_sub_image(self):
        if not self.check_image_loaded():
            return

        # Carregar o modelo salvo
        modelo = load_model('./codigo/modelo_multiclasse.keras')
        # Caminho da imagem a ser classificada
        caminho_da_imagem = self.image_path
        # Carregar a imagem
        img = Image.open(caminho_da_imagem)
        # Redimensionar a imagem para o tamanho esperado pelo modelo (exemplo: 224x224)
        img = img.resize((224, 224))
        # Converter a imagem em array numpy
        img_array = image.img_to_array(img)
        # Expandir as dimensões para que corresponda ao esperado pelo modelo (ex: (1, 224, 224, 3))
        img_array = np.expand_dims(img_array, axis=0)
        # Normalizar os valores da imagem se necessário (por exemplo, escala entre 0 e 1)
        # img_array = img_array / 255.0
        # Fazer a previsão
        previsoes = modelo.predict(img_array)
        
        # Interpretar o resultado
        classe_predita = np.argmax(previsoes, axis=1)
        message = f"Classe Predita: {classe_predita}"
        self.show_message("Classificação da Sub-Imagem", message)

app = QApplication(sys.argv)
ex = ImageApp()
sys.exit(app.exec_())
