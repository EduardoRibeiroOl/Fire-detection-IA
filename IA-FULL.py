import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carrega o modelo salvo
model_path = r'your folder\name.keras'
model = load_model(model_path)

def preprocess_frame(frame):
    # Redimensiona a imagem para o tamanho esperado pelo modelo e normaliza
    img = cv2.resize(frame, (250, 250))
    img = img / 255.0  # Normalização
    img = np.expand_dims(img, axis=0)  # Adiciona uma dimensão para lote
    return img

# Inicializa a captura de vídeo (0 para webcam padrão, ou insira o caminho para um vídeo)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Lê o quadro da câmera
    if not ret:
        print("Falha ao capturar a imagem.")
        break
    
    # Processa o quadro para predição
    img = preprocess_frame(frame)
    prediction = model.predict(img)  # Faz a predição

    # Define o rótulo com base na predição
    label = "fogo" if prediction[0][0] > 0.5 else "sem fogo" #invertido o texto e a cor
    color = (0, 0, 255) if label == "Fogo detectado!" else (0, 255, 0)

    # Exibe o rótulo na imagem
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Detecção de Fogo em Tempo Real", frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
