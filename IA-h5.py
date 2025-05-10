import tensorflow as tf
import numpy as np
import cv2

# Carrega o modelo treinado
model = tf.keras.models.load_model('modelo_pessoa_fogo.h5')

# Classes — ajuste conforme as pastas usadas no treinamento
classes = ['fogo', 'pessoa']

# Inicializa a webcam (0 é a webcam padrão)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona o frame para o tamanho esperado pelo modelo
    resized = cv2.resize(frame, (150, 150))
    img_array = np.expand_dims(resized / 255.0, axis=0)

    # Faz a previsão
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index] * 100
    label = f"{classes[class_index]} ({confidence:.1f}%)"

    # Mostra o resultado no frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow('IA - Webcam', frame)

    # Sai com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a webcam e fecha a janela
cap.release()
cv2.destroyAllWindows()
