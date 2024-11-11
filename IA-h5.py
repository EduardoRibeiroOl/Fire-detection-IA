import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configurações dos caminhos
train_dir = r'your folder\name.keras'
# Saída para o arquivo .keras
output_model_path = r'your folder\name.keras'

# Configurando o ImageDataGenerator para treinamento e validação
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)  # 20% para validação

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,  # pasta pai que contém as subpastas 'fire' e 'nofire'
    classes=['fire', 'nofire'],  # especifica as subpastas
    target_size=(250, 250),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    classes=['fire', 'nofire'],
    target_size=(250, 250),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Construindo o modelo de rede neural convolucional (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Saída para classificação binária (fogo/sim ou não)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=23,  # ajuste conforme necessário
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator)
)

# Salvando o modelo treinado em um arquivo .keras
model.save(output_model_path)
print(f"Modelo salvo em: {output_model_path}")

# Gráficos de precisão e perda durante o treinamento
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.show()
