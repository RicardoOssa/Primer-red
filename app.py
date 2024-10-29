import tensorflow as tf
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Cargar el modelo previamente entrenado 
model = tf.keras.models.load_model(r'C:\Users\LAPTOP\OneDrive\Escritorio\UdeA\Ingenieria de materiales\Celdas de combustible\Python\BooCham\mlp_modelo_final.h5')

#funcion para procesar imagen

def preprocess_image(image):
    # Convertir la imagen a escala de grises y redimensionar a 28x28
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    image = np.reshape(image, (1, 28*28))
    return image

# Titulo framework
st.title('Clasificacion de titulos Manuscritos')

# Cargar la imagen

uploaded_file = st.file_uploader("Carga una imagen de un digito(0-9)", type=['png', 'jpg'])

if uploaded_file is not None:
    # Convertir imagen con streamlt
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    #preprocesar la imagen
    processed_image = preprocess_image(image)
    # Hacer la prediccion
    prediction = model.predict(processed_image)
    predict_digit =  np.argmax(prediction) 
    # Mostrar la prediccion
    st.write(f"Prediccion: **{predict_digit}")
    # Mostrar la probabilidad de cada clase
    for i in range(10):
        st.write(f"Digito {i}: {prediction[0][i ]:.4f}")
    
    #Mostrar imagen procesada para ver el preprocesamiento
    plt.imshow(processed_image.reshape(28,28), cmap='gray')
    plt.axis('off')
    st.pyplot(plt)