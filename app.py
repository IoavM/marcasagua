import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Quitar Marcas de Agua", layout="wide")

st.title("🧽 Quitar marcas de agua de una imagen")
st.write("Sube una imagen, dibuja sobre las marcas de agua y la aplicación las eliminará usando la técnica de inpainting.")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Leer la imagen
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    # Configurar dos columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Dibuja sobre la marca de agua")
        st.write("Usa el lápiz para dibujar sobre las áreas que deseas eliminar")
        
        # Configurar el canvas para dibujar
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",  # Color semitransparente para ver qué se dibuja
            stroke_width=st.slider("Grosor del pincel", 5, 40, 20),
            stroke_color="#FFFFFF",
            background_image=image,
            update_streamlit=True,
            height=image_np.shape[0],
            width=image_np.shape[1],
            drawing_mode="freedraw",
            key="canvas",
        )
    
    if canvas_result.image_data is not None:
        # Crear la máscara binaria
        mask = cv2.cvtColor(np.array(canvas_result.image_data), cv2.COLOR_RGBA2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)  # Corrección del typo
        
        with col2:
            st.subheader("2. Resultado")
            
            # Mostrar la máscara
            st.write("Máscara generada:")
            st.image(mask, caption="Máscara de marca de agua", use_column_width=True)
            
            # Botón para aplicar inpainting
            inpaint_method = st.radio(
                "Método de inpainting:",
                ["TELEA (Rápido)", "NS (Mejor calidad, más lento)"],
                index=0
            )
            
            inpaint_radius = st.slider("Radio de inpainting", 1, 10, 3, 
                                      help="Radio que determina el tamaño del área a considerar para rellenar. Valores más altos pueden mejorar el resultado pero toman más tiempo.")
            
            if st.button("Quitar marca de agua"):
                # Seleccionar método de inpainting
                method = cv2.INPAINT_TELEA if "TELEA" in inpaint_method else cv2.INPAINT_NS
                
                # Aplicar inpainting
                with st.spinner("Procesando imagen..."):
                    inpainted = cv2.inpaint(image_np, mask, inpaint_radius, method)
                
                # Mostrar resultado
                st.write("Imagen sin marca de agua:")
                st.image(inpainted, caption="Resultado final", use_column_width=True)
                
                # Opción para descargar la imagen resultante
                result_pil = Image.fromarray(inpainted)
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                
                st.download_button(
                    label="Descargar imagen sin marca de agua",
                    data=buf.getvalue(),
                    file_name="imagen_sin_marca_de_agua.png",
                    mime="image/png"
                )
else:
    st.info("👆 Sube una imagen para comenzar")
    
st.write("---")
st.write("""
### Instrucciones:
1. Sube una imagen que contenga marcas de agua
2. Dibuja sobre las áreas que contienen las marcas de agua
3. Ajusta los parámetros si es necesario
4. Haz clic en 'Quitar marca de agua'
5. Descarga el resultado
""")
