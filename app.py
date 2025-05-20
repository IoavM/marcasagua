import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image
import io
import base64

st.set_page_config(page_title="Quitar Marcas de Agua", layout="wide")

st.title("🧽 Quitar marcas de agua de una imagen")
st.write("Sube una imagen, dibuja sobre las marcas de agua y la aplicación las eliminará usando la técnica de inpainting.")

# Función para servir imagen compatible con st_canvas
def get_image_base64(pil_img):
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode()

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
        
        # Ajustar tamaño para mejor visualización en el canvas
        # Limitar a un tamaño máximo para evitar problemas de rendimiento
        max_width = 800
        max_height = 600
        width = image_np.shape[1]
        height = image_np.shape[0]
        
        # Calcular la relación de aspecto y redimensionar si es necesario
        if width > max_width or height > max_height:
            if width/height > max_width/max_height:  # Si es más ancha que alta
                new_width = max_width
                new_height = int(height * (max_width / width))
            else:  # Si es más alta que ancha
                new_height = max_height
                new_width = int(width * (max_height / height))
            
            # Guardar las dimensiones originales para el procesamiento posterior
            original_dimensions = (width, height)
            # Redimensionar para visualización
            display_image = image.resize((new_width, new_height))
            canvas_width = new_width
            canvas_height = new_height
        else:
            # Usar dimensiones originales
            display_image = image
            canvas_width = width
            canvas_height = height
            original_dimensions = None
        
        # Mostrar la imagen original
        st.image(display_image, caption="Imagen original", use_column_width=True)
        
        # Configurar el canvas para dibujar
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",  # Color semitransparente
            stroke_width=st.slider("Grosor del pincel", 5, 40, 20),
            stroke_color="#FFFFFF",
            background_color="#000000",  # Fondo negro
            background_image=display_image,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key="canvas",
        )
    
        if canvas_result.image_data is not None:
            # Crear la máscara binaria
            mask_array = np.array(canvas_result.image_data)
            mask = cv2.cvtColor(mask_array, cv2.COLOR_RGBA2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            
            # Si redimensionamos la imagen para el canvas, redimensionamos la máscara de vuelta al tamaño original
            if original_dimensions is not None:
                mask = cv2.resize(mask, original_dimensions, interpolation=cv2.INTER_NEAREST)
            
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
                    # Asegurarse de que la máscara y la imagen tengan el mismo tamaño
                    if original_dimensions is not None:
                        # Usamos la imagen original para el inpainting
                        img_for_inpaint = image_np
                    else:
                        img_for_inpaint = image_np
                    
                    # Seleccionar método de inpainting
                    method = cv2.INPAINT_TELEA if "TELEA" in inpaint_method else cv2.INPAINT_NS
                    
                    # Aplicar inpainting
                    with st.spinner("Procesando imagen..."):
                        # Verificar dimensiones
                        if mask.shape[:2] != img_for_inpaint.shape[:2]:
                            st.error(f"Error: La máscara ({mask.shape}) y la imagen ({img_for_inpaint.shape[:2]}) tienen dimensiones diferentes.")
                        else:
                            inpainted = cv2.inpaint(img_for_inpaint, mask, inpaint_radius, method)
                            
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
