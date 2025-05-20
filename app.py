import streamlit as st
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
        st.subheader("1. Visualiza la imagen original")
        st.image(image, caption="Imagen original", use_column_width=True)
        st.write("⬇️ Dibuja en el área en blanco debajo, copiando la posición de las marcas de agua")
        
        # Dimensiones para el área de dibujo
        width = min(image_np.shape[1], 800)
        height = min(image_np.shape[0], 600)
        
        # Crear un canvas simple sin imagen de fondo
        import base64
        from streamlit_drawable_canvas import st_canvas
        
        st.write("Área de dibujo (dibuja sobre las marcas de agua):")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",
            stroke_width=st.slider("Grosor del pincel", 5, 40, 20),
            stroke_color="#FF0000",  # Rojo para mejor visibilidad
            background_color="#FFFFFF",  # Fondo blanco
            height=height,
            width=width,
            drawing_mode="freedraw",
            key="canvas",
        )
    
    if canvas_result.image_data is not None and uploaded_file is not None:
        with col2:
            st.subheader("2. Procesamiento")
            
            # Crear la máscara binaria
            mask_array = np.array(canvas_result.image_data)
            mask = cv2.cvtColor(mask_array, cv2.COLOR_RGBA2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            
            # Redimensionar la máscara al tamaño de la imagen original
            mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Mostrar la máscara
            st.write("Máscara generada (áreas a eliminar):")
            st.image(mask, caption="Máscara de marca de agua", use_column_width=True)
            
            # Botón para aplicar inpainting
            inpaint_method = st.radio(
                "Método de inpainting:",
                ["TELEA (Rápido)", "NS (Mejor calidad, más lento)"],
                index=0
            )
            
            inpaint_radius = st.slider("Radio de inpainting", 1, 10, 3, 
                                    help="Radio que determina el tamaño del área a considerar para rellenar.")
            
            if st.button("Quitar marca de agua"):
                # Seleccionar método de inpainting
                method = cv2.INPAINT_TELEA if "TELEA" in inpaint_method else cv2.INPAINT_NS
                
                # Aplicar inpainting
                with st.spinner("Procesando imagen..."):
                    try:
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
                    except Exception as e:
                        st.error(f"Ocurrió un error durante el procesamiento: {str(e)}")
                        st.write("Intenta dibujar nuevamente o sube otra imagen.")
else:
    st.info("👆 Sube una imagen para comenzar")
    
st.write("---")
st.write("""
### Instrucciones:
1. Sube una imagen que contenga marcas de agua
2. Observa la imagen original y luego dibuja sobre el área blanca siguiendo la ubicación de las marcas de agua
3. Usa el pincel rojo para marcar las áreas que quieres eliminar
4. Ajusta los parámetros si es necesario
5. Haz clic en 'Quitar marca de agua'
6. Descarga el resultado
""")

# Información adicional
expander = st.expander("Más información")
with expander:
    st.write("""
    ### ¿Cómo funciona el inpainting?
    
    El inpainting es una técnica que permite reconstruir partes de una imagen utilizando la información de las áreas circundantes.
    Los algoritmos analizan los píxeles alrededor de la región marcada y generan nuevos píxeles que mantienen la coherencia visual.
    
    #### Métodos disponibles:
    - **TELEA**: Algoritmo rápido basado en Fast Marching Method, bueno para áreas pequeñas.
    - **NS (Navier-Stokes)**: Algoritmo basado en ecuaciones de fluidos, mejor para detalles y texturas complejas.
    
    #### Consejos:
    - Dibuja con precisión sobre la marca de agua
    - Ajusta el radio según el tamaño de la marca de agua
    - Para mejores resultados, prefiere el método NS en marcas de agua sobre áreas con texturas
    """)
