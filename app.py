import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image

st.title("🧽 Quitar marcas de agua de una imagen")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Dibuja sobre la marca de agua")
    canvas_result = st_canvas(
        fill_color="white",  # Color para la máscara
        stroke_width=10,
        stroke_color="white",
        background_image=image,
        update_streamlit=True,
        height=image_np.shape[0],
        width=image_np.shape[1],
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Crear la máscara binaria (zonas blancas = marcas a borrar)
        mask = cv2.cvtColor(np.array(canvas_result.image_data), cv2.COLOR_RGBA2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        st.subheader("Máscara generada")
        st.image(mask, caption="Máscara de marca de agua", use_column_width=True)

        if st.button("Quitar marca de agua"):
            # Aplicar inpainting
            inpainted = cv2.inpaint(image_np, mask, 3, cv2.INPAINT_TELEA)
            st.subheader("Resultado")
            st.image(inpainted, caption="Imagen sin marca de agua", use_column_width=True)
