from schwarpy_code.physics import ray_tracing
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from tqdm.auto import tqdm
import streamlit as st
from PIL import Image
import tempfile
import os

# --- UI Setup ---
st.set_page_config(page_title="Schwarpy", layout="wide")
st.title("Interactive Black Hole Ray Tracing!")

st.sidebar.header("Parameters")
inclination_deg = st.sidebar.slider("Inclination Angle (degrees)", 0, 90, 10)
l = st.sidebar.slider("Image Resolution (pixels)", 128, 1024, step=64, value=512)
r0 = st.sidebar.slider("Initial Radius (r0)", 1.0, 10.0, value=3.0, step=0.1)
dt = st.sidebar.slider("Time Step (dt)", 0.01, 0.5, value=0.075, step=0.025)

render_disk = st.sidebar.checkbox("Render Accretion Disk", value=True)

bg_choice = st.sidebar.selectbox(
    "Background Image",
    ["None", "Milky Way", "Nebula", "Rainbow Tile", "Upload your own"]
)

background_image_path = None

if bg_choice == "Milky Way":
    background_image_path = "milkyway.png"
elif bg_choice == "Nebula":
    background_image_path = "nebula.jpg"
elif bg_choice == "Rainbow Tile":
    background_image_path = "rainbowtile.jpg"
elif bg_choice == "Upload your own":
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        import tempfile
        import shutil
        # Save uploaded file temporarily and use its path
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)
        background_image_path = temp_path

# --- Convert and render ---
inclination_rad = np.radians(inclination_deg)
total_steps = int(2 * r0 / dt)
#estimated_time_sec = total_steps * l**2 
estimated_time = (l ** 2 / 128**2) * (0.5 / dt)
#k = 6.1e-5 
#estimated_time = k * (l ** 2) * (r0 / dt)
if estimated_time <= 60:
    st.sidebar.markdown(f"⏱️ **Estimated Render Time:** {estimated_time:.2f} seconds")
else:
    estimated_time /= 60
    st.sidebar.markdown(f"⏱️ Estimated Render Time: **~{estimated_time:.1f} minutes**")
status_placeholder = st.empty()

status_placeholder = st.empty()
progress_bar = st.progress(0)  # Set initial progress to 0%

if st.button("Start Render"):
    # Show the "Rendering..." message temporarily
    status_placeholder.info("Rendering...")

    M = ray_tracing(
        l=l,
        r0=r0,
        inclination=inclination_rad,
        render_disk=render_disk,
        background_image_path=background_image_path,
        dt=dt,
        progress_bar=progress_bar
    )

    # Clear the message after rendering is done
    status_placeholder.empty()

    # Display the rendered image
    fig, ax = plt.subplots(figsize=(8, 8))
    if M.ndim == 2:
        ax.imshow(M, cmap='gnuplot2', origin='lower', interpolation='bicubic')
    else:
        ax.imshow(M, origin='lower', interpolation='bicubic')
    ax.axis("off")
    ax.set_aspect('equal')

    st.pyplot(fig)
