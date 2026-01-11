import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Shape & Contour Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Advanced UI Styling (Gradient + Glass)
# ---------------------------------
st.markdown("""
<style>

/* Full background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    color: #f8fafc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid #334155;
}

/* Main title */
.main-title {
    font-size: 44px;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.sub-title {
    font-size: 18px;
    color: #cbd5f5;
    margin-bottom: 30px;
}

/* Glass card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 22px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0px 10px 40px rgba(0,0,0,0.4);
}

/* Section title */
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #e5e7eb;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.15);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Header
# ---------------------------------
st.markdown('<div class="main-title">Shape & Contour Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Contour-based geometric shape detection and feature extraction using Computer Vision</div>',
    unsafe_allow_html=True
)

# ---------------------------------
# Sidebar Controls
# ---------------------------------
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.write("Adjust parameters for better detection")

min_area = st.sidebar.slider("Minimum Contour Area", 100, 3000, 500)
canny_low = st.sidebar.slider("Canny Low Threshold", 10, 100, 50)
canny_high = st.sidebar.slider("Canny High Threshold", 100, 300, 150)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application uses contour extraction and feature-based classification "
    "to detect geometric shapes and compute area and perimeter."
)

# ---------------------------------
# Shape Detection Function
# ---------------------------------
def detect_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    sides = len(approx)

    if sides == 3:
        return "Triangle"
    elif sides == 4:
        return "Rectangle"
    elif sides > 4:
        return "Circle"
    else:
        return "Unknown"

# ---------------------------------
# Upload Section
# ---------------------------------
st.markdown("## 📤 Upload Image")
uploaded_file = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)
    original = img.copy()
    img_h, img_w = img.shape[:2]

    col1, col2 = st.columns(2, gap="large")

    # -------------------------------
    # Original Image
    # -------------------------------
    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Original Image</div>', unsafe_allow_html=True)
        st.image(original, width=420)
        st.markdown('</div>', unsafe_allow_html=True)


    # -------------------------------
    # Preprocessing (ROBUST VERSION)
    # -------------------------------

    # If image has 3 channels → convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # Already grayscale
        gray = img.copy()



    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    results = []
    valid_contours = []

    # -------------------------------
    # Analyze Contours
    # -------------------------------
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > min_area:
            peri = cv2.arcLength(cnt, True)
            shape = detect_shape(cnt)

            valid_contours.append(cnt)
            results.append([shape, round(area, 2), round(peri, 2)])

            # Draw contour
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 5)

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # -------- SAFE LABEL PLACEMENT --------
            label = shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3

            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            text_x = x
            text_y = y - 15

            # If text goes outside top boundary, move it inside/below
            if text_y - text_h < 0:
                text_y = y + text_h + 15

            # Background rectangle
            cv2.rectangle(
                img,
                (text_x, text_y - text_h - 10),
                (text_x + text_w + 10, text_y),
                (0, 0, 0),
                -1
            )

            # Text
            cv2.putText(
                img,
                label,
                (text_x + 5, text_y - 5),
                font,
                font_scale,
                (0, 255, 255),
                thickness
            )

    # -------------------------------
    # Detected Image
    # -------------------------------
    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Detected Shapes</div>', unsafe_allow_html=True)
        st.image(img, width=420)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # Metrics
    # -------------------------------
    st.markdown("## 📊 Detection Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Objects Detected", len(valid_contours))
    m2.metric("Min Area Threshold", f"{min_area} px²")
    m3.metric("Canny Thresholds", f"{canny_low} – {canny_high}")

    # -------------------------------
    # Results Table
    # -------------------------------
    if results:
        df = pd.DataFrame(
            results,
            columns=["Shape", "Area (px²)", "Perimeter (px)"]
        )
        st.markdown("## 📋 Shape Measurements")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No valid shapes detected. Adjust parameters.")

else:
    st.info("⬆️ Upload an image to start analysis")
