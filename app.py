import streamlit as st
from PIL import Image
from xray_explain import process_image
import os
import warnings
warnings.filterwarnings('ignore')

# Define paths
MODEL_PATH = "model/xray_model2.keras"
SAMPLE_IMAGES_PATH = "sample_images/"

# Sidebar for user input
st.sidebar.title("Pneumonia Detector")

option = st.sidebar.radio("Upload or Select an Image", ('Upload', 'Select from Sample'))

if option == 'Upload':
    uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image_path = 'uploaded_image.jpg'
        image.save(image_path)

elif option == 'Select from Sample':
    category = st.sidebar.radio("Choose a category", ('Pneumonic', 'Non-Pneumonic'))

    if category == 'Pneumonic':
        sample_images_path = SAMPLE_IMAGES_PATH + 'pneumonic'
    else:
        sample_images_path = SAMPLE_IMAGES_PATH + 'non_pneumonic'

    images = os.listdir(sample_images_path)
    selected_image = st.selectbox('Select an Image', images, index=1)
    image_path = sample_images_path + "/" + selected_image

    image = Image.open(image_path)
    st.image(image, caption='Selected Image', use_column_width=True)

# Process image and display results

if st.button('Predict'):
    if 'image_path' not in locals():  # Check if image_path exists
        st.warning("⚠️ Please upload or select an image before predicting.")

    else :
        with st.status("⏳ Analyzing the X-ray... Please wait.", expanded=False) as status:
            processed_image, gradcam_output, scores, CLASS_NAMES = process_image(image_path, MODEL_PATH)
            status.update(label="✅ Prediction complete!", state="complete", expanded=False)

        st.markdown(f"### 🩺 **Prediction Result:**")

        color_code = 0
        for score, name in zip(scores, CLASS_NAMES):
            if color_code==0 :
                st.markdown(
                    f"**<span style='color:green; font-size:22px;'>{100 * float(score):.2f}% {name}</span>**",
                    unsafe_allow_html=True)
                color_code+=1

            else:
                st.markdown(
                    f"**<span style='color:red; font-size:22px;'>{100 * float(score):.2f}% {name}</span>**",
                    unsafe_allow_html=True)

        # Display processed image and Grad-CAM output side by side
        col1, col2 = st.columns(2)
        col1.header("Processed Image")
        col1.image(processed_image, use_column_width=True)

        col2.header("Grad-CAM Output")
        col2.image(gradcam_output, use_column_width=True)

        st.markdown(
            "<span style='font-size:22px;'>👀Reason of my prediction - light (yellow) region of your image !!</span>",
            unsafe_allow_html=True)