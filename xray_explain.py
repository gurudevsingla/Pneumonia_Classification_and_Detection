import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Floating Result Precision
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow warnings
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM

# Define image size
IMAGE_SIZE = [224, 224]

# Function to resize an image
def resize_img(img):
    return tf.image.resize(img, IMAGE_SIZE)

# Function to convert Matplotlib plot to an image
def plot_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)

# Function to load and preprocess image
def process_image(image_path, model_path):
    # Load model
    loaded_mobilenet = tf.keras.models.load_model(model_path)

    # Read and preprocess image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = resize_img(image)

    # Predict
    image_batch = tf.expand_dims(image, axis=0)
    prediction = loaded_mobilenet.predict(image_batch)[0][0]
    scores = [1 - prediction, prediction]

    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

    # Grad-CAM explanation
    last_conv_layer_name = 'Conv_1_bn'

    # Grad-CAM explanation
    explainer = GradCAM()
    data = ([image.numpy()], None)
    grid = explainer.explain(data, loaded_mobilenet, class_index=0, layer_name=last_conv_layer_name)

    return image.numpy()/255.0, grid, scores, CLASS_NAMES
