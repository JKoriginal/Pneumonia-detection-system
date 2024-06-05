# Import libraries
import gradio as gr
from fastai.vision.all import *
from PIL import Image
import io

# Load the pre-trained model (replace 'path/to/export.pkl' with your actual path)
learn_inf = load_learner(Path('D:\\pneumonia1\\export.pkl'))

# Function to process uploaded image and make prediction
def predict_image(img):
    # Convert Gradio input (numpy array) to PIL Image
    img = PIL.Image.fromarray(img)

    # Resize the image to match model input size
    img = img.resize((128, 128))

    # Make prediction using the loaded model
    pred, _, _ = learn_inf.predict(img)
    
    # Return the predicted class label
    return pred

# Define the Gradio interface
interface = gr.Interface(
    fn=predict_image,  # Function to be called
    inputs=gr.components.Image(label="Upload Chest X-Ray"),  # Input component for image upload
    outputs=gr.components.Textbox(label="Predicted Pneumonia Type"),  # Output component for prediction
    title="Pneumonia Classifier",
    description="Upload an image of a chest X-ray to classify it as normal, bacterial pneumonia, or viral pneumonia.",
)

# Launch the Gradio interface
interface.launch()
