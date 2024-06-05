from fastai.vision.all import *
from PIL import Image

# Load the exported learner (assuming you've exported after training)
learn_inf = load_learner('D:\\pneumonia1\\export.pkl')

# Function to predict on a single image
def predict_image(img_path):
  # Open and resize the image
  img = Image.open(img_path)
  img = img.resize((128, 128))

  # Get prediction
  pred, _, _ = learn_inf.predict(img)
  return pred

# Example usage (replace with your image paths)
img_path1 = "D:\\pneumonia\\Dataset\\chest_xray\\test\\NORMAL\\IM-0001-0001.jpeg"
img_path2 = "D:\\pneumonia\\Dataset\\chest_xray\\test\\PNEUMONIA\\person86_bacteria_429.jpeg"

prediction1 = predict_image(img_path1)
prediction2 = predict_image(img_path2)

print(f"Image 1: Predicted class - {prediction1}")
print(f"Image 2: Predicted class - {prediction2}")
