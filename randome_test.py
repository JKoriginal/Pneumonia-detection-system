from fastai.vision.all import *
import random
from PIL import Image

# Define paths
test_path = Path("D:\\pneumonia\\Dataset\\random_test")
learn_inf = load_learner('D:\\pneumonia1\\export.pkl')  # Load the learner

# Function to display images with predictions
def show_predictions(test_images):
  fig, ax = plt.subplots(2, 5, figsize=(12, 6))  # Create a 2x5 grid

  for i, img_path in enumerate(test_images):
    img = Image.open(img_path)
    img = img.resize((128, 128))  # Adjust to match training image size

    # Get prediction
    predicted_label, _, _ = learn_inf.predict(img)

    # Display image and predicted label
    row = i // 5
    col = i % 5
    ax[row, col].imshow(img)
    ax[row, col].axis('off')
    ax[row, col].set_title(f"{img_path.name}\n{predicted_label}", fontsize=8)

  plt.tight_layout()
  plt.show()

# Get 10 random images from test set
test_images = get_image_files(test_path)
random_images = random.sample(test_images, len(test_images)) #10 images

# Show predictions for the random images
show_predictions(random_images)
