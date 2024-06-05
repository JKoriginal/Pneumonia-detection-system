import cv2

# Replace with your actual model path
learn_inf = load_learner(Path('D:\\pneumonia1\\export.pkl'))

# Load an image (replace with your image path)
img = cv2.imread("D:\\pneumonia\\Dataset\\random_test\\1.jpeg")

# Resize the image to match model input size (assuming 224x224)
img = cv2.resize(img, (224, 224))

# Convert BGR to RGB (OpenCV uses BGR by default, FastAI expects RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Make prediction using FastAI
pred, _, _ = learn_inf.predict(img)
print(f"Predicted class: {pred}")
