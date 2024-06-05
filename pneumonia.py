import fastai
from fastai.vision.data import ImageDataLoaders
from fastai.vision import cnn_learner  # Import all submodules from fastai.vision
from fastai.metrics import error_rate
import os
import pandas as pd
import numpy as np
from pathlib import Path
from imgaug import augmenters as iaa

# Set data paths (replace with actual paths)
train_path = Path('D:/pneumonia/Dataset/chest_xray/train')
valid_path = Path('D:/pneumonia/Dataset/chest_xray/valid')
test_path = Path('D:/pneumonia/Dataset/chest_xray/test')

# Import PILImage
from PIL import Image

# Define data augmentation pipeline (customize as needed)
def my_aug_pipeline(img):
    """Applies data augmentation to a PIL image."""
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),  # Flip horizontally with 50% probability
        iaa.GaussianBlur(sigma=(0, 0.5)),  # Apply Gaussian blur with varying sigma
    ])
    return aug.augment_image(np.array(img))

# Load data with augmentation
data = ImageDataLoaders.from_dsets(
    path=train_path, valid_path=valid_path, seed=42,
    item_tfms=my_aug_pipeline, size=224, bs=32
)


# Define learner with chosen model (replace with desired architecture)
learn = cnn_learner(data, models.efficientnet_b3, metrics=[accuracy, Precision(average='weighted')], model_dir=Path('models'), path=Path('.'))

# Find suitable learning rate
learn.lr_find()
learn.recorder.plot(suggestions=True)

# Training (adjust epochs and learning rates based on validation)
lr1 = 1e-3  # Initial learning rate
lr2 = 1e-1  # Final learning rate
learn.fit_one_cycle(4, slice(lr1, lr2))

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10, slice(1e-4, 1e-3))

# Evaluation
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(10, 10))  # Show top 9 misclassified images

# Test on an image (replace with your image path)
img_path = 'c:\\Users\\dilee\\Desktop\\IM-0003-0001'
try:
    img = open_image(img_path)
    pred, pred_label, pred_prob = learn.predict(img)
    print(f"Predicted class: {pred_label} (Probability: {pred_prob:.4f})")
except FileNotFoundError:
    print(f"Error: Image file '{img_path}' not found.")
