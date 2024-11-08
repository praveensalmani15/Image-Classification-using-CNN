# Image-Classification-using-CNN
Image Classification: Cats vs Dogs

This project aims to build an image classification model using Convolutional Neural Networks (CNN) to classify images as either **cats** or **dogs**.

### Overview
The model uses **TensorFlow** and **Keras** to train on a dataset of images containing cats and dogs. The images are preprocessed and augmented to improve the performance and generalization of the model. The final model predicts whether an image is of a cat or a dog.

### Dataset
The dataset contains two main directories:
1. `training_set/` - Contains images for training the model.
   - **cats/** - Folder with images of cats.
   - **dogs/** - Folder with images of dogs.
2. `test_set/` - Contains images for validating the model.

### Steps:
1. **Data Preprocessing**: The images are resized to 150x150 pixels and normalized to bring the pixel values in the range of 0-1.
2. **Data Augmentation**: The images undergo augmentation (rotation, zoom, flipping, etc.) to improve the model's ability to generalize to unseen data.
3. **Model Architecture**:
   - **Convolutional Layers**: Used to detect patterns in the images.
   - **Max Pooling Layers**: To downsample the image and reduce dimensionality.
   - **Dense Layers**: To perform the final classification.
4. **Model Training**: The model is trained using the Adam optimizer and binary cross-entropy loss function, with early stopping to avoid overfitting.

### Training
The model is trained for 25 epochs with the following parameters:
- `Batch Size`: 20
- `Steps per Epoch`: 100
- `Validation Steps`: 100

### Model Evaluation
After training, the model's performance is evaluated based on its accuracy and loss values on both the training and validation sets.

### Example Prediction
Once the model is trained, it can be used to predict whether an image is of a cat or a dog. The example below demonstrates the process:

```python
img = image.load_img('/content/WhatsApp Image 2024-11-08.jpg', target_size=(150, 150))
img_pred = image.img_to_array(img)
img_pred = np.expand_dims(img_pred, axis=0)
result = model.predict(img_pred)
```
Based on the prediction result, the image is classified as either a "cat" or a "dog".

### Requirements
- TensorFlow >= 2.0
- Keras
- Matplotlib
- Numpy
- Pandas
