# Automated Waste Classification Using CNN

A deep learning project that uses a Convolutional Neural Network (CNN) to classify waste images into seven categories. This project demonstrates end-to-end image classification with preprocessing, data augmentation, model training, and evaluation using TensorFlow/Keras.

---

## Project Objectives
- Automate the classification of waste images into appropriate categories.
- Address class imbalance using data augmentation techniques.
- Improve model generalization and evaluate performance effectively.
- Extract key insights from training and testing metrics.

---

## Dataset Overview
The dataset contains images labeled into the following 7 classes:
- Cardboard
- Food_Waste
- Glass
- Metal
- Other
- Paper
- Plastic

**Notes:**
- Images were resized and normalized.
- Dataset was split into training, validation, and testing sets.

---

## Technologies and Libraries Used
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV (cv2)
- Scikit-learn

**Development Environment:** Google Colab  
**Version Control:** Git

---

## Project Workflow

### 1. Data Preprocessing
- Loaded image dataset into NumPy arrays.
- Normalized pixel values.
- Encoded labels using `LabelEncoder`.
- Converted labels to categorical format with `to_categorical`.

### 2. Exploratory Data Analysis (EDA)
- Bar plot visualizations of class distribution.
- Observed significant class imbalance.

### 3. Data Augmentation
To enhance training data and reduce overfitting, used:

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


### Contact
gowthambavireddy@gmail.com
