# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

# Check TensorFlow and Keras version
print('TensorFlow version:',tf.__version__)
print('Keras version:',keras.__version__)

# Confirm whether GPU is connected
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

"""# 1. Read File"""

# Connect to Google Drive
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path

# Get folder path
data_path = Path("./drive/MyDrive/Colab Notebooks/Fast_Furious_Insured")

# Read csv files into DataFrames
train_df = pd.read_csv(data_path/"train.csv")
test_df = pd.read_csv(data_path/"test.csv")

"""# 2. Data Preparation"""

# Change correct data type (str) to fit the model
train_df["Condition"] = train_df["Condition"].astype(str)

# Load and transform image 
image_data_folder = Path("./drive/MyDrive/Colab Notebooks/Fast_Furious_Insured/trainImages")
pretrained_size = (224,224)
batch_size = 30

print("Getting Data...")
datagen = ImageDataGenerator(rescale=1./255, # normalize pixel values
                             validation_split=0.3) # hold back 30% of the images for validation

print("Preparing training dataset...")
train_generator = datagen.flow_from_dataframe(
    train_df,
    image_data_folder,
    x_col = "Image_path",
    y_col = "Condition",
    target_size = pretrained_size, # resize to match model expected input
    batch_size = batch_size,
    class_mode = "categorical",
    subset = "training") # set as training data

print("Preparing validation dataset...")
validation_generator = datagen.flow_from_dataframe(
    train_df,
    image_data_folder,
    x_col = "Image_path",
    y_col = "Condition",
    target_size = pretrained_size, # resize to match model expected input
    batch_size = batch_size,
    class_mode = "categorical",
    subset = "validation") # set as validation data

classnames = list(train_generator.class_indices.keys())
print("class names: ", classnames)

"""# 3. Transfer Learning"""

# Load pre-trained model as base model
#base_model = keras.applications.resnet.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

base_model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3), include_top = False, weights = "imagenet")

# View base model architecture
print(base_model.summary())

"""# Create Prediction Layer (2nd run onwards)"""

# Freeze pre-trained layers in the base model
base_model.trainable = False

# Create prediction layer for image classification
x = base_model.output
x = Flatten()(x)
prediction_layer = Dense(len(classnames), activation='softmax')(x) 
model = Model(inputs=base_model.input, outputs=prediction_layer)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# View full model (including layers of the base model and the added dense layer)
print(model.summary())

"""# Train Model"""

# Train the model over 3 epochs
num_epochs = 4
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = num_epochs)

"""# View Loss History"""

epoch_nums = range(1,num_epochs+1)
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

"""# Evaluate Model Performance"""

print("Generating predictions from validation data...")
# Get the image and label arrays for the first batch of validation data
x_test = validation_generator[0][0]
y_test = validation_generator[0][1]

# Use the model to predict the class
class_probabilities = model.predict(x_test)

# The model returns a probability value for each class
# The one with the highest probability is the predicted class
predictions = np.argmax(class_probabilities, axis=1)

# The actual labels are hot encoded (e.g. [0 1 0], so get the one with the value 1
true_labels = np.argmax(y_test, axis=1)

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classnames))
plt.xticks(tick_marks, classnames, rotation=85)
plt.yticks(tick_marks, classnames)
plt.xlabel("Predicted Shape")
plt.ylabel("Actual Shape")
plt.show()

# Save the trained model
modelFileName = 'ResNet50_classifier1.h5'
model.save(modelFileName)
print('model saved as', modelFileName)

"""# 4. Prediction on Test Data
Note: Create a new folder in "testImages", name it as "images", transfer all images into the folder "images" to en
"""

# Load and transform test image 
image_data_folder = Path("./drive/MyDrive/Colab Notebooks/Fast_Furious_Insured/testImages")
pretrained_size = (224,224)
batch_size = 30

datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
        image_data_folder,
        target_size = pretrained_size,
        batch_size = batch_size,
        class_mode= "categorical",
        shuffle=False)

test_generator.reset()

# Predict from generator (returns probabilities)
pred=model.predict_generator(test_generator, steps=len(test_generator), verbose=1)

# Get classes by rounding the predicted probabilities
cl = np.round(pred)

# Get filenames 
filenames = test_generator.filenames

# Save prediction results to a DataFrame
result_df = pd.DataFrame({"Image_path":filenames, "Condition":cl[:,0]})

# Clean image path to match with test set
result_df['Image_path'] = result_df['Image_path'].map(lambda x: x.replace('testImages/', ''))

result_df

# Merge prediction result to the test set
left_join = pd.merge(test_df , result_df, on ='Image_path', how ='left')

left_join

# Export result to csv
left_join.to_csv(data_path/'image_classification_result.csv', index=False)
