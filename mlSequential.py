import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

# Define the dataset directory and class names
train_dir = 'clothing-dataset-small-master/train/'
test_dir = 'clothing-dataset-small-master/test/'
val_dir = 'clothing-dataset-small-master/validation/'
class_names = ['pants', 't-shirt', 'skirt', 'dress', 'shorts', 'shoes', 'hat', 'longsleeve', 'outwear', 'shirt']
num_classes = len(class_names)

# Define a function to load and preprocess an image
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Define a function to load and preprocess a dataset with one-hot encoded labels
def load_dataset_one_hot(folder):
    data = []
    labels = []
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(folder, class_name)
        file_names = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
        labels.extend([tf.one_hot(i, num_classes)] * len(file_names))
        data.extend(file_names)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y))
    return dataset

# Load the dataset with one-hot encoded labels
train_dataset = load_dataset_one_hot(train_dir).batch(batch_size=32)
test_dataset = load_dataset_one_hot(test_dir).batch(batch_size=32)
val_dataset = load_dataset_one_hot(val_dir).batch(batch_size=32)

def plot_training_history(history, save_path=None):
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Definirea arhitecturii modelului
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(256, activation='relu'), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compilarea modelului cu parametrii impuși
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Antrenarea modelului cu monitorizarea acurateții pe setul de validare
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
history = model.fit(train_dataset, batch_size=32, epochs=50, validation_data=val_dataset,  callbacks=[early_stop, reduce_lr])

# Evaluarea modelului pe datele de test
test_loss, test_accuracy = model.evaluate(test_dataset)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Salvarea modelului
model.save('sequentialModel.h5')

# Generarea graficului
plot_training_history(history, save_path='sequentialModel_training_history.png')