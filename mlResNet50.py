import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt


# Clear the previous session 
tf.keras.backend.clear_session()

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

# Load the dataset with one-hot encoded labels
train_dataset = load_dataset_one_hot(train_dir).batch(batch_size=32)
test_dataset = load_dataset_one_hot(test_dir).batch(batch_size=32)
val_dataset = load_dataset_one_hot(val_dir).batch(batch_size=32)

# Definirea modelului
base_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pooling')(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
model = tf.keras.Model(inputs, outputs, name='image_classifier')

# Compilarea modelului cu parametrii impuși
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Avoid duplication for weight names
for i in range(len(model.weights)):
    model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)

# Antrenarea modelului cu monitorizarea acurateții pe setul de validare
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
history = model.fit(train_dataset, batch_size=32, epochs=50, validation_data=val_dataset, callbacks=[early_stop, reduce_lr])

# Evaluarea modelului pe datele de test
test_loss, test_accuracy = model.evaluate(test_dataset)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

for i, w in enumerate(model.weights): 
    print(i, w.name)

# Salvarea modelului
model.save('ResNet50Model.keras')

# Generarea graficului
plot_training_history(history, save_path='ResNet50_training_history.png')