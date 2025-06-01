# import os
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # MobileNetV2 specific preprocessing

# # Load a pretrained MobileNetV2 model (without top layers)
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Add custom layers for your specific problem (optional)
# x = base_model.output
# x = GlobalAveragePooling2D()(x)  # Global average pooling to reduce dimensions
# x = Dense(1024, activation='relu')(x)  # Fully connected layer
# x = Dropout(0.5)(x)  # Dropout layer to prevent overfitting
# x = BatchNormalization()(x)  # Batch normalization for better performance
# predictions = Dense(1, activation='sigmoid')(x)  # Final output layer for binary classification

# # Create the model with the base model and the custom layers
# model = Model(inputs=base_model.input, outputs=predictions)

# # Freeze all layers of the pretrained MobileNetV2 model
# for layer in base_model.layers:
#     layer.trainable = False

# # Compile the model (no training, just for inference)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Save the model (optional)
# MODEL_PATH = 'vio_vedio.h5'
# model.save(MODEL_PATH)

# print("Pretrained MobileNetV2 model loaded and saved successfully!")

# # Image preprocessing for prediction
# def predict_image(img_path):
#     """Make predictions on a single image."""
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = preprocess_input(img_array)  # Preprocess the image for MobileNetV2

#     # Prediction
#     prediction = model.predict(img_array)
    
#     # Adjust the threshold if necessary
#     if prediction >= 0.5:
#         return "Violence detected"
#     else:
#         return "No violence detected"

# Example of using the function:
# result = predict_image("path_to_your_image.jpg")
# print(result)


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import Flask, request, jsonify
import io
from PIL import Image

# Paths
MODEL_PATH = 'combined_detection_model.h5'
DATASET_PATH = 'dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 2
EPOCHS = 5  # For demo; increase for real training

# Create the model
def create_model():
    input_layer = Input(shape=(224, 224, 3))
    base_model = DenseNet121(weights=None, include_top=False, input_tensor=input_layer)

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Prepare dataset
def prepare_data():
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_gen = datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    return train_gen, val_gen

# Train or load model
def load_or_train_model(model_path):
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Proceeding to retrain.")

    model = create_model()
    train_gen, val_gen = prepare_data()
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")
    return model

# Load or train the model once
model = load_or_train_model(MODEL_PATH)

# Setup Flask app
app = Flask(_name_)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
    except Exception as e:
        return jsonify({'error': f'Error processing the image: {e}'})

    try:
        prediction = model.predict(img)
        index = np.argmax(prediction[0])
        label = ['Theft', 'Violence'][index]

        return jsonify({'prediction': label,
                        'probabilities': prediction.tolist()[0]})
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {e}'})

# Run app
if _name_ == '_main_':
    app.run(debug=True)