import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_ds_cnn(input_shape=(49, 40, 1), num_classes=2):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(64, (10, 4), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (1, 1), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    print("Building DS-CNN...")
    model = build_ds_cnn()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Generate dummy data for the purpose of this minimal version (in a real scenario we'd use Speech Commands)
    print("Generating dummy training data for 'marvin'...")
    x_train = np.random.rand(100, 49, 40, 1).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)
    
    print("Training model...")
    model.fit(x_train, y_train, epochs=1, batch_size=16)
    
    print("Saving float32 model to ml/model.h5")
    model.save("ml/model.h5")
    
    print("Quantizing to INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_dataset():
        for _ in range(10):
            yield [np.random.rand(1, 49, 40, 1).astype(np.float32)]
            
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant_model = converter.convert()
    
    with open("ml/model_quantized.tflite", "wb") as f:
        f.write(tflite_quant_model)
    print("Saved INT8 TFLite model to ml/model_quantized.tflite")

if __name__ == "__main__":
    main()
