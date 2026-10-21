
# Week 5 Task: Deep Learning Application in Data Science
# Project: Fashion-MNIST Clothing Image Classification

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# 1. Set random seeds for reproducibility
# --------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

# Fashion-MNIST class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# --------------------------------------------------
# 2. Load the public Fashion-MNIST dataset
# --------------------------------------------------
print("Loading Fashion-MNIST dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# --------------------------------------------------
# 3. Preprocess the image data
# Pixel values change from 0-255 to 0-1
# --------------------------------------------------
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Data preprocessing completed.")

# --------------------------------------------------
# 4. Display sample dataset images
# --------------------------------------------------
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(class_names[y_train[i]], fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_images.png", dpi=150)
plt.close()

# --------------------------------------------------
# 5. Design the neural network
# --------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),

    # Convert 28x28 image into a 1D vector
    tf.keras.layers.Flatten(),

    # Hidden layer 1
    tf.keras.layers.Dense(128, activation="relu"),

    # Reduce overfitting
    tf.keras.layers.Dropout(0.20),

    # Hidden layer 2
    tf.keras.layers.Dense(64, activation="relu"),

    # Output layer: 10 clothing classes
    tf.keras.layers.Dense(10, activation="softmax")
])

# --------------------------------------------------
# 6. Compile the model
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nNeural Network Architecture:")
model.summary()

# --------------------------------------------------
# 7. Train the model
# 20% of training data is used for validation
# Early stopping helps reduce overfitting
# --------------------------------------------------
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nTraining the neural network...")

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# --------------------------------------------------
# 8. Evaluate the model on test data
# --------------------------------------------------
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n========== TEST RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# --------------------------------------------------
# 9. Predict clothing classes
# --------------------------------------------------
y_probabilities = model.predict(x_test, verbose=0)

y_pred = np.argmax(y_probabilities, axis=1)

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(10)),
        target_names=class_names,
        zero_division=0
    )
)

# --------------------------------------------------
# 10. Generate and save the confusion matrix
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(11, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Fashion-MNIST Confusion Matrix")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("\nConfusion matrix saved.")

# --------------------------------------------------
# 11. Plot training and validation performance
# --------------------------------------------------
plt.figure(figsize=(12, 5))

# Accuracy graph
plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

# Loss graph
plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.close()

print("Training graph saved.")

# --------------------------------------------------
# 12. Display actual and predicted sample images
# --------------------------------------------------
plt.figure(figsize=(12, 8))

for i in range(15):
    plt.subplot(3, 5, i + 1)

    plt.imshow(x_test[i], cmap="gray")

    predicted_label = class_names[y_pred[i]]
    actual_label = class_names[y_test[i]]

    if y_pred[i] == y_test[i]:
        result_color = "green"
    else:
        result_color = "red"

    plt.title(
        f"Predicted: {predicted_label}\nActual: {actual_label}",
        fontsize=8,
        color=result_color
    )

    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.close()

# --------------------------------------------------
# 13. Save the trained model
# --------------------------------------------------
model.save("fashion_mnist_model.keras")

# --------------------------------------------------
# 14. Final output
# --------------------------------------------------
print("\n========== PROJECT COMPLETED ==========")

print("Generated files:")
print("1. sample_images.png")
print("2. confusion_matrix.png")
print("3. training_history.png")
print("4. sample_predictions.png")
print("5. fashion_mnist_model.keras")

print("\nDeep Learning classification completed successfully!")
