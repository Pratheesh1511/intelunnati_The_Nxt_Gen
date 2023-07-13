import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time

# Start time
start_time = time.time()

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape the data
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Convert the labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create the model
model = Sequential()

# Add the convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Add the dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=128)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)

# Print the accuracy and time
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('fashion_mnist_cnn_dropout.h5')

# End time
end_time = time.time()
print("Total Time Taken :",end_time-start_time)

# Save the output to a text file
with open("dropout_output.txt", "w") as f:
    f.write("Epoch\tAccuracy\tLoss\n")
    for epoch in range(len(history.history['accuracy'])):
        f.write("{}\t\t{:.4f}\t\t{:.4f}\n".format(epoch, history.history['accuracy'][epoch], history.history['loss'][epoch]))
    f.write("Accuracy: " + str(score[1]) + "\n")
    f.write("Time: " + str(end_time - start_time) + "'s")