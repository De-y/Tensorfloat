import tensorflow as tf
import os
import re
import numpy as np

# Define the neural network model
class MathModel(tf.keras.Model):
    def __init__(self):
        super(MathModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=256, output_dim=64, input_length=50)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

def train_math_model(folder_path):
    inputs = []
    outputs = []

    # read files from the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                input_seq = file.readline().strip()
                input_seq = re.sub('[()]', '', input_seq)  # remove non-numeric characters
                output_seq = input_seq.split('=')
                print("input_seq:", input_seq)
                print("output_seq:", output_seq)
                inputs.append(input_seq)
                input_seq = input_seq.split('=')
                outputs.append(np.array(float(output_seq[1])))

    # determine the maximum length of the input sequence
    max_length = max([len(seq) for seq in inputs])

    # tokenize the input sequences
    token_dict = {char: i+1 for i, char in enumerate(set(''.join(inputs)))}
    token_dict['('] = len(token_dict) + 1  # Add '(' to token_dict
    token_dict[')'] = len(token_dict) + 1
    inputs = np.array([[token_dict[char] for char in seq] + [0]*(max_length-len(seq)) for seq in inputs])

    # convert outputs to 1D array
    outputs = np.squeeze(outputs)

    # return the inputs and outputs
    return inputs, outputs, token_dict, max_length


def run_math_model(model, input_text, token_dict, max_length):
    # Parse the input text to get input tensor
    input_seq = re.sub('[^0-9+\-*/(). ]', '', input_text)
    input_tensor = np.array([[token_dict[char] for char in input_seq]])

    # Pad the input sequence to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length, padding='post')

    # Make prediction using the model
    prediction = model(input_tensor)

    # Return the prediction
    return prediction.numpy()[0][0]

if __name__ == '__main__':
    # Define the folder path containing the training data
    folder_path = './math/'

    # Train the math model
    inputs, outputs, token_dict, max_length = train_math_model(folder_path)

    # Initialize and compile the math model
    math_model = MathModel()
    math_model.compile(loss='mse', optimizer='adam')

    # Train the math model using the obtained inputs and outputs
    math_model.fit(inputs, outputs, epochs=1000, batch_size=16, validation_split=0.2)

    # Test the math model
    input_text = "sin(45)="
    prediction = run_math_model(math_model, input_text, token_dict, max_length)
    print("Prediction: %.2f" % prediction)
