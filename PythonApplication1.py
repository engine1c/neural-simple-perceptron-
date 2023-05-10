
import numpy as np

def activation_function(x):
    if x >= 0.5:
        return 1
    else:
        return 0

def predict(vodka, rain, friend):
    inputs = np.array([vodka, rain, friend])
    weights_input_to_hidden_1 = [0.25, 0.25, 0]
    weights_input_to_hidden_2 = [0.5, -0.4, 0.9]
    weights_input_to_hidden = np.array([weights_input_to_hidden_1, weights_input_to_hidden_2])
    weights_hidden_to_output = np.array([-1, 1])

    hidden_input = np.dot(weights_input_to_hidden, inputs)
    hidden_output = np.array([activation_function(x) for x in hidden_input])
    output = np.dot(weights_hidden_to_output, hidden_output)
    return activation_function(output) == 1

vodka = 0.0
rain = 1.0
friend = 0.0
result = predict(vodka, rain, friend)
print("result:", result)
