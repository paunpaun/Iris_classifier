# Iris Classifier (simple MLP model, only Numpy) 

Structure: 4 inputs, 1 hidden layer with 4 neurons, output layer with 3 neurons. (expandable)

Inputs: "sepal_length", "sepal_width", "petal_length", "petal_width"

Hidden layer uses ReLU activation function, output layer uses softmax.

![image](https://github.com/user-attachments/assets/428e0e99-eac5-4df0-b3ac-96dc503db411)


Loss is calculated with Categorical Cross Entropy.

![structure](https://github.com/user-attachments/assets/eab47550-1b92-41a2-8aaa-61e724b3cde9)

Target: One-hot encoded "class" for the iris species.

![image](https://github.com/user-attachments/assets/e20939f8-ebb4-4e47-a8ad-0ebba77e645c)


![forward](https://github.com/user-attachments/assets/ce24a900-7780-47ae-8411-06950ba4adbc)  ![backwards](https://github.com/user-attachments/assets/5de0a14d-fe7d-45ee-abe5-f8594b75f2ef)

![formulas](https://github.com/user-attachments/assets/78eb1b1f-e1bc-4121-8063-5f278e3c2c4f)



Weights initialization.
![image](https://github.com/user-attachments/assets/83171d49-b367-4215-932a-eecf9ed333c3)

80% 20% split for training and testing.

![image](https://github.com/user-attachments/assets/180403da-ce9f-491f-b5e3-c3a3e1f8d20b)

0.01 for learning rate and 1000 total iterations gives about 95% accuracy.

![image](https://github.com/user-attachments/assets/b41f43d5-dbc4-403a-9e22-a2f8eab860b5)

![accuracy(terminal)](https://github.com/user-attachments/assets/bf59a062-189d-464c-ab58-2dc96466947a)   
![accuracy(plot)](https://github.com/user-attachments/assets/166175b3-a05b-46fd-b6f6-d040a0764d6b)
