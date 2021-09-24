TSAI Assignment: Sunny Manchanda

# END3.0 Assignment 1

Assignment:

1. Rewrite the Colab file and 

      i.remove the last activation function
      
      ii. make sure there are in total 44 parameters
      
      iii. run it for 2001 epochs

2. You must upload your assignment to a public GitHub Repository and share the link as the submission to this assignment 

3. Add a readme file to your project and describe these things:

      i. What is a neural network neuron?

      ii. What is the use of the learning rate?

      iii. How are weights initialized?

      iv. What is "loss" in a neural network?

      v. What is the "chain rule" in gradient flow?

4. This assignment is worth 300pts (150 for Code, and 150 for your readme file (directly proportional to your description). 

Answers:

i. What is a neural network neuron?

Answer: An artificial neural network neuron is a mathematical function that forms the building block of a an artificial neural network. These artificial neurons when stacked up as multiple layers, enable compositional power of an artificial neural network. A neural network neuron is loosely inspired by the information processing structure of a brain albeit in a simple form. A neural network neuron takes multiple inputs and sums them up to produce an output which is passed through a non-linear (activation/transfer) function.  

A neuron in a linear network acts as a linear classifier w'X where X is the data and w are the weights of the neural network.

![image](https://user-images.githubusercontent.com/91345062/134740815-a674b178-8a23-435c-9164-cbe64077c600.png)

Ref: (https://playground.tensorflow.org/#activation=tanh&batchSize=1&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.00385&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

The objective of neural network training is to learn which weights are important in order to learn a good mapping between the inputs and outputs. This is usually done using the machinery of optimization using gradient based techniques like gradient descent.

ii. What is the use of learning rate?

Since neural networks loss are non-convex in nature it is difficult to use the tool set of convex optimzation or to search for a global minima. Moreover the feature space is generally quite large, which makes the numerical optimization both hard and an computationally expensive task. Though second order methods like computing the Hessian could be used for optimization, generally first order gradient based optimization methods are used because gradients are relatively easy to compute. Now, the gradients give the direction of steepest ascent. For optimizing the weights of a neural network (say a feed forward neural network, MLP) moving in the direction opposite to the gradient would nudge to weights
in a direction that reduces the loss function. 

      Gradient Descent: new_weight = existing_weight — learning_rate * gradient

While the gradient determines in which direction to move, the learning rate determines the rate at which we move in the direction opposite to the gradient. Since we do not know the optimal minimum loss value of the neural network, the learning rate helps control the movement towards the region of local minimum of the loss function.

![image](https://user-images.githubusercontent.com/91345062/134743088-acc86598-b5bd-46e8-87b5-97b1bd8c3d22.png)

If the learning rate is too slow, the learning will be slow since it will take a long time to reach a region of hopefully a local minima, while a very large learning rate could result in overshooting the region of local minima. Thus, the learning rate is an empirical parameter set as part of what is called as hyperparameter tuning along with other neural network parameters. 

![image](https://user-images.githubusercontent.com/91345062/134743221-6c1c8ed4-7e8c-41f7-a945-1d79a0b5c8a3.png)
     
iii. How are weights initialized?

Weight initialization is important aspect for efficient learning of the neural network. The way typical neural network training occurs is 

a. Initialize the neural network parameters
b. Decide on the optimization algorithm
c. Do a Forward Pass of the Neural Network
d. Compute gradient of the loss function wrt neural network weights using algorithm like back propagation
e. update parameters using gradient descent

Initilization is in face critical to neural network performance. If we initialize the neural network with any constant weight, the network will not learn. This is due to the effect of symmetry. For example assume we take two inputs x1 and x2 to the neural network and we initialize the weights using 'beta' for the hidden layer. Then each neuron of the hidden layer would have the same tanh(beta * x1 + beta * x2) as the output. So all the neurons of this hidden layer would have the same effect on the cost, and thus same gradient wrt to the loss function. So all neurons would behave symmetrically rather than learning different features. This would not allow the neural network to learn. Thus it is import to have random weights for initializing the neural network. 

Secondly, if the outputs of the neuron are too large or small (or the network is deep) can lead to the problem of exploding or vanishing gradients respectively. A good weight initialization scheme allows to reduce the severity of exploding/vanishing gradient. 

As a good heuristic it is recommended to initialize weights as random between -1 to 1 (variance of 1 or same across every layer) with mean as 0. Although, a good weigh initialization scheme does depends on the choice of the activation function. If the activation function is tanh it is recommended to initialize with 'Xavier initialization' among others, while for activation function ReLU, 'He initialization' is recommended. 

Xavier initialization :
All the weights of layer l are picked randomly from a normal distribution with mean μ=0 and variance 1/(#neurons in l-1 layer) 
Biases are initialized with zeros.

iv. What is "loss" in a neural network?

The difference between the predicted output of the neural network vs the ground truth/desired output is referred to as loss of a neural network. The higher the deviation of the ground truth the higher is the loss if the neural network. The loss of the neural network is non-convex. The loss function acts as the feedback for the optimization process to tune the weights of the neural network such that the loss of the network reduces. Thus, the loss of the neural network acts as the objective function which we want to minimize using an appropriate optimization method. For regression problems we generally have a regression loss like Mean Square Error (MSE), for discrete value/classification problems cross entropy loss/negative log likelihood loss is common, while for self-supervised learning usually contrastive loss is used. Also the loss function of the neural network depends on the problem being solved. For example, CTC loss is commonly used for speech processing while centre loss is common for biometric problems. Choice of loss function has a whole research area in neural network research commonly referred to as 'metric learning'.

v. What is the "chain rule" in gradient flow?
Generally following are the steps followed for a Neural Network training: 

a. Initialize the neural network parameters
b. Decide on the optimization algorithm
c. Do a Forward Pass of the Neural Network
d. Compute gradient of the loss function wrt neural network weights using algorithm like back propagation
e. update parameters using gradient descent

For the step d above, in order to fine tune the neural network weights to achieve a lower loss, we need to compute the gradient of the weights of the neural network wrt the loss function at the output neuron.  The computation of this gradient for the weights on the hidden neurons is computed using backpropagation. The gradient from the output node blows backward to the hidden nodes and is computed using the chain rule of calculus.

![image](https://user-images.githubusercontent.com/91345062/134747885-c42756b9-df39-46bd-9407-8af7817fcab3.png)

Ref for Image: CS231n

A node with inputs  (x,y)  and output  z=h(x,y) , where  h  represents the function performed at the node. Lets assume that the gradient  ∂L/∂z  is known. Using the Chain Rule of Differentiation, the gradients  ∂L/∂x  and  ∂L/∂y  can be computed as:
∂L/∂x = ∂L/∂z * ∂z/∂x
 
∂L/∂y = ∂L/∂z * ∂z/∂y
