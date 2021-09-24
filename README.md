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

Answer: An artificial neural network neuron is a mathematical function that forms the building block of a an artificial neural network. These artificiaol neurons when stacken up as multiple layers, enable compositional power of an artificial neural network. A neural network neuron is loosely inspired by the information processing structure of a brain albeit in a simple form. A neural network neuron taken mutiple inputs and sums them up to produce an output which is passed through a non-linear (activation/transfer) function.  

A neuron in a linear network acts as a linear classifier w'X where X is the data and w are the weights of the neural network.

![image](https://user-images.githubusercontent.com/91345062/134740815-a674b178-8a23-435c-9164-cbe64077c600.png)

Ref: (https://playground.tensorflow.org/#activation=tanh&batchSize=1&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.00385&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

The objective of neural network training is learn which weights are inpotant in order to learn a good mapping between the inputs and outputs. This is usually done using the machinery of optimization using gradient based techniques like gradient descent.

ii. What is the use of learning rate?

Since neural networks loss are non-convex in nature it is difficult to use the tool set of convex optimzation or to search for a global minima. Moreover the feature space is generally quite large make numerical optimzation both hard and an computationally expensive task. Though second order methods like computing the Hessian could be used for optimzation, generally first order gradient based optimzation methods are used because gradients are relatively easy to compute. Now, the gradients give the direction of steepest ascent. For optiming the weights of a neural network (say a feed forward neural network, MLP) moving in the direction opposite to the gradient would nudge to weights
in a direction that reduces the loss function. 

      Gradient Descent: new_weight = existing_weight — learning_rate * gradient

While the gradient determines in which direction to move, the learning rate detrmines the rate at which we move in the direction opposite to the gradient. Since we do not know the optimal minimum loss value of the neural network, the learning rate helps control the movement towards the region of local minimum of the loss function.

![image](https://user-images.githubusercontent.com/91345062/134743088-acc86598-b5bd-46e8-87b5-97b1bd8c3d22.png)

If the learning rate is too slow, the learning will be slow since it will take a long time to reach a region of hopefully a local minima, while a very large learning rate could result in overshooting the region of local minima. Thus learning rate is an empirical parameter set as part of what is called as hyperparameter tuning along with other neural network parameters. 

     
iii. How are weights initialized?

Weight intialization is important aspect for efficient learning of the neural network.  
