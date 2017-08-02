# PyNet
Neural Nets framework in Python, from scratch

1. Extract "mnist_png.7z" (extract here, where the file is) (from: https://github.com/myleott/mnist_png)

2. cd PyNet/pynet/

3. Run "python test_mnist.py"


## CIFAR-10

1. download cifar-10: wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

2. Extract it in the root

3. cd PyNet/pynet/

4. Run "python test_cifar.py"


## NOTES

The default run will load the saved weights and test the net.
If you want to train it yourself, change

load_weights_from_file = True 
to
load_weights_from_file = False

in the test files