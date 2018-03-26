import torch
from torch.autograd import Variable
from torch import FloatTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import gzip
import pickle


class MLPClassifier(nn.Module):
    def __init__(self, input_dim = 784, hidden_nodes=100, output_dim=10):
        super(MLPClassifier, self).__init__()
        self.lin_input = nn.Linear(input_dim, hidden_nodes)
        self.lin_hidden = nn.Linear(hidden_nodes, hidden_nodes)
        self.lin_output = nn.Linear(hidden_nodes, output_dim)
    def forward(self, x):
        x = self.lin_input(x)
        x = F.relu(x)
        x = self.lin_hidden(x)
        x = F.sigmoid(x)
        x = self.lin_output(x)
        # Apply output conversion
        x = F.sigmoid(x)
        return x

def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            m.weight.data.normal_(0, 1)
            #m.weight.data.normal_(0.1,0.2)

def calc_miscl_err(X, Y, Yhat):
    """
    Calculate the misclassification error (according to class canvas)
    """
    indsYhat=np.argmax(Yhat,axis=1)
    indsY=np.argmax(Y,axis=1)
    errors = (indsYhat-indsY)!=0
    return (100*sum(errors)/(Yhat.shape[0]*1.0))

def calc_avg_sq_err(X, Y, Y_hat):
    """
    Function for calculating the average squared error
    """
    # Define dimensionality
    N = len(X)
    d = len(X[0])
    # Calculate the error
    error = ((1)/(2*N)) * (np.linalg.norm( Y - Y_hat ))**2
    return error
def stand(x):
    # gives an affine transformation to place x between 0 and 1 (from canvas)
    x=x-np.min(x[:])
    x=x/(np.max(x[:])+1e-12)
    return x

def plot_10(X, eta, epochs):
    """
    Visualize ten weights
    """
    fig = plt.figure(figsize=(16,6))
    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(stand(X[1]).reshape(28,28), cmap='gray')
    axarr[0, 0].set_title('Node 1')
    axarr[0, 1].imshow(stand(X[3]).reshape(28,28), cmap='gray')
    axarr[0, 1].set_title('Node 2')
    axarr[0, 2].imshow(stand(X[5]).reshape(28,28), cmap='gray')
    axarr[0, 2].set_title('Node 3')
    axarr[0, 3].imshow(stand(X[7]).reshape(28,28), cmap='gray')
    axarr[0, 3].set_title('Node 4')
    axarr[0, 4].imshow(stand(X[9]).reshape(28,28), cmap='gray')
    axarr[0, 4].set_title('Node 5')
    axarr[1, 0].imshow(stand(X[11]).reshape(28,28), cmap='gray')
    axarr[1, 0].set_title('Node 6')
    axarr[1, 1].imshow(stand(X[13]).reshape(28,28), cmap='gray')
    axarr[1, 1].set_title('Node 7')
    axarr[1, 2].imshow(stand(X[15]).reshape(28,28), cmap='gray')
    axarr[1, 2].set_title('Node 8')
    axarr[1, 3].imshow(stand(X[17]).reshape(28,28), cmap='gray')
    axarr[1, 3].set_title('Node 9')
    axarr[1, 4].imshow(stand(X[19]).reshape(28,28), cmap='gray')
    axarr[1, 4].set_title('Node 10')
    plt.savefig("Problem_3.1_"+str(eta)+"_"+str(epochs)+"_weights.png", bbox_inches='tight')
    plt.close(fig)

def plot_single(data):
    """
    Plot a single digit
    """
    plt.imshow(data.reshape(28,28), cmap='gray')
    plt.title('Digit')
    plt.show()

def main():
    """
    Main function to run the procedure
    """
    # Import all the data and define some dimensionalities
    print("Reading in data...")
    with gzip.open("mnist.pkl.gz") as f:
        train_set, dev_set, test_set = pickle.load(f, encoding = 'bytes')
    Xtrain = train_set[0]
    train_label = train_set[1]
    Xdev = dev_set[0]
    dev_label = dev_set[1]
    Xtest = test_set[0]
    test_label = test_set[1]
    N_train = len(Xtrain)
    d = len(Xtrain[0])
    # Add bias to data
    Xtrain[:, 783] = 1
    Xdev[:, 783] = 1
    Xtest[:, 783] = 1
    # Create label matrices
    Ytrain = np.zeros(len(Xtrain)*10).reshape(len(Xtrain), 10)
    Ydev = np.zeros(len(Xdev)*10).reshape(len(Xdev), 10)
    Ytest = np.zeros(len(Xtest)*10).reshape(len(Xtest), 10)
    for i in range(len(Xtrain)):
        Ytrain[i, train_label[i]] = 1
    for i in range(len(Xdev)):
        Ydev[i, dev_label[i]] = 1
    for i in range(len(Xtest)):
        Ytest[i, test_label[i]] = 1
    print("...done")
    print()
    # Load model and initialize weigths and loss function
    model = MLPClassifier()
    weights_init(model)
    loss_func = nn.MSELoss()
    # Define further parameters
    epochs = 500
    m = 100
    eta = 0.1
    # Define optimizer as stochastic gradient descent
    optimizer = optim.SGD(model.parameters(), lr=eta, momentum=0.9)
    # Number of iterations per standard epoch
    iterations = int(N_train/m)
    # Create tensors for pytorch
    Xtrain_pt = Variable(FloatTensor(Xtrain), requires_grad=False)
    Ytrain_pt = Variable(FloatTensor(Ytrain), requires_grad=False)
    Xdev_pt = Variable(FloatTensor(Xdev), requires_grad=False)
    Ydev_pt = Variable(FloatTensor(Ydev), requires_grad=False)
    Xtest_pt = Variable(FloatTensor(Xtest), requires_grad=False)
    Ytest_pt = Variable(FloatTensor(Ytest), requires_grad=False)
    # Create Arrays for storing the errors
    error_train_arr = np.array(0)
    error_train_arr = np.delete(error_train_arr, 0)
    error_dev_arr = np.array(0)
    error_dev_arr = np.delete(error_dev_arr, 0)
    error_test_arr = np.array(0)
    error_test_arr = np.delete(error_test_arr, 0)
    misc_train_arr = np.array(0)
    misc_train_arr = np.delete(misc_train_arr, 0)
    misc_dev_arr = np.array(0)
    misc_dev_arr = np.delete(misc_dev_arr, 0)
    misc_test_arr = np.array(0)
    misc_test_arr = np.delete(misc_test_arr, 0)
    # Start training the neural network
    print("Starting training of MLP Classifier")
    print()
    for i in range(epochs):
        # For each epoch, shuffle Training set randomly but keep target value assignments
        temp = np.append(Xtrain, Ytrain, axis=1)
        temp_shuffled = np.random.shuffle(temp)
        Xtrain_shuffled = temp[:,:784]
        Ytrain_shuffled = temp[:,784:]
        # Start looping over the minibatches
        for j in range(iterations):
            # Create random minibatches
            #indices = np.random.randint(0, len(Xtrain), 200)
            #Ymini_pt = Variable(FloatTensor(Ytrain_shuffled[indices]), requires_grad=False)
            #Xmini_pt = Variable(FloatTensor(Xtrain_shuffled[indices]), requires_grad=False)
            # Create minibatches
            Xmini_pt = Variable(FloatTensor(Xtrain_shuffled[j*m:(j*m)+m, :]), requires_grad=False)
            Ymini_pt = Variable(FloatTensor(Ytrain_shuffled[j*m:(j*m)+m, :]), requires_grad=False)
            # Initialize optimizer
            optimizer.zero_grad()
            # Predict label from subset
            Y_hat_pt = model(Xmini_pt)
            # Compute error every half epoch
            if ( j == 0 or j == int(iterations/2)):
                misc_err_train = calc_miscl_err(Xtrain_pt.data.numpy(), Ytrain_pt.data.numpy(), model(Xtrain_pt).data.numpy())
                sq_err_train = calc_avg_sq_err(Xtrain_pt.data.numpy(), Ytrain_pt.data.numpy(), model(Xtrain_pt).data.numpy())
                misc_train_arr = np.append(misc_train_arr, misc_err_train)
                error_train_arr = np.append(error_train_arr, sq_err_train)
                misc_err_test = calc_miscl_err(Xtest_pt.data.numpy(), Ytest_pt.data.numpy(), model(Xtest_pt).data.numpy())
                sq_err_test = calc_avg_sq_err(Xtest_pt.data.numpy(), Ytest_pt.data.numpy(), model(Xtest_pt).data.numpy())
                misc_test_arr = np.append(misc_test_arr, misc_err_test)
                error_test_arr = np.append(error_test_arr, sq_err_test)
                misc_err_dev = calc_miscl_err(Xdev_pt.data.numpy(), Ydev_pt.data.numpy(), model(Xdev_pt).data.numpy())
                sq_err_dev = calc_avg_sq_err(Xdev_pt.data.numpy(), Ydev_pt.data.numpy(), model(Xdev_pt).data.numpy())
                misc_dev_arr = np.append(misc_dev_arr, misc_err_dev)
                error_dev_arr = np.append(error_dev_arr, sq_err_dev)
            # Calculate loss on minibatch
            loss = loss_func.forward(Y_hat_pt, Ymini_pt)
            # Perform backpropagation
            loss.backward()
            # Increase optimizer
            optimizer.step()
        # Print output for loop control
        print("Epoch: %i, Error: %3.2f, Miscl. rate: %3.2f%% (all on dev set)" % (i+1, sq_err_dev, misc_err_dev))

    #Print out lowest errors
    print()
    print("+++ Results: +++")
    print("Lowest avg. sq. loss on training set: %3.5f at it. step %i" % (np.amin(error_train_arr), np.argmin(error_train_arr)*0.5))
    print("Lowest avg. sq. loss on dev set: %3.5f at it. step %i" % (np.amin(error_dev_arr), np.argmin(error_dev_arr)*0.5))
    print("Lowest avg. sq. loss on test set: %3.5f at it. step %i" % (np.amin(error_test_arr), np.argmin(error_test_arr)*500))
    print("Lowest miscl. error rate on training set: %3.2f%% at it. step %i" % (np.amin(misc_train_arr), np.argmin(misc_train_arr)*0.5))
    print("Lowest miscl. error rate on dev set: %3.2f%% at it. step %i" % (np.amin(misc_dev_arr), np.argmin(misc_dev_arr)*0.5))
    print("Lowest miscl. error rate on test set: %3.2f%% at it. step %i" % (np.amin(misc_test_arr), np.argmin(misc_test_arr)*0.5))
    print("Smallest number of total mistakes on training set: %i/60000 at it. step %i" % (np.amin(misc_train_arr)*60000/100, np.argmin(misc_train_arr)*0.5))
    print("Smallest number of total mistakes on dev set: %i/10000 at it. step %i" % (np.amin(misc_dev_arr)*10000/100, np.argmin(misc_dev_arr)*0.5))
    print("Smallest number of total mistakes on test set: %i/10000 at it. step %i" % (np.amin(misc_test_arr)*10000/100, np.argmin(misc_test_arr)*0.5))
    print()
    # Extract some weights from input hidden_nodes
    model_list = list(model.parameters())
    model_weights = model_list[0].data.numpy()
    # Create plots
    x = np.arange(1,2*epochs+1,1, dtype='float')
    x /= 2
    fig = plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    #axes = plt.gca()
    #axes.set_ylim([0.01,0.1])
    plt.semilogy(x, error_train_arr, label = "Training error", linewidth=2)
    plt.semilogy(x, error_dev_arr, label = "Dev error", linewidth=2)
    plt.semilogy(x, error_test_arr, label = "Test error", linewidth=2)
    plt.grid(True)
    plt.xlabel("Standardized epoch")
    plt.ylabel("Averaged squared error")
    plt.legend()
    plt.subplot(1,2,2)
    axes = plt.gca()
    axes.set_ylim([0,90])
    plt.plot(x, misc_train_arr, label = "Training misclassification", linewidth=2)
    plt.plot(x, misc_dev_arr, label = "Dev misclassification", linewidth=2)
    plt.plot(x, misc_test_arr, label = "Test misclassification", linewidth=2)
    plt.grid(True)
    plt.xlabel("Standardized epoch")
    plt.ylabel("Misclassification rate in %")
    plt.legend()
    plt.savefig("Problem_3.1_"+str(eta)+"_"+str(epochs)+".png", bbox_inches='tight')
    plt.close(fig)
    # Plot weigths
    plot_10(model_weights, eta, epochs)




if __name__ == '__main__':
    main()
