import sys
from optparse import OptionParser
import numpy as np
import time
from pprint import pprint # library for print formatting


### NOTES ###
# Bias term is added to the training data in the __main__
#############


# l is an array
def mean(l): 
    return sum(l)/len(l)


# Cross Validation function that returns a list of training and 
# testing arrays that have been sliced appropriately
def cv(X, Y, k):
    folds = np.array_split(np.arange(X.shape[0]), k)
    sets = []

    for f in folds:
        test_x = X[f]
        test_y = Y[f]

        folds_true = np.ones(X.shape[0], bool)
        folds_true[f] = False

        train_x = X[folds_true]
        train_y = Y[folds_true]

        sets += [(train_x, train_y, test_x, test_y)]

    return sets


def ols(X_train, y_train):
    t0 = time.time()
    folds = 5
    mse_cv = []
    weights = []
    sets = cv(X_train, y_train, folds)

    for (X_train, y_train, X_test, y_test) in sets:
        # Calculate weights based on closed form solution to OLS
        w = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        # Add the weights to the weights array for reporting
        weights += [w]
        # Calculate MSE
        mse = np.square(np.subtract(X_test @ w, y_test)).mean()
        # Add MSE to array to calculate average below with CV
        mse_cv += [mse]

    # Calculate the average MSE as found by CV
    cv_mean_squared_error = mean(mse_cv)

    # Pick one of the weights and use it to report most relevant features
    weight_picked = weights[0].tolist()
    most_relevant = weight_picked.index(max(weight_picked))
    print("The most relevant feature was found to be feature", most_relevant)

    """ You should report the parameters selected """
    print("Parameters (weights): ")
    pprint(weights)

    """ You should report the estimated MSE from CV """
    print("Method finished in {:.3f} s:".format(time.time() - t0))
    print("Estimated generalization error (MSE):",\
          cv_mean_squared_error)


def ols_sgd(X_train, y_train):
    t0 = time.time()
    """ You should report the parameters selected """

    """ You should report the estimated MSE from CV """
    cv_mean_squared_error = np.inf


    print("Method finished in {:.3f} s:".format(time.time() - t0))
    print("Estimated generalization error (MSE):",\
          cv_mean_squared_error)


def ridge(X_train, y_train):
    t0 = time.time()
    lambda_values = [0.01, 0.02, 0.05, 0.1, 1]
    folds = 5
    sets = cv(X_train, y_train, folds)
    mse_cv = []
    mse_best = np.inf

    for l in lambda_values:
        for (X_train, y_train, X_test, y_test) in sets:
            # Solve according to the closed form of the primal ridge optimization problem
            w = np.linalg.pinv(X_train.T @ X_train + l*np.eye(X_train.shape[1])) @ X_train.T  @ y_train
            # Calculate MSE
            mse = np.square(np.subtract(X_test @ w, y_test)).mean()
            # Add MSE to array to calculate average below with CV
            mse_cv += [mse]
        
        # Set the average MSE found through CV for the
        # lambda that is currently chosen
        mse_current_l = mean(mse_cv)

        # Check if this lambda returned a better average MSE
        if mse_current_l < mse_best:
            mse_best = mse_current_l
            weights_best = w
            lambda_best = l

        # ResetMSE array for the next lambda iteration
        mse_cv = []

    # Report the most relevant feature
    weight_picked = weights_best.tolist()
    most_relevant = weight_picked.index(max(weight_picked))
    print("The most relevant feature was found to be feature", most_relevant)

    """ You should report the parameters selected """
    print("Parameters: ")
    print("Weights:")
    pprint(weights_best)
    print("Lambdas: ", lambda_values)
    print("The lambda with the lowest average MSE: ", lambda_best)

    """ You should report the estimated MSE from CV """
    print("Method finished in {:.3f} s:".format(time.time() - t0))
    print("Estimated generalization error (MSE):",\
          mse_best)


def lasso(X_train, y_train):
    t0 = time.time()
    lambda_values = [0.01, 0.02, 0.05, 0.1, 1]
    folds = 5
    sets = cv(X_train, y_train, folds)
    mse_cv = []
    mse_best = np.inf

    for l in lambda_values:
        for (X_train, y_train, X_test, y_test) in sets:
            # Solve according to the closed form of the primal lasso optimization problem
            B = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
            w = B * max(0, 1 - ((X_train.shape[0]*l)/(B.shape[0])))
            # Calculate MSE
            mse = np.square(np.subtract(X_test @ w, y_test)).mean()
            # Add MSE to array to calculate average below with CV
            mse_cv += [mse]
        
        # Set the average MSE found through CV for the
        # lambda that is currently chosen
        mse_current_l = mean(mse_cv)

        # Check if this lambda returned a better average MSE
        if mse_current_l < mse_best:
            mse_best = mse_current_l
            weights_best = w
            lambda_best = l

        # Reset weights and MSE arrays for the next lambda iteration
        mse_cv = []

    # Report the most relevant feature
    weight_picked = weights_best.tolist()
    most_relevant = weight_picked.index(max(weight_picked))
    print("The most relevant feature was found to be feature", most_relevant)

    """ You should report the parameters selected """
    print("Parameters: ")
    print("Weights:")
    pprint(weights_best)
    print("Lambdas: ", lambda_values)
    print("The lambda with the lowest average MSE: ", lambda_best)

    """ You should report the estimated MSE from CV """
    print("Method finished in {:.3f} s:".format(time.time() - t0))
    print("Estimated generalization error (MSE):",\
          mse_best)


def ridge_nonlin(X_train, y_train):
    t0 = time.time()
    """ You should report the parameters selected """

    """ You should report the estimated MSE from CV """
    cv_mean_squared_error = np.inf


    print("Method finished in {:.3f} s:".format(time.time() - t0))
    print("Estimated generalization error (MSE):",\
          cv_mean_squared_error)

    
def kernelridge(X_train, y_train):
    t0 = time.time()
    """ You should report the parameters selected """

    """ You should report the estimated MSE from CV """
    cv_mean_squared_error = np.inf

    print("Method finished in {:.3f} s:".format(time.time() - t0))
    print("Estimated generalization error (MSE):",\
          cv_mean_squared_error)


def nonparametric(X_train, y_train):
    t0 = time.time()
    """ You should report the parameters selected """

    """ You should report the estimated MSE from CV """
    cv_mean_squared_error = np.inf

    print("Method finished in {:.3f} s:".format(time.time() - t0))
    print("Estimated generalization error (MSE):",\
          cv_mean_squared_error)


def default(str):
    return str + ' [Default: %default]'




### MAIN ###
if __name__ == "__main__":

    usageStr = """
    USAGE:      python hw3.py <options>
    EXAMPLES:   (1) python hw3.py -m ols x.csv y.csv
                    - Use OLS with the given dataset: x.csv features, y.csv target values
    """

    parser = OptionParser(usageStr)

    parser.add_option('-m', '--method',  dest='method',
                      help=default('Regression method to use'),
                      default="ols")


    options, args = parser.parse_args(sys.argv[1:])
    if options.method not in ['ols', 'ols_sgd', 'ridge', 'lasso', 'ridge_nonlin', 'kernelridge', 'nonparametric']:
        raise Exception("Invalid method {}".format(options.method))
    if len(args) != 2:
        raise Exception('Command line input not understood: ' \
                        + str(args) + " Expected two input files")

    X_train = np.genfromtxt(args[0])
    ### Adding bias ###
    b = np.ones((X_train.shape [0], 1))
    X_train = np.append(b, X_train, axis = 1)
    ###################
    y_train = np.genfromtxt(args[1])

    print("Loaded training set: {} data points, {} attributes, {} target values".
          format(X_train.shape[0], X_train.shape[1], y_train.shape[0]))

    if options.method == 'ols':
        ols(X_train, y_train)
    elif options.method == 'ols_sgd':
        ols_sgd(X_train, y_train)
    elif options.method == 'ridge':
        ridge(X_train, y_train)
    elif options.method == 'lasso':
        lasso(X_train, y_train)
    elif options.method == 'ridge_nonlin':
        ridge_nonlin(X_train, y_train)
    elif options.method == 'kernelridge':
        kernelridge(X_train, y_train)
    else:
        nonparametric(X_train, y_train)
