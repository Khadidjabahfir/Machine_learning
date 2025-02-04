# import the library numpy to represent vectors as numpy arrays
import numpy as np 
# define the model function 
def linear_regression_model (x , W ,b ):
    # x is the vector of features 
    # W is the vector of parameters w 
    # b is a parameter represented as an integer 
    # the return is an integer f(x) 
    n = len(x) # n is the number of feature 
    f_X = 0.0 # the return value 
    for i in range (n) : 
        # looping over the feature 
        f_X += (x[i]*W[i]) 
    f_X += b 
    return f_X 
        
# here we will focus on testing the function 
# define x 
x = np.array([1,2,3])
# define w 
w = np.array ([4,5,6])
#define b 
b = 10 
# we will check for the equality of the size 
xn = len (x) 
wn = len (w) 
if xn == wn : 
    print (linear_regression_model(x,w,b)) 
else : 
    print ("the size of x and w has to be equal")
    
# define the model 
def linear_regression_model_with_vec (x , W , b )  : 
    # x is the vector of features (one example)
    # W is the vector of parameters w 
    # b is a parameter represented as an integer 
    # the return is an integer f(x) 
    f_X = 0.0 # the returned value 
    f_X += np.dot(x, W)
    f_X += b 
    return f_X 
    
    
# here we will focus on testing the function 
# define x 
x = np.array([1,2,3])
# define w 
w = np.array ([4,5,6])
#define b 
b = 10 
# we will check for the equality of the size 
xn = len (x) 
wn = len (w) 
if xn == wn : 
    print (linear_regression_model_with_vec(x,w,b)) 
else : 
    print ("the size of x and w has to be equal")



# define the cost function 
def squared_error_cost_fuction (X,Y,W,b ): 
    # X is the set of features in the training set 
    # Y is the set of outputs in the training set 
    # W is a vector of parameters 
    # b is a parameter 
    m = len(X) # the number of training example 
    diff = 0.0
    for i in range (m) : 
        f_X=linear_regression_model_with_vec(X[i],W ,b) # compute the value using the model 
        diff += (f_X - Y[i])**2 # compute the error for one training example 
    diff /= (2*m ) # compute the average 
    return diff 



# define X
X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])

# define Y 
Y = np.array([7.0, 8.0, 9.0])
# define W 
W = np.array ([1,2])
# define b 
b = 10 
# print the cost 
print (f'the cost is {squared_error_cost_fuction(X,Y,W,b) }')
