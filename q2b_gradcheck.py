import random

import numpy as np
from numpy.testing import assert_allclose


def gradcheck_naive(f, x, gradient_text=""):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         loss and its gradients
    x -- the point (numpy array) to check the gradient at
    gradient_text -- a string detailing some context about the gradient computation
    """
    #print(x)
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4         # Do not change this!

    #numgrad = np.zeros_like(x)
    #print(fx)
    #print(grad)
    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
           
        # Try modifying x[ix] with h defined above to compute numerical
        # gradients (numgrad).

        # Use the centered difference of the gradient.
        # It has smaller asymptotic error than forward / backward difference
        # methods. If you are curious, check out here:
        # https://math.stackexchange.com/questions/2326181/when-to-use-forward-or-central-difference-approximations

        # Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
        x[ix]+=h
        random.setstate(rndstate)
        plus = f(x)[0]

        x[ix]-=2*h
        random.setstate(rndstate)
        minus = f(x)[0]
        numgrad = (plus-minus)/(2*h)
        #print("now numgrad:")
        #print(numgrad)
        #raise NotImplementedError
        ### END YOUR CODE
        x[ix]+=h
        # Compare gradients
        
        assert_allclose(numgrad, grad[ix], rtol=1e-5,
                        err_msg=f"Gradient check failed for {gradient_text}.\n"
                                f"First gradient error found at index {ix} in the vector of gradients\n"
                                f"Your gradient: {grad[ix]} \t Numerical gradient: {numgrad}")

        it.iternext()  # Step to next dimension

    print("Gradient check passed!")


def test_gradcheck_basic():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), 2*x)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))       # scalar test
    gradcheck_naive(quad, np.random.randn(3,))     # 1-D test
    gradcheck_naive(quad, np.random.randn(4, 5))   # 2-D test
    print()


def your_gradcheck_test():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR OPTIONAL CODE HERE
    custom_scalar_loss = lambda x: (x**3, 3 * x**2)
    gradcheck_naive(custom_scalar_loss, np.array(2.0), "custom scalar loss")

    # Test 2: Custom loss function with a 1-D input
    custom_1d_loss = lambda x: (np.sum(x ** 3), 3 * x ** 2)
    gradcheck_naive(custom_1d_loss, np.random.randn(5), "custom 1-D loss")

    # Test 3: Custom loss function with a 2-D input
    custom_2d_loss = lambda x: (np.sum(x ** 3), 3 * x ** 2)
    gradcheck_naive(custom_2d_loss, np.random.randn(3, 4), "custom 2-D loss")

    ### END YOUR CODE



if __name__ == "__main__":
    test_gradcheck_basic()
    your_gradcheck_test()
