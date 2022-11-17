import numpy as np


def gradient(f, x, delta=1e-5):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method


    Returns:
        ret (numpy.array): gradient of f at the point x
    """
    n, = x.shape
    gradient = np.zeros(n).astype('float64')
    # finite difference method f(x) = f(x + delta) + f(x - delta) / 2delta
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += delta
        x_minus = x.copy()
        x_minus[i] -= delta
        gradient[i] = (f(x_plus) - f(x_minus)) / (2 * delta)

    return gradient


def jacobian(f, x, delta=1e-5):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    n, = x.shape
    m, = f(x).shape
    x = x.astype('float64') #Need to ensure dtype=np.float64 and also copy input. 
    gradient = np.zeros((m, n)).astype('float64')

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += delta
        x_minus = x.copy()
        x_minus[i] -= delta
        gradient[:, i] = (f(x_plus) - f(x_minus)) / (2 * delta)
    
    return gradient



def hessian(f, x, delta=1e-5):
    """
    Returns the Hessian of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (x.shape[0], x.shape[0])
                            which is the Hessian of f at the point x
    """
    n, = x.shape
    x = x.astype('float64') #Need to ensure dtype=np.float64 and also copy input.
    hessian = np.zeros((n, n)).astype('float64')

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += delta
        x_minus = x.copy()
        x_minus[i] -= delta
        hessian[:, i] = (gradient(f, x_plus) - gradient(f, x_minus)) / (2 * delta)

    return hessian
    


