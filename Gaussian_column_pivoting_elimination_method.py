import numpy as np

def GuassianColumnPivoting(A: np.ndarray, b: np.ndarray)->np.ndarray:
    """
    Use Gaussian column pivoting elimination method to solve A and b (Ax=b)

    Parameters:
    A(numpy.ndarray): Coefficient matrix A
    b(numpy.ndarray): constant vector b

    Returns:
    Ab(numpy.ndarray): Augmented matrix Ab(i.e. matrix[A, b])
    
    """
    Ab = np.hstack((A, b))
    n = np.shape(Ab)[0]
    # Number of iterations: n-1
    for k in range(n-1):
        index = np.argmax(abs(Ab[:, k]))
        C = np.eye(n)
        C[k, k] =  0
        C[k, index] = 1
        C[index, index] = 0
        C[index, k] = 1

        Ab = np.dot(C, Ab)
    
        L = np.eye(n)
        for m in range(k+1, n):
            l = Ab[m, k]/Ab[k, k]
            L[m ,k] = -l
        Ab = np.dot(L, Ab)
    
    return Ab


def SolveX(Ab: np.ndarray)->np.ndarray:
    """
    Solve for the solution x to the equation Ax = b with the augmented matrix Ab obtained
    through Gaussian column pivoting elimination method

    Parameters:
    Ab(numpy.ndarray): the augmented matrix obtained through Gaussian column pivoting elimination method

    Returns:
    xs(numpy.ndarray): the solution to the equation Ax = b
    """
    n = np.shape(Ab)[0]
    xs = np.zeros((np.shape(Ab)[0], 1))
 
    for i in range(np.shape(Ab)[0]):
        reA = Ab[n-1-i, : -1].reshape(1, -1)
        x = (Ab[n-1-i, -1] - np.dot(reA, xs))/Ab[n-1-i, n-1-i]
        xs[n-1-i, 0] = x[0, 0]
    
    return xs


if __name__ == '__main__':
    A = np.array([[1e-8, 2, 3], [-1, 3.712, 4.623], [-2, 1.072, 5.643]])
    b = np.array([1, 2 ,3]).reshape(3, -1)

    Ab = GuassianColumnPivoting(A, b)
    print(Ab)

    xs = SolveX(Ab)
    print(xs)
