
import numpy as np
import Function as TestFunction


# xk : vector in Rn
# dk: direction vector
# f: Function to evaluate
def BackTracking(xk, dk, f:TestFunction, rho=0.5, c1=0.0001):
    alpha = 1
    #alpha = random.randint(1, 10)
    while f.function(xk + alpha*dk) > (f.function(xk) + (c1 * alpha) * (np.dot(f.gradient(xk), dk))):
        alpha = rho * alpha

    return alpha


def steepest_descent(x0, f:TestFunction, tol=0.0001, rho=0.5, c1=0.0001, debug=False):
    plotPoints = []
    plotGrad = []

    # init values
    k = 0;
    xk = x0
    dk = -f.gradient(x0)

    while np.linalg.norm(dk) >= tol:
        alpha = BackTracking(xk, dk, f, rho, c1)

        # Add points to be plotted
        plotPoints.append((k, f.function(xk)))
        plotGrad.append((k, np.linalg.norm(f.gradient(xk))))

        # steepest descent core
        xk = xk + alpha * dk
        dk = -f.gradient(xk)
        k = k + 1

        # debug option
        if debug: print(xk)

    return xk, k, plotPoints, plotGrad


def bisection(xk, dk, f:TestFunction, c1=0.0001, c2=0.9, tol=0.0001):
    alpha = 0
    alpha_i = 1
    inf = np.inf
    beta = inf

    while True:
        # first 2 conditions are to avoid infinite loop and too small alphas
        if alpha_i < tol:
            break
        elif abs(alpha - beta) < 0.000000000001:
            break
        elif f.function(xk + alpha_i * dk) > f.function(xk) + (c1 * alpha_i) * np.dot(f.gradient(xk), dk):
            beta = alpha_i
            alpha_i = 0.5 * (alpha + beta)
        elif np.dot(f.gradient(xk + alpha_i*dk), dk) < c2 * np.dot(f.gradient(xk), dk):
            alpha = alpha_i
            alpha_i = 2. * alpha if (beta == inf) else 0.5 * (alpha + beta)
        else:
            break

    return alpha_i


def steepest_descent_bisection(x0, f:TestFunction, tol=0.00001, c1=0.0001, c2=0.9, max_it = 20000, debug=False):
    # init values
    k = 0; xk = x0
    dk = f.gradient(x0)
    norm = np.linalg.norm(dk)
    while norm >= tol and k < max_it:
        #alpha = bisection(xk, dk, f, c1, c2)
        alpha = 0.05

        # steepest descent core
        xk = xk + alpha * dk
        dk = f.gradient(xk)
        norm = np.linalg.norm(dk)
        k = k + 1

        # debug option
        if debug:
            print("it: ", k)
            #print(xk)
            #print(norm)

    return xk, k