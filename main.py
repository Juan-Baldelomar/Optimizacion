import matplotlib.pyplot as plt
import  random
import numpy as np
import Optimizacion as op
import Function as TestFunction
import ReadMnist as loader


# Function to generate ti vs yi plots
def graph_ti():
    n = 128
    func = TestFunction.f(n, 0, sigma=1)
    v = np.random.normal(func.y, 0.5, (n,))
    r = op.steepest_descent_bisection(v, func, tol=0.00001, c1=0.0001, c2=0.9, debug=True)

    vec_ti_yi = func.ti_vs_yi()
    vec_ti_xi = []
    for i in range(n):
        vec_ti_xi.append((vec_ti_yi[i][0], r[0][i]))

    plt.plot(*zip(*vec_ti_yi))
    plt.plot(*zip(*vec_ti_xi), c='green')

    proxy = [plt.Rectangle((0, 0), 1, 1, fc='skyblue'),
             plt.Rectangle((0, 0), 1, 1, fc='green')]
    plt.legend(proxy, ["ti vs yi", "ti vs xi"])
    plt.show()


# Function to get statistics
def stats():
    return


def stats_2():
    return

# print result functions
def printResults(tol, c1, c2, r, f:TestFunction):
    print("Tolerancia : {}, c1 = {}, c2 = {}".format(tol, c1, c2))
    print("iterations: ", r[1])
    print("Min x* is : \n", r[0])
    #print("f(x*) = ", f.function(r[0]))
    print("||grad(x*)|| = ", np.linalg.norm(f.gradient(r[0])))


def printResultsRho(tol, rho, c1, r, f:TestFunction):
    print("Tolerancia : {}, rho = {}, c1 = {}".format(tol, rho, c1))
    print("iterations: ", r[1])
    print("Min x* is : \n", r[0])
    print("f(x*) = ", f.function(r[0]))
    print("||grad(x*)|| = ", np.linalg.norm(f.gradient(r[0])))

    # Generate Graphics
    plt.plot(*zip(*r[2]))
    plt.show()
    plt.plot(*zip(*r[3]), c='green')
    plt.show()


def error(test_set, beta):
    n = len(test_set[0])
    x = test_set[0]
    y = test_set[1]
    func = TestFunction.h(785, x, y)
    sum = 0
    for i in range(n):
        if y[i] != 0 and y[i] != 1:
            continue

        pi = func.pi2(beta, beta[0], i)
        print(pi, " ", y[i])
        if pi > 0.5:
            sum +=  abs(pi - y[i])
        else:
            sum+= abs(0 - y[i])

    print("Error: ", sum/n)


def mnist():
    train_set, val_set, test_set = loader.loadData()
    #x= []; y = []
    #for i in range(10):
    #    x.append((train_set[0][i][407], train_set[0][i][408]));
    #    y.append(train_set[1][i]/10);
    # x = np.array(x)
    # y = np.array(y)

    x = []
    y = []
    for i in range(len(train_set[1])):
        if train_set[1][i] == 1 or train_set[1][i] == 0:
            x.append(train_set[0][i])
            y.append(train_set[1][i])

    x = np.array(x)
    y = np.array(y)

    func = TestFunction.h(785, x, y)
    v = np.random.random(size=(785,))
    v[0] = 0
    r = op.steepest_descent_bisection(v, func, tol=0.0001, c1=0.0001, c2=0.9, max_it=100, debug=True)
    printResults(0.0001, 0.0001, 0.9, r, func)
    error(test_set, r[0])



# PRUEBA
random.seed(0)
np.random.seed(0)
#main()

mnist()
