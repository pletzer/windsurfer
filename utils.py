import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def equations(vars, el, lam, h):
    """
    
    C
    +
    | \
    |  \
    |   o B
    |  /
    | /
    |/
    +
    A
    
    
    AC = h
    AB = el
    BC = lam
    
    alpha: < AB, AC
    beta: < CA, CB
    """
    alpha, beta = vars
    eq1 = el * np.sin(alpha) - lam * np.sin(beta)
    eq2 = el * np.cos(alpha) + lam * np.cos(beta) - h
    return [eq1, eq2]

def visualize_equations(el, lam, h):
    alphas = np.linspace(0.01, np.pi/2, 101)
    betas  = np.linspace(0.01, np.pi/2, 102)
    aa, bb = np.meshgrid(alphas, betas)
    cost = (el * np.sin(aa) - lam * np.sin(bb))**2 + (el * np.cos(aa) + lam * np.cos(bb) - h)**2
    plt.pcolor(alphas*180/np.pi, betas*180/np.pi, cost)
    plt.contour(alphas*180/np.pi, betas*180/np.pi, cost, levels=[0.0001,0.001])
    #plt.colorbar()
    plt.xlabel('alpha deg')
    plt.ylabel('beta deg')
    plt.title('cost (0 = solution)')
    plt.show()
    
def visualize_solution(el, lam, h, alpha, beta):
    a = np.array([0.,0.])
    b = el*np.array([np.sin(alpha), np.cos(alpha)])
    c = np.array([0, h])
    # b2 should match b
    b2 = c + lam*np.array([np.sin(beta), -np.cos(beta)])
    plt.arrow(a[0], a[1], b[0] - a[0], b[1] - a[1], color='red')
    plt.arrow(c[0], c[1], b2[0] - c[0], b2[1] - c[1], color='blue')
    plt.axis('equal')
    plt.title(f'alpha = {alpha*180/np.pi:.2f} deg beta = {beta*180/np.pi:.2f} deg')
    
    plt.show()

def alpha_beta(el, lam, h):
    """
    Find alpha and beta given 
    :param el: length of windurfer's legs
    :param lam: half harness line length
    :param h: boom height
    """

    initial_guess = [np.pi*30/180, np.pi*45/180] # Initial guess for alpha and beta
    solution = fsolve(equations, initial_guess, args=(el, lam, h))
    alpha, beta = solution
    return alpha, beta



###############################################################################
def main():
    el = 1.2 # length of leg
    lam = 0.7 # half length of harness lines
    h = 1.6 # boom height
    alpha, beta = alpha_beta(el, lam, h)
    print(f'alpha = {alpha} beta = {beta}')
    visualize_solution(el, lam, h, alpha, beta)
    visualize_equations(el, lam, h)
    
if __name__ == '__main__': main()
    