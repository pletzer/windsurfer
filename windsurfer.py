from scipy.optimize import root
from math import cos, sin, atan2, sqrt
import numpy
from matplotlib import pylab

class Windsurfer:

    def __init__(self):
        self.boomHeight = 1.60 # metre
        self.halfHarnessLength = 0.50
        self.halfBoardWidth = 0.0 #0.10
        self.legLength = 1.30
        self.alpha = None
        self.beta = None

        self.sailForceHeight = 2.0

        self.windsurferWeight = 1.0 # Newtons
        self.sailForce = 0.4


    def computeAngles(self):
        """
        Compute the angles
        """

        # compute the angle between the horizontal and the harness lines
        def func(x, *params):
            return self.legLength**2 - \
                   (self.halfHarnessLength*cos(x[0]) - self.halfBoardWidth)**2 - \
                   (self.boomHeight - self.halfHarnessLength*sin(x[0]))**2

        x0 = numpy.array([0.])
        sol = root(func, x0, tol=1.e-3)
        self.alpha = sol.x[0]
        
        # compute the angle between windusurfer legs and the horizontal 
        self.beta = atan2(self.boomHeight - self.halfHarnessLength*sin(self.alpha), 
                          self.halfHarnessLength*cos(self.alpha) - self.halfBoardWidth)

        # check 
        errorX = self.halfBoardWidth + self.legLength*cos(self.beta) - self.halfHarnessLength*cos(self.alpha)
        errorY = self.legLength*sin(self.beta) + self.halfHarnessLength*sin(self.alpha) - self.boomHeight
        assert(max(abs(errorX), abs(errorY)) < 1.e-4)
        #print('angle errorX = {} errorY = {}'.format(errorX, errorY))

    def getLegForce(self):
        fx = self.sailForce * (self.sailForceHeight/self.boomHeight)
        fy = fx/cos(self.alpha) - self.windsurferWeight
        return numpy.array([fx, fy])



################################################################################

def main():
    wsrf = Windsurfer()

    numHarnessLines = 6
    numBooms = 21
    clrs = ['b', 'g', 'r', 'c', 'm', 'k']

    halfHarnessLengths = numpy.linspace(0.3, 0.6, numHarnessLines)
    for i in range(numHarnessLines):

        wsrf.halfHarnessLength = halfHarnessLengths[i]
        maxBoomHeight = wsrf.legLength + wsrf.halfHarnessLength
        boomHeights = numpy.linspace(1.30, 0.9*maxBoomHeight, numBooms)
        forceLegs = numpy.zeros((numBooms,), numpy.float64)

        for j in range(numBooms):

            wsrf.boomHeight = boomHeights[j]

            wsrf.computeAngles()

            forceLegVec = wsrf.getLegForce()
            forceLegs[j] = sqrt(forceLegVec.dot(forceLegVec))

        pylab.plot(boomHeights, forceLegs, clrs[i] + '-')

    #pylab.axes([1.3, 1.8, 0., wsrf.windsurferWeight])
    pylab.legend(['hrnss lgth/2=' + str(hl) for hl in halfHarnessLengths])
    pylab.xlabel('boom height [m]')
    pylab.ylabel('leg force relative to weight')

    pylab.show()

if __name__ == '__main__': main()

