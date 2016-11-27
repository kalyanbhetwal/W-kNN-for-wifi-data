from pylab import *
import pylab
import random
from math import *
import pyGPs
import numpy as np
import random
import math
import csv
import sys 
import os
import socket
import operator
import interpolate
import parser

sys.path.append(os.path.abspath("D:\knn\initial_pos.py"))

#f = open("sampleText.txt","w")
#Sequentail Monte-Carlo Estimation
#Here a particle filter for tracking people in a envirionemnt is implemented
#at first particle are distributed through out world randomly 
#calculating WkNN for intial sample of RSSI converge particle
#this gives initial weight and position
#now using motion model propagate the particles 
#here is little problem
#while propating at fist since we don't have past exprience
#we have propagate in every direction posible
#now incooporate the sensor measurement for each particle at given positions
#assign new weight to the paticles on the basis of GPR result and MEMS data
#re-sampling which is the most important step in particle 
#filtering should be done carefully
#particles which have greater weight than other are more frequently sampled
#some particles die which have very less probability
#new position is estimated calulating average based on weight
#in next step is to propagate particle based on previous exprience
#now continue
world_size = 30

def loadDataset(filename):
        trainingSet=[]
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(1,len(dataset)):
	        for y in range(9):
	            dataset[x][y] = float(dataset[x][y])

	        trainingSet.append(dataset[x])
	return trainingSet
	

trainingSet = loadDataset('rssi.csv')



X=[]
y1=[]
y2=[]
y3=[]
y4=[]
y5=[]
y6=[]
y0=[]
for i in range(len(trainingSet)):
	temp=[trainingSet[i][7],trainingSet[i][8]]
	#temp =[trainingSet[i][7]]
	X.append(temp)
	temp =[trainingSet[i][0]]
	y0.append(temp)
	temp =[trainingSet[i][1]]
	y1.append(temp)
	temp =[trainingSet[i][2]]
	y2.append(temp)
	temp =[trainingSet[i][3]]
	y3.append(temp)
	temp =[trainingSet[i][4]]
	y4.append(temp)
	temp =[trainingSet[i][5]]
	y5.append(temp)
	temp =[trainingSet[i][6]]
	y6.append(temp)

x = np.array(X)
y0 = np.array(y0)
y1 = np.array(y1)
y2 = np.array(y2)

#m = pyGPs.mean.Zero()
#k = pyGPs.cov.RBFard(log_ell_list=[0.05,0.17], log_sigma=1.)
#model.setPrior(mean=m, kernel=k)
#model.setNoise( log_sigma = np.log(0.1) )

model1 = pyGPs.GPR()           # model
model1.setData(x,y0)

model2 = pyGPs.GPR()
model2.setData(x,y1)


model3 = pyGPs.GPR()
model3.setData(x,y2)

host = ''
port = 8221
address = (host, port)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(address)
server_socket.listen(5)

print "Listening for client . . ."
conn, address = server_socket.accept()
print "Connected to client at ", address
    
class human_pos:
    #Create 1000 random particles
    def __init__(self):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
       
    
    #provide intial position, orientation based on WKNN , also assign weight
    def set(self, new_x, new_y, new_orientation):
        if new_x < 0 or new_x >= world_size:
            raise ValueError, 'X coordinate out of bound'
        if new_y < 0 or new_y >= world_size:
            raise ValueError, 'Y coordinate out of bound'
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError, 'Orientation must be in [0..2pi]'
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
    
    
    
    #movement of people in enviroment using motion dynamic model
    def motionmodel(self, turn, forward):
      # move particles based upon previous movement
      #random
        orientation = abs(self.orientation + float(turn) + random.gauss(0,0.05))       #error in angle measurement by magnetometer
        orientation %= 2 * pi
          
        # move, and add randomness to the motion command
        dist = abs(float(forward) + random.gauss(0.0,0.2))                         # error on
        x = self.x + (cos(orientation) * dist)
        y = self.y + (sin(orientation) * dist)
        x %= world_size    # cyclic truncate
        y %= world_size
        
        # set particle
        res = human_pos()
        res.set(x, y, orientation)
        #res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res

    def my_fun(self):
        my_p.append([self.x,self.y])


    def ret_avg(self):
        x.append(self.x)
        y.append(self.y)
        #f.write(str(self.x)+','+str(self.y)+','+str(self.orientation)+'\n')

    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))
      
def Gaussian(mu, sigma, x):
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ) / 2.0) / sqrt(2.0 * pi * (sigma ))
    
    
def measurement_prob(p, measurement):
        
        # calculates how likely a measurement is
        #call GPR from matlab production server to get mean
        #and variance for likelihood estimation
        #after that call gaussian
        #from here feed data into production server and get reply from there
        #based upon reply update weight of particles
        #also incooporate motion MEMS sensor measurement
        #input (x,y) coordinates get mean and variance
        # call seosnor get data
        # call 
        #a = kalyan()
        #print a

        #print self.x ,self.y
        #z = np.array([[self.x,self.y]])
        #model1.predict(z)
        
        ym, ys2, fmu, fs2, lp = model1.predict(p)       

        ym1,ys21,fmu,fs2,lp = model2.predict(p) 

        ym2,ys23,fmu,fs2,lp = model3.predict(p) 
        w = []
        for i in range(len(ym)):
            prob1 = Gaussian(ym[i], ys2[i] , measurement[0])
            prob2 =Gaussian(ym1[i], ys21[i] , measurement[1])
            prob3 =Gaussian(ym2[i],ys23[i],measurement[2])

            prob = (prob1*prob2*prob3)**0.33334
            w.append(prob)
        return w
        
        '''
        prob = self.Gaussian(ym, ys2 , -76)
        prob1 = self.Gaussian(ym1 , ys21 , -80)
        prob2 = self.Gaussian(ym2,ys23,-70)
        #print p1,p2,p3
        #print self.x ,' ' ,self.y ,prob ,' ', prob1,' ', prob2

        return (prob*prob1*prob2)**0.333334
        '''
    

#call WkNN algorithm
def sqmcl():
    p=[]
    N= 1000
    '''
    for i in range(N):
        x=human_pos()
        p.append(x)
    '''
    output = conn.recv(2048)
    values = parser.firstParser(output)
    strength = values[0]

    var1,var2 = initialCor(trainingSet,values[0])  

    #print var1,var2
     
    robbie = human_pos()
    po=[]

    for o in range(N):
            par = human_pos()
            circle_x = var1
            circle_y = var2
            circle_r = 2
            # random angle
            alpha = 2 * math.pi * random.random()
            # random radius
            r = circle_r * random.random()
            # calculating cooringates
            x = abs(r * math.cos(alpha) + circle_x )
            y = abs(r * math.sin(alpha) + circle_y )
            orientation = random.random() * 2.0 * pi
            #orien = random.random()
            
            par.set(x,y,alpha)
            #par.set_noise(0.5, 0.5, 5.0)
            po.append(par)

    p =po

    #print po
    #robbie.motionmodel(10,2) # deg and rad

    while True:
        output = conn.recv(2048)
        if output.strip() == "disconnect":
            conn.close()
            sys.exit("Received disconnect message.  Shutting down.")
            conn.send("dack")
            
        elif output:
            print output
            values = parser.firstParser(output)
            strength = value[0]
            orientation = values[1]
            totalSteps = values[2]

            p2 = []
            for i in range(N):
                p2.append(p[i].motionmodel(orientation,2))  # the orientation value chai normalize garna baaki cha
            p = p2
            # define new list that calls for the fucntion that gets rssi value

            w = []

            my_p =[]

            Z = strength
            
            for i in range(N):
                p[i].my_fun()

            w = measurement_prob(np.array(my_p),Z)

            #for i in range(N):
              # w.append(p[i].measurement_prob(Z))        
            p3 = []
               
            # random starting particle index
            index = int(random.random() * N)

            # beta
            b = 0
            w_max = max(w)
            for i in range(N):
                b += random.random() * 2.0 * w_max
                while b > w[index]:
                    b = b - w[index]
                    index = (index + 1) % N
                p3.append(p[index])
                
            p = p3

            xa = 0
            ya = 0

            for j in range(N):
                xa = xa + p[j].x
                ya = ya + p[j].y

                
            xa = xa/N
            ya = ya/N

            print xa,ya

            location = interpolate.interpolate(xa, ya)
            f = open('db.json', 'wb')
            f.write('[{"geometry": {"type": "Point", "coordinates": [' + str(location[0]) +
                    ',' + str(location[1]) + ']}, "type": "Feature", "properties": {}}]')
            f.close()

            '''

            for i in range(N):
                p[i].ret_avg()

            x_avg= sum(x)/N
            y_avg= sum(y)/N


            plt.figure('Robot in the world')
            plt.title('Particle filter')
            grid = [0, world_size, 0, world_size]
            plt.axis(grid)
            circle = plt.Circle((x_avg,y_avg), 1./5, facecolor='#66ffff', edgecolor='#009911', alpha=1)
            plt.gca().add_patch(circle)
            plt.plot(x,y,'*')
            print x_avg,y_avg
            #plt.show()


            '''

if __name__ == "__main__":
    sqmcl()