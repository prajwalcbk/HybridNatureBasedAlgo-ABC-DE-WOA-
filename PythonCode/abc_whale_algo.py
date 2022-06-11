
import numpy as np
import random, math, copy
from data import Threading,trace
from data import color,endc,pp
from termcolor import colored

def GrieFunc(data): 
    s1 = 0.
    s2 = 1.
    for k, x in enumerate(data):
        s1 = s1 + x ** 2
        s2 = s2 * math.cos(x/math.sqrt(k+1))
    y = (1./4000.) * s1-s2 + 1
    return 1./(1. + y)

def RastFunc(data):
    """
    Rastrigin function
    """
    s = 10 * 25
    for i in data:
        s = s + i ** 2 - 10 * math.cos(2 * math.pi * i)
    return 1./(1.+s)

def Sphere(x):
    ss=sum(np.power(x, 2))
    return ss

class ABCIndividual:
    def __init__(self, bound):
        self.score = 0.
        self.invalidCount = 0 
        self.chrom = [random.uniform(a,b) for a,b in zip(bound[0,:],bound[1,:])] 
        self.calculateFitness()        

    def calculateFitness(self):
        self.score = GrieFunc(self.chrom)

class ArtificialBeeColony:
    def __init__(self, foodCount, onlookerCount, bound, maxIterCount=1000, maxInvalidCount=200):
        self.foodCount = foodCount                 
        self.onlookerCount = onlookerCount           
        self.bound = bound                        
        self.maxIterCount = maxIterCount           
        self.maxInvalidCount = maxInvalidCount    
        self.foodList = [ABCIndividual(self.bound) for k in range(self.foodCount)]
        self.foodScore = [d.score for d in self.foodList] 
        self.bestFood = copy.copy(self.foodList[np.argmax(self.foodScore)] )  
        self.rnd = random.Random(0)
        

    def updateFood(self, i):                                                  
        k = random.randint(0, self.bound.shape[1] - 1) 
        j = random.choice([d for d in range(self.foodCount) if d !=i])  
        vi = copy.deepcopy(self.foodList[i])
        vi.chrom[k] += random.uniform(-1.0, 1.0) * (vi.chrom[k] - self.foodList[j].chrom[k])
        vi.chrom[k] = np.clip(vi.chrom[k], self.bound[0, k], self.bound[1, k])               
        vi.calculateFitness()
        
        if vi.score > self.foodList[i].score:           
            self.foodList[i] = vi
            if vi.score > self.foodScore[i]:            
                self.foodScore[i] = vi.score
                if vi.score > self.bestFood.score:     
                    self.bestFood = copy.copy(vi)
            self.foodList[i].invalidCount = 0
        else:
            self.foodList[i].invalidCount += 1
            
    def employedBeePhase(self):
        for i in range(0, self.foodCount):             
            self.updateFood(i)
            
    def woa(self,i,n,a,a2):
        A = 2 * a * self.rnd.random() - a
        C = 2 * self.rnd.random()
        b = 1
        l = (a2-1)*self.rnd.random()+1
        p = self.rnd.random()
        dim=len(self.bound[0])
        D = [0.0 for i in range(dim)]
        D1 = [0.0 for i in range(dim)]
        Xnew = [0.0 for i in range(dim)]
        Xrand = [0.0 for i in range(dim)]
        if p < 0.5:
            if abs(A) > 1:
                for j in range(dim):
                    D[j] = abs(C * self.bestFood.chrom[j] - self.foodList[i].chrom[j])
                    Xnew[j] = self.bestFood.chrom[j] - A * D[j]
            else:
                p = random.randint(0, n - 1)
                while (p == i):
                    p = random.randint(0, n - 1)
                Xrand = self.foodList[p].chrom

                for j in range(dim):
                    D[j] = abs(C * Xrand[j] - self.foodList[i].chrom[j])
                    Xnew[j] = Xrand[j] - A * D[j]
        else:
            for j in range(dim):
                D1[j] = abs(self.bestFood.chrom[j] - self.foodList[i].chrom[j])
                Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l) + self.bestFood.chrom[j]
        
        vi = copy.deepcopy(self.foodList[i])
        vi.chrom = Xnew                         
        for j in range(dim):
            vi.chrom[j] = max(vi.chrom[j], self.bound[0][0])
            def mini(x,y):
                if(x<y):
                    return x
                else:
                    return y
            vi.chrom[j] = mini(vi.chrom[j], self.bound[1][0])
        vi.calculateFitness()
        if vi.score > self.foodList[i].score:           
            self.foodList[i] = vi
            if vi.score > self.foodScore[i]:            
                self.foodScore[i] = vi.score
                if vi.score > self.bestFood.score:     
                    self.bestFood = copy.copy(vi)

    def onlookerBeePhase(self,Iter):
        a = 2 * (1 - Iter / self.maxIterCount)
        a2=-1+Iter*((-1)/self.maxIterCount)
        maxScore = np.max(self.foodScore)        
        accuFitness = [(0.9*d/maxScore+0.1, k) for k,d in enumerate(self.foodScore)] 
        for k in range(0, self.onlookerCount):
            arr2=[d[1] for d in accuFitness if d[0] >= random.random()]
            arr1=[d[1] for d in accuFitness if d[0] < random.random()]
            if(len(arr1)>0):
                j = random.choice(arr1)
                self.updateFood(j)
                #self.woa(j,self.foodCount,a,a2)
            if(len(arr2)>0):
                l=random.choice(arr2)
                self.woa(k,self.foodCount,a,a2)



    def scoutBeePhase(self):
        for i in range(0, self.foodCount):
            if self.foodList[i].invalidCount > self.maxInvalidCount:                   
                self.foodList[i] = ABCIndividual(self.bound)
                self.foodScore[i] = max(self.foodScore[i], self.foodList[i].score)

    def solve(self):
        trace = []
        trace.append(self.bestFood.score)
        for k in range(self.maxIterCount):
            self.employedBeePhase()
            self.onlookerBeePhase(k)
            self.scoutBeePhase()
            trace.append(self.bestFood.score)
            if(color and pp):
                print(colored("\r[At iteration "+str(k)+' the best fitness is '+str(self.bestFood.score)+"]",'blue',attrs=['bold']),end=endc)
            elif(pp):
                print("\r['At iteration '"+ str(k)+ "' the best fitness is '"+ str(self.bestFood.score)+"']",end=endc)
        if(pp):
            print("\n\n")
        return trace 

def run(population,iter,limit,lb,ub,vardim):
    random.seed()
    bound = np.tile([[lb], [ub]], vardim)
    abs = ArtificialBeeColony(population, population, bound, iter, limit)
    print(colored("ABC WHALE Started",'green'))
    if(Threading==True):
        trace[4]=(abs.solve(),"ABC_WHALE",'m')
    else:
        return abs.solve()
    return abs.solve()
