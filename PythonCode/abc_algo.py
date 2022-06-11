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

class ABSIndividual:
    def __init__(self, bound):
        self.score = 0.
        self.invalidCount = 0  
        self.chrom = [random.uniform(a,b) for a,b in zip(bound[0,:],bound[1,:])] 
        self.calculateFitness()        
    def calculateFitness(self):
        self.score = GrieFunc(self.chrom) 
        
class ArtificialBeeSwarm:
    def __init__(self, foodCount, onlookerCount, bound, maxIterCount=1000, maxInvalidCount=200):
        self.foodCount = foodCount                  
        self.onlookerCount = onlookerCount          
        self.bound = bound                          
        self.maxIterCount = maxIterCount            
        self.maxInvalidCount = maxInvalidCount      
        self.foodList = [ABSIndividual(self.bound) for k in range(self.foodCount)]   
        self.foodScore = [d.score for d in self.foodList]                             
        self.bestFood = self.foodList[np.argmax(self.foodScore)]                    

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
                    self.bestFood = vi
            self.foodList[i].invalidCount = 0
        else:
            self.foodList[i].invalidCount += 1
            
    def employedBeePhase(self):
        for i in range(0, self.foodCount):              
            self.updateFood(i)            

    def onlookerBeePhase(self): 
        maxScore = self.bestFood.score     
        accuFitness = [(0.9*d/maxScore+0.1, k) for k,d in enumerate(self.foodScore)]       
        for k in range(0, self.onlookerCount):
            i = random.choice([d[1] for d in accuFitness if d[0] >= random.random()]) 
            self.updateFood(i)

    def scoutBeePhase(self):
        for i in range(0, self.foodCount):
            if self.foodList[i].invalidCount > self.maxInvalidCount:                  
                self.foodList[i] = ABSIndividual(self.bound)
                self.foodScore[i] = max(self.foodScore[i], self.foodList[i].score)
                self.bestFood = copy.copy(self.foodList[np.argmax(self.foodScore)])

    def solve(self):
        trace = []
        trace.append(self.bestFood.score)
        for k in range(self.maxIterCount):
            self.employedBeePhase()
            self.onlookerBeePhase()
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
    abs = ArtificialBeeSwarm(population, population, bound, iter, limit)
    print(colored("ABC Started",'green'))
    if(Threading==True):
        trace[0]=(abs.solve(),"ABC",'k')
    else:
        return abs.solve()