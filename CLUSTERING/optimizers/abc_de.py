import numpy as np
import random, copy
from numpy.random import rand
from numpy.random import choice
from numpy import clip
from termcolor import colored

from solution import solution
import time
from data import endc,color




class ABCDEIndividual:
    def __init__(self, bound,objf,points, k, metric):
        self.score = 0.0
        self.objf=objf
        self.labelsPred=[]
        self.bound=bound
        self.points=points
        self.k=k 
        self.metric=metric
  
        self.invalidCount = 0 
        self.chrom = [random.uniform(a,b) for a,b in zip(bound[0,:],bound[1,:])] 
        self.calculateFitness()

    def calculateFitness(self):
        temp_chrom=np.reshape(self.chrom, (self.k,(int)(len(self.bound[0])/self.k)))
        if self.objf.__name__ == 'SSE' or self.objf.__name__ == 'SC' or self.objf.__name__ == 'DI':
            self.score,self.labelsPred=self.objf(temp_chrom,self.points, self.k, self.metric) 
        else:
            self.score,self.labelsPred=self.objf(temp_chrom,self.points, self.k)
        self.score=1.0/(1.0+self.score)

def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[0,i], bounds[1,i]) for i in range(len(bounds[0]))]
    return mutated_bound


def crossover(mutated, target, dims, cr):
    p = rand(dims)
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial

class ABCDE:
    def __init__(self,objf, foodCount, onlookerCount, bound, maxIterCount, maxInvalidCount,points, k, metric):
        self.foodCount = foodCount                 
        self.onlookerCount = onlookerCount           
        self.bound = bound
        self.F = 0.5 
        self.cr=0.7  
        self.maxIterCount = maxIterCount           
        self.maxInvalidCount = maxInvalidCount
        self.objf=objf
        self.points=points
        self.k=k 
        self.metric=metric    
        self.foodList = [ABCDEIndividual(self.bound,self.objf,self.points,self.k,self.metric) for k in range(self.foodCount)]
        self.foodScore = [d.score for d in self.foodList]  
        self.bestFood = copy.copy(self.foodList[np.argmax(self.foodScore)])

        

    def updateFood(self, i):         

        
        vj = copy.deepcopy(self.foodList[i]) 
        candidates = [candidate for candidate in range(self.foodCount) if candidate != i]
        ia,ib,ic=tuple(choice(candidates,3,replace=False))
        a=np.asarray(self.foodList[ia].chrom)
        b=np.asarray(self.foodList[ib].chrom)
        c=np.asarray(self.foodList[ic].chrom)

        mutated = mutation([a, b, c], self.F)
        mutated = check_bounds(mutated, self.bound)
        trial = crossover(mutated, self.foodList[i].chrom, len(self.bound[0]), self.cr)
        vj.chrom=trial
        vj.calculateFitness()
        
        if vj.score > self.foodList[i].score:           
            self.foodList[i] = vj
            if vj.score > self.foodScore[i]:            
                self.foodScore[i] = vj.score
                if vj.score > self.bestFood.score:     
                    self.bestFood = vj
            self.foodList[i].invalidCount = 0
        else:
            self.foodList[i].invalidCount += 1
            
    def employedBeePhase(self):
        for i in range(0, self.foodCount):              
            self.updateFood(i)            

    def onlookerBeePhase(self):
        maxScore = np.max(self.foodScore)        
        accuFitness = [(0.9*d/maxScore+0.1, k) for k,d in enumerate(self.foodScore)] 
        for k in range(0, self.onlookerCount):
            i = random.choice([d[1] for d in accuFitness if d[0] >= random.random()]) 
            self.updateFood(i)

    def scoutBeePhase(self):
        for i in range(0, self.foodCount):
            if self.foodList[i].invalidCount > self.maxInvalidCount:                    
                self.foodList[i] = ABCDEIndividual(self.bound,self.objf,self.points,self.k,self.metric)
                self.foodScore[i] = max(self.foodScore[i], self.foodList[i].score)

    def solve(self):
        trace1 = []
        trace=[]
        trace1.append((self.bestFood.score, np.mean(self.foodScore)))
        for k in range(self.maxIterCount):
            self.employedBeePhase()
            self.onlookerBeePhase()
            self.scoutBeePhase()
            tt=self.bestFood.score
            tt=(1-tt)/tt
            trace.append(tt)
            trace1.append((self.bestFood.score, np.mean(self.foodScore)))
            if(color):
                print(colored("\r[At iteration "+str(k)+' the best fitness is '+str(tt)+"]",'blue',attrs=['bold']),end=endc)
            else:
                print("\r['At iteration '"+ str(k)+ "' the best fitness is '"+ str(tt)+"']",end=endc)

        print("\n\n")
        return trace,self.bestFood.score,self.bestFood.chrom,self.bestFood.labelsPred



def ABC_DE(objf,lb,ub,dim,SearchAgents_no,Max_iter,k,points, metric):
    bound = np.tile([[lb], [ub]], dim)
    if(not color):
        print("ABC+DE is optimizing  \""+objf.__name__+"\"") 
    else:
        print(colored("ABC_DE is optimizing using "+objf.__name__,'yellow'))   
    abc_de = ABCDE(objf,SearchAgents_no, SearchAgents_no, bound, Max_iter, 200,points, k, metric) 
    s=solution()
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    trace,best,best_chrom,best_label=abc_de.solve()
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=trace
    s.optimizer="ABC_DE" 
    s.objfname=objf.__name__
    s.best = best
    s.bestIndividual = best_chrom
    s.labelsPred = np.array(best_label, dtype=np.int64)
    return s