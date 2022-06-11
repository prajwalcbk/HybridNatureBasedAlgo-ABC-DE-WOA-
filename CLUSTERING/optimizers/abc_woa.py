import random, math, copy


from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
import numpy as np
from numpy import argmax
from numpy import around
from termcolor import colored

from solution import solution
import time
from data import endc,color


class ABC_WOAIndividual:
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




class ABC_WOA:
    def __init__(self,objf, foodCount, onlookerCount, bound, maxIterCount, maxInvalidCount,points, k, metric):
        self.foodCount = foodCount                 
        self.onlookerCount = onlookerCount           
        self.bound = bound
        self.objf=objf
        self.points=points
        self.k=k 
        self.metric=metric

        self.F = 0.5 
        self.cr=0.7  
        self.maxIterCount = maxIterCount           
        self.maxInvalidCount = maxInvalidCount    
        self.foodList = [ABC_WOAIndividual(self.bound,self.objf,self.points,self.k,self.metric) for k in range(self.foodCount)]
        self.foodScore = [d.score for d in self.foodList] 
        self.bestFood = copy.copy(self.foodList[np.argmax(self.foodScore)])
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
        D = [0.0 for ch in range(dim)]
        D1 = [0.0 for ch in range(dim)]
        Xnew = [0.0 for ch in range(dim)]
        Xrand = [0.0 for ch in range(dim)]
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
        vi.chrom=Xnew

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
            if(len(arr2)>0):
                l=random.choice(arr2)
                self.woa(k,self.foodCount,a,a2)
                
                        
            

    def scoutBeePhase(self):
        for i in range(0, self.foodCount):
            if self.foodList[i].invalidCount > self.maxInvalidCount:                   
                self.foodList[i] = ABC_WOAIndividual(self.bound,self.objf,self.points,self.k,self.metric)
                self.foodScore[i] = max(self.foodScore[i], self.foodList[i].score)

    def solve(self):
        trace1 = []
        trace=[]
        trace1.append((self.bestFood.score, np.mean(self.foodScore)))
        
        
        for k in range(self.maxIterCount):
            self.employedBeePhase()
            self.onlookerBeePhase(k)
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




def ABCWOA(objf,lb,ub,dim,SearchAgents_no,Max_iter,k,points, metric):
    bound = np.tile([[lb], [ub]], dim)
    if(not color):
        print("ABC_WOA is optimizing  \""+objf.__name__+"\"")  
    else:
        print(colored("ABC_WOA is optimizing using "+objf.__name__,'yellow'))   
    abc_de = ABC_WOA(objf,SearchAgents_no, SearchAgents_no, bound, Max_iter, 200,points, k, metric) 
    s=solution()
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    trace,best,best_chrom,best_label=abc_de.solve()
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart

    s.convergence=trace
    s.optimizer="ABC_WOA" 
    s.objfname=objf.__name__
    s.best = best
    s.bestIndividual = best_chrom
    s.labelsPred = np.array(best_label, dtype=np.int64)

    return s