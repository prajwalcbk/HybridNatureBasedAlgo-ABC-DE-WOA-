


import random
import math  
import copy
import numpy as np
from data import endc,color
from termcolor import colored
from solution import solution
import time


class whale:
    def __init__(self, objf, dim, minx, maxx, seed,points, k, metric):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]
        self.objf=objf
        self.labelsPred=[]
        self.points=points
        self.k=k 
        self.metric=metric
        self.fitness=0
        self.dim=dim
        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
        
    def calculateFitness(self):
        temp_chrom=np.reshape(self.position, (self.k,(int)(self.dim/self.k)))
        if self.objf.__name__ == 'SSE' or self.objf.__name__ == 'SC' or self.objf.__name__ == 'DI':
            self.fitness,self.labelsPred=self.objf(temp_chrom,self.points, self.k, self.metric) 
        else:
            self.fitness,self.labelsPred=self.objf(temp_chrom,self.points, self.k)
        self.fitness=1.0/(1.0+self.fitness)
         
def woa(fitness, max_iter, n, dim, minx, maxx,points, k, metric) :
	rnd = random.Random(0)

	whalePopulation = [whale(fitness, dim, minx, maxx, i,points, k, metric) for i in range(n)]

	Xbest = copy.copy(whalePopulation[0].position)
	Fbest = whalePopulation[0].fitness
	label_pred=whalePopulation[0].labelsPred
	Xb=[]
	Fb=[]

	for i in range(1,n): 
		if whalePopulation[i].fitness > Fbest:
			Fbest = whalePopulation[i].fitness
			Xbest = copy.copy(whalePopulation[i].position)
			label_pred=whalePopulation[i].labelsPred

	Iter = 0
	while Iter < max_iter:
		a = 2 * (1 - Iter / max_iter)
		a2=-1+Iter*((-1)/max_iter)

		for i in range(n):
			A = 2 * a * rnd.random() - a
			C = 2 * rnd.random()
			b = 1
			l = (a2-1)*rnd.random()+1
			p = rnd.random()
			D = [0.0 for i in range(dim)]
			D1 = [0.0 for i in range(dim)]
			Xnew = [0.0 for i in range(dim)]
			Xrand = [0.0 for i in range(dim)]
			if p < 0.5:
				if abs(A) > 1:
					for j in range(dim):
						D[j] = abs(C * Xbest[j] - whalePopulation[i].position[j])
						Xnew[j] = Xbest[j] - A * D[j]
				else:
					p = random.randint(0, n - 1)
					while (p == i):
						p = random.randint(0, n - 1)

					Xrand = whalePopulation[p].position

					for j in range(dim):
						D[j] = abs(C * Xrand[j] - whalePopulation[i].position[j])
						Xnew[j] = Xrand[j] - A * D[j]
			else:
				for j in range(dim):
					D1[j] = abs(Xbest[j] - whalePopulation[i].position[j])
					Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest[j]

			for j in range(dim):
				whalePopulation[i].position[j] = Xnew[j]

		for i in range(n):
			for j in range(dim):
				whalePopulation[i].position[j] = max(whalePopulation[i].position[j], minx)
				whalePopulation[i].position[j] = min(whalePopulation[i].position[j], maxx)
			whalePopulation[i].calculateFitness()
			if (whalePopulation[i].fitness > Fbest):
				Xbest = copy.copy(whalePopulation[i].position)
				Fbest = whalePopulation[i].fitness
				label_pred=whalePopulation[i].labelsPred
		if Iter % 1 == 0:
			Xb.append(Iter)
			tt=Fbest
			tt=(1-tt)/tt
			Fb.append(tt)
			if(color):
				print(colored("\r[At iteration "+str(Iter)+' the best fitness is '+str(tt)+"]",'blue',attrs=['bold']),end=endc)
			else:
				print("\r['At iteration '"+ str(Iter)+ "' the best fitness is '"+ str(tt)+"']",end=endc)
		Iter += 1
	print("\n\n")
	return Fb,Fbest,Xbest,label_pred




def WOA(objf,lb,ub,dim,SearchAgents_no,Max_iter,k,points, metric):
    if(color):  
        print(colored("WOA is optimizing using "+objf.__name__,'yellow')) 
    else:
        print("WOA is optimizing  \""+objf.__name__+"\"")  
    s=solution()
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    trace,best,best_chrom,best_label= woa(objf,Max_iter,SearchAgents_no, dim, lb,ub ,points, k, metric) 
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart

    s.convergence=trace
    s.optimizer="WOA" 
    s.objfname=objf.__name__
    s.best = best
    s.bestIndividual = best_chrom
    s.labelsPred = np.array(best_label, dtype=np.int64)

    return s
