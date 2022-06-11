import random
import math 
import copy 
from data import Threading,trace
from data import color,endc,pp
from termcolor import colored

def fitness_rastrigin(data): 
    s1 = 0.
    s2 = 1.
    for k, x in enumerate(data):
        s1 = s1 + x ** 2
        s2 = s2 * math.cos(x/math.sqrt(k+1))
    y = (1./4000.) * s1-s2 + 1
    return 1./(1.+y)


def fitness_sphere(position):
	fitness_value = 0.0
	for i in range(len(position)):
		xi = position[i]
		fitness_value += (xi * xi)
	return fitness_value



class whale:
	def __init__(self, fitness, dim, minx, maxx, seed):
		self.rnd = random.Random(seed)
		self.position = [0.0 for i in range(dim)]

		for i in range(dim):
			self.position[i] = ((maxx - minx) * self.rnd.random() + minx)

		self.fitness = fitness(self.position) 



def woa(fitness, max_iter, n, dim, minx, maxx):
	rnd = random.Random(0)
	whalePopulation = [whale(fitness, dim, minx, maxx, i) for i in range(n)]
	Xbest = copy.copy(whalePopulation[0].position)
	Fbest = whalePopulation[0].fitness
	Fb=[]
	for i in range(1,n):
		if whalePopulation[i].fitness > Fbest:
			Fbest = whalePopulation[i].fitness
			Xbest = copy.copy(whalePopulation[i].position)
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
			whalePopulation[i].fitness = fitness(whalePopulation[i].position)
			if (whalePopulation[i].fitness > Fbest):
				Xbest = copy.copy(whalePopulation[i].position)
				Fbest = whalePopulation[i].fitness
		Fb.append(Fbest)
		if(color and pp):
			print(colored("\r[At iteration "+str(Iter)+' the best fitness is '+str(Fbest)+"]",'blue',attrs=['bold']),end=endc)
		elif(pp):
			print("\r['At iteration '"+ str(Iter)+ "' the best fitness is '"+ str(Fbest)+"']",end=endc)
		Iter += 1
	if(pp):
		print("\n\n")
	Fb.append(Fbest)
	return Fb



def run(population,iter,limit,lb,ub,vardim):
	fitness = fitness_rastrigin
	print(colored("WHALE Started",'green'))
	if(Threading==True):
		trace[2]=(woa(fitness, iter, population, vardim, lb, ub),"WHALE",'b')
	else:
		return woa(fitness, iter, population, vardim, lb, ub)

