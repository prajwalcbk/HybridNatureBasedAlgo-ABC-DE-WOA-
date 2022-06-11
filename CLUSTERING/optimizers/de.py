# differential evolution search of the two-dimensional sphere objective function
from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmax
from termcolor import colored
import numpy as np


from data import endc,color

from solution import solution
import time





def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound


def crossover(mutated, target, dims, cr):
    p = rand(dims)
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial


def differential_evolution(objf,pop_size, bounds, iter, F, cr, points, k, metric,dim) :
    def obj(temp_chrom,objf,points, k, metric,dim):
        temp_chrom=np.reshape(temp_chrom, (k,(int)(dim/k)))
        if  objf.__name__ == 'SSE' or  objf.__name__ == 'SC' or  objf.__name__ == 'DI':
             score, labelsP= objf(temp_chrom, points,  k,  metric) 
        else:
             score, labelsP= objf(temp_chrom, points,  k)
        score=1.0/(1.0+score)
        return score , labelsP
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    obj_all = [obj(ind,objf,points, k, metric,dim) for ind in pop]
    obj_all_score=[]
    obj_all_labels=[]
    for i in obj_all:
        obj_all_score.append(i[0])
        obj_all_labels.append(i[1])
    
    obj_all=obj_all_score
    
    best_vector = pop[argmax(obj_all)]
    best_labels=obj_all_labels[argmax(obj_all)]
    best_obj = max(obj_all)
    prev_obj = best_obj
    obj_iter = list()
    for i in range(iter):
        for j in range(pop_size):
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            mutated = mutation([a, b, c], F)
            mutated = check_bounds(mutated, bounds)
            trial = crossover(mutated, pop[j], len(bounds), cr)
            obj_target = obj(pop[j],objf,points, k, metric,dim)
            obj_trial = obj(trial,objf,points, k, metric,dim)
            if obj_trial[0] > obj_target[0]:
                pop[j] = trial
                obj_all[j] = obj_trial[0]
                obj_all_labels[j]=obj_trial[1]
        best_obj = max(obj_all)
        if best_obj > prev_obj:
            best_vector = pop[argmax(obj_all)]
            best_labels=obj_all_labels[argmax(obj_all)]
            prev_obj = best_obj
        tt=best_obj
        tt=(1-tt)/tt
        if(color):
            print(colored("\r[At iteration "+str(i)+' the best fitness is '+str(tt)+"]",'blue',attrs=['bold']),end=endc)
        else:
            print("\r['At iteration '"+ str(i)+ "' the best fitness is '"+ str(tt)+"']",end=endc)
        obj_iter.append(tt)
    print("\n\n")
    return obj_iter,best_obj,best_vector,best_labels




def DE(objf,lb,ub,dim,SearchAgents_no,Max_iter,k,points, metric):
    bound = asarray([(lb, ub)]*dim)
    if(not color):
        print("DE is optimizing  \""+objf.__name__+"\"")  
    else:
        print(colored("DE is optimizing using "+objf.__name__,'yellow')) 
    F = 0.5
    cr = 0.7
    s=solution()
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    trace,best,best_chrom,best_label=differential_evolution(objf,SearchAgents_no, bound, Max_iter, F, cr , points, k, metric,dim) 
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart

    s.convergence=trace
    s.optimizer="DE" 
    s.objfname=objf.__name__
    s.best = best
    s.bestIndividual = best_chrom
    s.labelsPred = np.array(best_label, dtype=np.int64)

    return s

