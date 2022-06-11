import abc_algo 
import whale_algo
import de_algo
import abc_de_algo
import abc_whale_algo
import abc_de_whale_algo
import matplotlib.pyplot as plt
import threading
import time
import numpy
from data import Threading,trace,ppp
from termcolor import colored

population=30
iter=100
limit=200
ub=600
lb=-600
vardim=30
number_of_runs=1



def f(pp,s,trace,name):
    if(pp):
        print(trace[-1])
        print(colored("Total time taken to execute is "+str(time.time()-s),'yellow'))
    print(colored(name,'red'))



def printResult(tt):
    for i in range(len(tt)):
        algo=tt[i][1]
        c=tt[i][2]
        tra=tt[i][0]
        plt.plot([(1-d)/d for d in tra], c, label=algo)
    plt.xlabel("Iteration")
    plt.ylabel("function value")
    plt.title("Hybrid algorithm for function optimization")
    plt.legend()
    plt.show()

s=time.time()
if(Threading):
    threading.Thread(target=abc_algo.run,args=(population,iter,limit,lb,ub,vardim)).start()
    print("ABC Completed")
    threading.Thread(target=de_algo.run,args=(population,iter,limit,lb,ub,vardim)).start()
    print("DE Completed")
    threading.Thread(target=whale_algo.run,args=(population,iter,limit,lb,ub,vardim)).start()
    print("WHALE Completed")
    threading.Thread(target=abc_de_algo.run,args=(population,iter,limit,lb,ub,vardim)).start()
    print("ABC DE Completed")
    threading.Thread(target=abc_whale_algo.run,args=(population,iter,limit,lb,ub,vardim)).start()
    print("ABC WHALE  Completed")
    threading.Thread(target=abc_de_whale_algo.run,args=(population,iter,limit,lb,ub,vardim)).start()
    print("ABC DE WHALE Completed")
    while(True):
        if(all([i[1] for i in trace])):
            break
def exe(func):
    global population,iter,limit,lb,ub,vardim,number_of_runs
    tt=[]
    for i in range(number_of_runs):
        tt.append(func(population,iter,limit,lb,ub,vardim))
    tt = numpy.mean(tt, axis=0,dtype=numpy.float128).tolist()
    return tt 
        
if(number_of_runs>1):
    trace=[]
    numpy.set_printoptions(precision=20)

    trace.append((exe(abc_algo.run),"ABC",'k'))
    f(ppp,s,trace,"ABC Completed\n")
    trace.append((exe(de_algo.run),"DE",'g'))
    f(ppp,s,trace,"DE Completed\n")
    trace.append((exe(whale_algo.run),"WHALE",'b'))
    f(ppp,s,trace,"WHALE Completed\n")
    trace.append((exe(abc_de_algo.run),"ABC_DE",'y'))
    f(ppp,s,trace,"ABC DE Completed\n")
    trace.append((exe(abc_whale_algo.run),"ABC_WHALE",'m'))
    f(ppp,s,trace,"ABC WHALE  Completed\n")
    trace.append((exe(abc_de_whale_algo.run),"ABC_DE_WHALE",'r'))
    f(ppp,s,trace,"ABC DE WHALE Completed\n")

else:
    trace=[]
    trace.append((abc_algo.run(population,iter,limit,lb,ub,vardim),"ABC",'k'))
    f(ppp,s,trace,"ABC Completed\n")
    trace.append((de_algo.run(population,iter,limit,lb,ub,vardim),"DE",'g'))
    f(ppp,s,trace,"DE Completed\n")
    trace.append((whale_algo.run(population,iter,limit,lb,ub,vardim),"WHALE",'b'))
    f(ppp,s,trace,"WHALE Completed\n")
    trace.append((abc_de_algo.run(population,iter,limit,lb,ub,vardim),"ABC_DE",'y'))
    f(ppp,s,trace,"ABC DE Completed\n")
    trace.append((abc_whale_algo.run(population,iter,limit,lb,ub,vardim),"ABC_WHALE",'m'))
    f(ppp,s,trace,"ABC WHALE  Completed\n")
    trace.append((abc_de_whale_algo.run(population,iter,limit,lb,ub,vardim),"ABC_WHALE_DE",'r'))
    f(ppp,s,trace,"ABC DE WHALE Completed\n")
e=time.time()
print(colored("Total time taken to execute is "+str(e-s),'yellow'))
printResult(trace)
