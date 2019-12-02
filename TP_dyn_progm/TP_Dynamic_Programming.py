#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:46:08 2019

@author: Lily Amsellem
"""
import math
import numpy as np
import numpy.random as random
import scipy.special as special

import matplotlib.pyplot as plt

"""
x is the stock
D is the demand
o is the order
v is the quantity sold at an instant t
V is the values of Bellman Function
"""

#data values
T=7
p=0.5
n=[0,15,12,10,10,10,40,40,0]

def f(x,o,d):
    #v=min(d,x+o)
    #return x+o-v
    return(max(x+o-d,0))

def L(x,o,d):
    v=min(d,x+o)
    return 2*v-o


###########SOLVING THE PROBLEM#########
def dynamicProgram(K=lambda x:0):
    V=np.zeros((51,8))
    optimalOrders=np.zeros((51,7))
    
    #Final Value
    for x in range(51):
        V[x,T]=K(x)

    for t in range(T-1,-1,-1):
        for x in range(51):
            V[x,t]=-math.inf        
            maxOrder=min(50-x,10)
            for o in range(maxOrder+1):
                Vo=0
                for k in range(n[t+1]+1):
                    #Vo=2*min(n[t+1]*p,x+o)-o
                    #Vo+=V[x+o-min(k,x+o),t+1]*special.binom(n[t+1],k)*(p**k)*(1-p)**(n[t+1]-k)
                    Vo+=(2*min(k,x+o)+V[f(x,o,k),t+1])*special.binom(n[t+1],k)*(p**k)*(1-p)**(n[t+1]-k)
                Vo=Vo-o
                if Vo>V[x,t]:
                    V[x,t]=Vo
                    optimalOrders[x,t]=o
    return(V,optimalOrders)

#Plot the optimal values in function of S0
def DP_display(K=lambda x:0):
    V,optimalOrders=dynamicProgram(K)
    X=np.arange(51)
    plt.plot(X,V[:,0])

#Question 5: Case where we buy an initial stock at price "buy_price" (0.75,1 or 1.25$)    
def buy_initial_stock(V,buy_price):
    return (np.argmax(V[:,0]-buy_price*np.arange(0,51)),np.max(V[:,0]-buy_price*np.arange(0,51)))



#########SIMULATING THE PROBLEM#############
    
#Question 6: Estimate optimal value through Monte-Carlo and plot in function of the orders u
def Monte_Carlo(nb_samples,order_strategy,x0=0):
    demand=np.zeros((nb_samples,T+1))
    orders=np.zeros((nb_samples,T+1))
    stock=np.zeros((nb_samples,T+1))
    stock[:,0]=x0*np.ones(nb_samples)
    V=np.zeros(nb_samples)
    meanMC=0
    
    
    for t in range(T+1):
        #draw binomial samples
        demand[:,t]=np.random.binomial(n[t],p,size=nb_samples)
        
    for i in range(nb_samples):
        for t in range(T):
            x=stock[i,t]
            orders[i,t]=min(order_strategy,50-x)
            o=orders[i,t]
            stock[i,t+1]=f(x,o,demand[i,t+1])
            V[i]+=L(x,o,demand[i,t+1])
        meanMC+=V[i]/nb_samples
    return(meanMC)

def MC_display(nb_samples,x0=0,compare=False):
    
    orders=np.arange(11)
    MC_results=np.zeros(11)
    for i in range(11):
        MC_results[i]=Monte_Carlo(nb_samples,orders[i],x0)
    
    if (compare==False):
        plt.plot(orders,MC_results) 
        plt.xlabel("order strategy")
        plt.plot(orders,MC_results)
        plt.title("Values through Monte-Carlo approach")
    
    else:
        V=dynamicProgram(K=lambda x:0)[0]
        print("Expected value of the optimal policy : ",V[x0,0])
        print("Value computed through Monte-Carlo : ",np.max(MC_results))


############INFORMATION STRUCTURE########@
#New info structure is hazard-decision--> decision is made with knowledge of d_{t+1}
            
    
def main():

    
    #QUESTION 6 : Plot Monte Carlo computations in function of order u 
    plt.figure()
    MC_display(1000,x0=20,compare=False)
    plt.show()
    
    #QUESTION 7: Check that optimal values through MC and Dynamic Programming coincide for s0=20
    MC_display(1000,x0=20,compare=True)
    
    
    #QUESTION 8 : Assume that final stock can be sold for 1$, and compare with the case 
    #where the final value function is 0
    plt.figure()
    DP_display()
    DP_display(K=lambda x:x)
    plt.title("Profit with and without selling the final stock")
    plt.show()
    


