# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:19:33 2018

@author: Erik
"""
import numpy as np

letter_dict = {1: 'a', 2:'b', 3:'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i',
               10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15:'o', 
               16:'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 
               23: 'w', 24: 'x', 25: 'y', 26: 'z'}

reverse_lookup = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 
                  'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 
                  'r':17, 's': 18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 
                  'y':24, 'z':25}

def read_data(file_name):
    file = open('./data/' + file_name , 'r')
    y = []
    X = []
    y.append([])
    X.append([])
    x = []
    cur = 0
    for line in file:
        temp = line.split()
        y[cur].append(reverse_lookup[temp[1]])
        x.append(np.array([1] + temp[5:]))
        
        if(temp[2] == str(-1)):
            y[cur] = np.array(y[cur])
            X[cur] = np.array(x).astype(np.int)
            y.append([])
            X.append([])
            x = []
            cur += 1
    #no matter what the last element will be [] at this point,  so just remove that.
    
    X = X[:-1]
    y = y[:-1]
    
    return X, y
            


  