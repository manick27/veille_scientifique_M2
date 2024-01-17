import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


n = 10
m = 5
Q = 15
c = np.array([[0, 3, 5, 7, 9, 11, 13, 15, 17, 19],
              [3, 0, 2, 4, 6, 8, 10, 12, 14, 16],
              [5, 2, 0, 2, 4, 6, 8, 10, 12, 14],
              [7, 4, 2, 0, 2, 4, 6, 8, 10, 12],
              [9, 6, 4, 2, 0, 2, 4, 6, 8, 10],
              [11, 8, 6, 4, 2, 0, 2, 4, 6, 8],
              [13, 10, 8, 6, 4, 2, 0, 2, 4, 6],
              [15, 12, 10, 8, 6, 4, 2, 0, 2, 4],
              [17, 14, 12, 10, 8, 6, 4, 2, 0, 2],
              [19, 16, 14, 12, 10, 8, 6, 4, 2, 0]])
o = np.array([1, 2, 3, 4, 5])
d = np.array([6, 7, 8, 9, 10])
q = np.array([4, 5, 6, 7, 8])
e = np.array([8, 8, 8, 10, 10])
f = np.array([10, 10, 10, 12, 12])
g = np.array([9, 9, 9, 11, 11])
h = np.array([11, 11, 11, 13, 13])
s = 16
y = np.array([[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 0],
              [1, 1, 1, 0, 1],
              [1, 1, 0, 1, 1],
              [1, 0, 1, 1, 1],
              [0, 1, 1, 1, 1],
              [1, 1, 0, 0, 0],
              [1, 0, 1, 0, 0],
              [1, 0, 0, 1, 0],
              [1, 0, 0, 0, 1],
              [0, 1, 1, 0, 0],
              [0, 1, 0, 1, 0],
              [0, 1, 0, 0, 1],
              [0, 0, 1, 1, 0],
              [0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1]])
p = np.array([0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # Vecteur des probabilités des scénarios

def is_terminal(s):
     return s[0] == 0 and np.sum(s[2]) == 0

def choose_action(s, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(m)
    else:
        return np.argmax(Q[s[0], min(s[1], Q), int(''.join(map(str, s[2].astype(int))), 2), :])

def take_action(s, a):
    i = s[0]
    l = s[1]
    r = s[2]
    if r[a] == 0:
        return s, 0
    elif i == o[a]:
        i_prime = d[a]
        l_prime = l + q[a]
        r_prime = r.copy()
        r_prime[a] = 0
        s_prime = (i_prime, l_prime, r_prime)
        r = -c[i, i_prime]
        return s_prime, r
    elif i == d[a]:
        i_prime = 0
        l_prime = l - q[a]
        r_prime = r.copy()
        s_prime = (i_prime, l_prime, r_prime)
        r = q[a] - c[i, i_prime]
        return s_prime, r
    else:
        i_prime = o[a]
        l_prime = l
        r_prime = r
        s_prime = (i_prime, l_prime, r_prime)
        r = -c[i, i_prime]
        return s_prime, r

alpha = 0.1
gamma = 0.9
epsilon = 0.1
max_episodes = 1000
Q = np.zeros((n + 1, Q + 1, 2 ** m, m))
for episode in range(max_episodes):
    i = 0
    l = 0
    s = (i, l, r)
    while not is_terminal(s):
        a = choose_action(s, Q, epsilon)
        s_prime, r = take_action(s, a)
        Q[s[0], min(s[1], Q), int(''.join(map(str, s[2].astype(int))), 2), a] = Q[s[0], min(s[1], Q), int(''.join(map(str, s[2].astype(int))), 2), a] + alpha * (r + gamma * np.max(Q[s_prime[0], s_prime[1], int(''.join(map(str, s_prime[2].astype(int))), 2), :]) - Q[s[0], min(s[1], Q), int(''.join(map(str, s[2].astype(int))), 2), a])
        s = s_prime
