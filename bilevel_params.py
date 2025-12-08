# Copyright (c) 2025 Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Authors: Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Email: giuseppe.stecca@gmail.com

import itertools
import math
import random
import pandas as pd
import numpy as np
import statistics as st
import os
import gurobipy as gb
class Data:
    def __init__(self, params: dict = None, inst_id: str = None, INSTANCE_FROM_FILE: bool = None, instance_file = 'instances_bilevel.csv'):
        # in this case we are generating a new instance
        if inst_id == None:
            self.params = params
            self.eta = 0.5
            self.Lmin = 1   
            self.Bmin = 1
            self.LS = 1
            self.LL = 4
            self.LT = 1
            self.LF = 1
            self.B = 1000
            self.alfa = 0.2
            self.alfa_prime = 0.5
            self.delta = 0.7
            self.tc = 0.05
            self.seed = 0
            self.DeltaRandom = 0.1
            self.DDecrease = False
            self.ins_id = None
            self.e_bar = {}
            self.e_hat = {}
            self.c_bar = {}
            self.c_hat = {}
            self.kappa = {}
            self.CF = {}
            self.CS = {}
            self.A = {}
            self.LT = 1
            self.LF = 1
            self.S = [i for i in range(1,self.LS+1)]
            self.F = [i for i in range(1,self.LS+1)]
            self.L = [l for l in range(1, self.LL+1)]
            self.T = [t for t in range(0,self.LT+1)]
            self.T0 = [t for t in range(0,self.LT+1)]

            self.tc = 0.05
            self.D = {t:100 for t in self.T}  
            self.A = [(i,j) for (i,j) in itertools.product(self.S,self.F) ]
            self.B = 1000
            self.alfa = 0.2
            self.alfa_prime = 0.5
            self.delta = 0.8

            self.e_bar = {(j,l): 1 + (50)*(1 -  math.log(l)/math.log(self.LL)) for j in self.F for l in self.L}
            self.e_hat = {(s,j): 1 for s in self.S for j in self.F}
            self.c_bar = {(j,l): 20 + 10*math.log(l) for j in self.F for l in self.L}
            self.c_hat = {(s,j): 5 for s in self.S for j in self.F}
            self.kappa = { (j,l): (self.B/(self.LF * self.LL))*(1 + math.log(l)) for j in self.F for l in self.L }
            self.CF = {(j,t) : (self.D[t]/(self.LF*2))*2 for j in self.F for t in self.T}
            self.CS = {(s,t) : (self.D[t]/self.LS)*2 for s in self.S for t in self.T}
            self.D = {t:100 for t in self.T}
        
        else:
            # in this case we are loading an instance from file
            if INSTANCE_FROM_FILE:
                self.params = params
                path = params.get('folder_input')
                self.from_excel(f'{path}/{inst_id}.xlsx')
            else:
            # in this case we are generating instance based on information given by csv file
                self.params = params
                df = pd.read_csv(instance_file, index_col='id')
                self.ins_id = inst_id
                self.DeltaRandom = df.loc[inst_id]['DeltaRandom']
                self.seed = int(df.loc[inst_id]['seed'])
                self.DDecrease = bool(df.loc[inst_id]['DDecrease'])
                self.delta = df.loc[inst_id]['delta'] #0.7  # decrease of cap over time

                self.eta = 0.5
                self.Lmin = 1
                self.Bmin = 1
                self.LT = df.loc[inst_id]['LT']
                self.T = [t for t in range(1,self.LT+1)] # Time horizon
                self.T0 = [t for t in range(0,self.LT+1)] # Time horizon
                self.LS = df.loc[inst_id]['LS']
                self.S = [i for i in range(1,self.LS+1)]
                self.LF = df.loc[inst_id]['LF']
                self.F = [j for j in range(1,self.LF+1)]
                self.LL = 4 # number of technologies
                self.L = [l for l in range(1, self.LL+1)]
                self.A = [(i,j) for (i,j) in itertools.product(self.S,self.F) ]
            
                #PARAMETERS
                #self.e_bar = {(j,l): 10 - (2/self.LL)*math.log(l) for j in self.F for l in self.L}
                self.e_bar = {(j,l): 1 + (50)*(1 -  math.log(l)/math.log(self.LL)) for j in self.F for l in self.L}
                #print('ebar ', e_bar)
                self.e_hat = {(s,j): 1 for s in self.S for j in self.F}
                #self.c_bar = {(j,l): 10 + (2/self.LL)*math.log(l) for j in self.F for l in self.L}
                self.c_bar = {(j,l): 20 + 10*math.log(l) for j in self.F for l in self.L}

                #print('cbar ', c_bar)
                self.c_hat = {(s,j): 5 for s in self.S for j in self.F}
                self.tc = 0.05
                self.D = {t:df.loc[inst_id]['D_t'] for t in self.T}
                if (self.DDecrease):
                    #DeltaRandom = self.Instaces[self.ins_id]['DeltaRandom']
                    for t in self.T:
                        if t>1:
                            #self.D[t] = max(0, (self.get_ebar_min(1) - DeltaRandom/2 )*self.D[t-1])
                            self.D[t] = self.get_ebar_min(1)*self.D[t-1]
                #self.b = {}
                
                self.B = df.loc[inst_id]['B']   #1000*self.LF # kEur
                self.alfa = 0.2
                self.alfa_prime = 0.5
                self.kappa = {(j,l): (self.B/(self.LF * self.LL))*(1 + math.log(l)) for j in self.F for l in self.L }
                            #{(j,l): 500*(1 + (2/self.LL)*math.log(l)) for j in self.F for l in self.L } # fixed technology activation cost
                #TODO aggiornare CF
                self.CF = {(j,t) : (self.D[t]/(self.LF*2))*2 for j in self.F for t in self.T}
                self.CS = {(s,t) : (self.D[t]/self.LS)*2 for s in self.S for t in self.T}
                print('###### CF #####')
                print(self.CF)
            
    ### compute max CO2 emissions
    def maxCO2(self):
        max_e_hat = max(self.e_hat.values())
        max_e_bar = max(self.e_bar.values())
        mean_e_hat = st.mean(self.e_hat.values())
        maxD = max(self.D.values())
        #TODO for randomized instance set E0 to max
        #E0 = maxD*max_e_bar + maxD*max_e_hat
        E0 = maxD*max_e_bar / (2* self.LF) # mean of e_bar
        #E0 = maxD*max_e_bar / (2* self.LF) + maxD*mean_e_hat/2 # mean of e_bar
        return E0
    ### compute min CO2 emissions
    def minCO2(self):
        min_e_hat = min(self.e_hat.values())
        min_e_bar = min(self.e_bar.values())
        mean_e_hat = st.mean(self.e_hat.values())
        minD = min(self.D.values())
        #TODO for randomized instance set E0 to max
        #E0 = maxD*max_e_bar + maxD*max_e_hat
        Emin = minD*min_e_bar / (self.LF* self.LF) # mean of e_hat
        return Emin

    def get_ebar_min(self, j:int):
        # min_{l \in L}\frac{e(jl)}{e(jl-1)}
        minEBar = float('inf')
        for l in range (2, len(self.L)+1):
            minEBar = min(minEBar, self.e_bar[j,l-1]/self.e_bar[j,l])
        return minEBar
    def randomize(self):
        random.seed(self.seed)
        #DeltaRandom = self.Instaces[self.ins_id]['DeltaRandom']
        self.D = {k:v *(1 + self.DeltaRandom*random.random() - self.DeltaRandom/2) for k,v in self.D.items() }
        #self.e_bar = {k:v *(1 + DeltaRandom*random.random() - DeltaRandom/2) for k,v in self.e_bar.items() } 
        self.e_hat = {k:v *(1 + self.DeltaRandom*random.random() - self.DeltaRandom/2) for k,v in self.e_hat.items() } # s,j
        #self.c_bar = {k:v *(1 + DeltaRandom*random.random() - DeltaRandom/2) for k,v in self.c_bar.items() } # j l
        self.c_hat = {k:v *(1 + self.DeltaRandom*random.random() - self.DeltaRandom/2) for k,v in self.c_hat.items() }
        #self.kappa = {k:v *(1 + DeltaRandom*random.random() - DeltaRandom/2) for k,v in self.kappa.items() }
        self.CF = {k:v *(1 + self.DeltaRandom*random.random() - self.DeltaRandom/2) for k,v in self.CF.items() }
        self.CS = {k:v *(1 + self.DeltaRandom*random.random() - self.DeltaRandom/2) for k,v in self.CS.items() }

    def to_excel(self, file_path: str):
        with pd.ExcelWriter(file_path) as writer:
            params = ['eta', 'Lmin', 'Bmin', 'LS', 'LL', 'LT', 'LF', 'B', 'alfa', 'alfa_prime', 'delta', 'tc', 'E0']
            values = [self.eta, self.Lmin, self.Bmin, self.LS, self.LL, self.LT, self.LF, self.B, self.alfa, 
                      self.alfa_prime, self.delta, self.tc, self.maxCO2()]
            dfg = pd.DataFrame(np.array([params, values]).T, columns=['parameter', 'value'])
            dfg.to_excel(writer, sheet_name='General', index=None)
            e_bar_df = pd.DataFrame([(k[0], k[1], v) for k, v in self.e_bar.items()], columns=['j', 'l', 'e_bar'])
            e_bar_df.to_excel(writer, sheet_name='e_bar', index=None)
            e_hat_df = pd.DataFrame([(k[0], k[1], v) for k, v in self.e_hat.items()], columns=['s', 'j', 'e_hat'])
            e_hat_df.to_excel(writer, sheet_name='e_hat', index=None)
            c_bar_df = pd.DataFrame([(k[0], k[1], v) for k, v in self.c_bar.items()], columns=['j', 'l', 'c_bar'])
            c_bar_df.to_excel(writer, sheet_name='c_bar', index=None)
            c_hat_df = pd.DataFrame([(k[0], k[1], v) for k, v in self.c_hat.items()], columns=['s', 'j', 'c_hat'])
            c_hat_df.to_excel(writer, sheet_name='c_hat', index=None)
            kappa_df = pd.DataFrame([(k[0], k[1], v) for k, v in self.kappa.items()], columns=['j', 'l', 'kappa'])
            kappa_df.to_excel(writer, sheet_name='kappa', index=None)
            CF_df = pd.DataFrame([(k[0], k[1], v) for k, v in self.CF.items()], columns=['j', 't', 'CF'])
            CF_df.to_excel(writer, sheet_name='CF', index=None)
            CS_df = pd.DataFrame([(k[0], k[1], v) for k, v in self.CS.items()], columns=['s', 't', 'CS'])
            CS_df.to_excel(writer, sheet_name='CS', index=None)
            D_df = pd.DataFrame([(k, v) for k, v in self.D.items()], columns=['t', 'D'])
            D_df.to_excel(writer, sheet_name='D', index=None)

            C = {t: self.maxCO2() *(1-(self.delta/(self.LT-1))*(t-1)) for t in self.T}

            df_C = pd.DataFrame([(k, v) for k, v in C.items()], columns=['t', 'C'])
            df_C.to_excel(writer, sheet_name='C', index=None)

    def from_excel(self, file_path: str):
        df = pd.read_excel(file_path, sheet_name=None)
        
        general_params = df['General']
        #for index, row in general_params.iterrows():
        #    setattr(self, row['parameter'], row['value'])
        self.eta = general_params.loc[general_params['parameter'] == 'eta']['value'].values[0]
        self.Lmin = int(general_params.loc[general_params['parameter'] == 'Lmin']['value'].values[0])
        self.Bmin = int(general_params.loc[general_params['parameter'] == 'Bmin']['value'].values[0])
        self.LS = int(general_params.loc[general_params['parameter'] == 'LS']['value'].values[0])
        self.S = [i for i in range(1,self.LS+1)]
        self.LL = int(general_params.loc[general_params['parameter'] == 'LL']['value'].values[0])
        self.L = [l for l in range(1, self.LL+1)]
        self.LT = int(general_params.loc[general_params['parameter'] == 'LT']['value'].values[0])
        self.T = [t for t in range(1,self.LT+1)] # Time horizon
        self.T0 = [t for t in range(0,self.LT+1)] # Time horizon
        self.LF = int(general_params.loc[general_params['parameter'] == 'LF']['value'].values[0])   
        self.F = [j for j in range(1,self.LF+1)]
        self.tc = general_params.loc[general_params['parameter'] == 'tc']['value'].values[0]

        self.A = [(i,j) for (i,j) in itertools.product(self.S,self.F) ]

        self.B = int(general_params.loc[general_params['parameter'] == 'B']['value'].values[0])
        self.alfa = general_params.loc[general_params['parameter'] == 'alfa']['value'].values[0]
        self.alfa_prime = general_params.loc[general_params['parameter'] == 'alfa_prime']['value'].values[0]
        self.delta = general_params.loc[general_params['parameter'] == 'delta']['value'].values[0]

        self.e_bar = {(int(row['j']), int(row['l'])): row['e_bar'] for index, row in df['e_bar'].iterrows()}
        self.e_hat = {(int(row['s']), int(row['j'])): row['e_hat'] for index, row in df['e_hat'].iterrows()}
        self.c_bar = {(int(row['j']), int(row['l'])): row['c_bar'] for index, row in df['c_bar'].iterrows()}
        self.c_hat = {(int(row['s']), int(row['j'])): row['c_hat'] for index, row in df['c_hat'].iterrows()}
        self.kappa = {(int(row['j']), int(row['l'])): row['kappa'] for index, row in df['kappa'].iterrows()}
        self.CF = {(int(row['j']), int(row['t'])): row['CF'] for index, row in df['CF'].iterrows()}
        self.CS = {(int(row['s']), int(row['t'])): row['CS'] for index, row in df['CS'].iterrows()}
        self.D = {int(row['t']): row['D'] for index, row in df['D'].iterrows()}

#np.random.seed(0)  # per ripetibilità test
#Capacity of supplier k in period t
#SK =  { (k,t): v for (k,t),v in zip (itertools.product(S,T), [100 for i in range(len(S)*len(T))] ) } # supplier capacity
#CJ =  { (k,t): v for (k,t),v in zip (itertools.product(F,T), [100 for i in range(len(F)*len(T))] )}# plant capacity

#class for variables
class Vars:
    def __init__(self):
        self.y = {}
        self.x = {}
        self.xbar = {}
        self.z = {}
        self.e = {}
        self.C = {}
        self.v = {}
        self.E = {}
    def initVars(self, y, x, xbar, z, e, C, v, E):
        self.y = y
        self.x = x
        self.xbar = xbar
        self.z = z
        self.e = e
        self.C = C
        self.v = v
        self.E = E

       

# class for follower variables
class FVars:
    def __init__(self):
        self.y = {}
        self.x = {}
        self.xbar = {}
        self.z = {}
        self.b = {}
        self.xi = {}
        self.e = {}
        self.C = {}
        self.v = {}
        self.Z1 = None
        self.Z2 = None
        self.E = {}
        self.Z_LB = None 
        self.lambdastar = 0
    def initVars(self, y, x, xbar, z, b, xi, e, C, v, Z1, Z2): 
        self.y = y
        self.x = x
        self.xbar = xbar
        self.z = z
        self.b = b
        self.xi = xi
        self.e = e
        self.C = C
        self.v = v
        self.Z1 = Z1
        self.Z2 = Z2
        self.E = {}
        self.Z_LB = None 
        self.lambdastar = 0
    def calcFollowerObj(self, dd : Data):
        Z1f = sum( dd.c_bar[j,l] * self.v[j,l,t].X for j in dd.F for l in dd.L for t in dd.T) 
        Z2f = sum(dd.c_hat[s,j]* self.xbar[s,j,t].X for s in dd.S for j in dd.F for t in dd.T)
        return Z1f, Z2f
    def calcZ1Leader(self, dd : Data):
        _E = {}
        for t in dd.T:
            _E[t] = sum(self.e[j,t].X for j in dd.F)
        Z1Leader = sum(_E[t] for t in dd.T)
        return Z1Leader
    def calcZ2Leader(self, dd : Data):
        Z2Leader = sum(self.b[j,0].X - sum (dd.alfa * dd.tc * self.z[j, jp, t].X for jp in dd.F for t in dd.T) for j in dd.F)
        return Z2Leader
    def calcLeaderObj(self, dd : Data):
        Z1 = self.calcZ1Leader(dd)
        Z2 = self.calcZ2Leader(dd)
        return Z1 + Z2

# class for leader variables
class LModel:  
    def __init__(self):
        self.B0 = {}
        self.E = {}
        self.C = {}
        self.B = 0
        self.Z1 = 0
        self.Z2 = 0
        self.ObjVal = 0
    def solve(self, fVars : FVars, dd : Data):
        for t in dd.T:
            _E = 0
            for j in dd.F:
                _E += fVars.e[j,t].X
            #for j in dd.F:
            #    for s in dd.S:
            #        _E += dd.e_hat[s,j]*fVars.xbar[s,j,t].X
            self.E[t] = _E
        self.Z1 = sum(self.E[t] for t in dd.T)
        self.Z2 = sum(self.B0[j] - sum (dd.alfa* dd.tc*fVars.z[j, jp, t].x for jp in dd.F for t in dd.T) for j in dd.F)
        self.ObjVal = self.Z1 + self.Z2
    def feasible(self, fVars : FVars, dd : Data):
        # controlla se la soluzione del leader è fattibile
        _E = {}
        for t in dd.T:
            _E[t] = sum(fVars.e[j,t].X for j in dd.F)  
            # + sum(dd.e_hat[s,j]*fVars.xbar[s,j,t].X for s in dd.S for j in dd.F)

        print('######CHECKING LEADER FEASIBILITY######')
        print ('_E ', _E)
        print('C ', fVars.C)
        print('E', fVars.E)
        #model.addConstrs(( C[t] == E[t] / dd.LF for t in dd.T), name = 'ctC-E')
        #model.addConstr(C[dd.T[-1]] <= (1-dd.delta)*C[dd.T[0]], name = 'ctCdelta')  #T[0] should be 1, while T[-1] is T #TODO check if correct
        #model.addConstrs((C[t] <= C[t-1] for t in dd.T if t != 1), name = 'Ct_Ct-1')

        if fVars.C[dd.T[-1]].X > (1-dd.delta)*fVars.C[dd.T[0]].X:
            return False
        for t in dd.T:
            if t != 1:
                if fVars.C[t].X > fVars.C[t-1].X:
                    return False
        for t in dd.T:
            if fVars.C[t].X*dd.LF < _E[t] :
                return False        
        return True
class Result:
    def __init__(self, lenS, lenF, seed, B, D, bestFound=0.0, worstFound=0.0, 
                    itBestFound=0, timeBestFound=0.0, bestUnfeasible=0.0,
                    worstUnfeasible=0.0, totalTime=0.0, maxSol=0.0) -> None:
        self.lenS : int = lenS
        self.lenF : int = lenF
        self.seed : int = seed
        self.B : int = B
        self.D : int = D
        self.bestFound : float = bestFound
        self.worstFound : float = worstFound
        self.itBestFound : int = itBestFound
        self.timeBestFound : float = timeBestFound
        self.bestUnfeasible : float = bestUnfeasible
        self.worstUnfeasible : float = worstUnfeasible
        self.totalTime : float = totalTime
        self.maxSol : float = maxSol
        self.improvement : float = 0

   
    def save (self, filename):
        df = pd.DataFrame(columns=['S', 'F', 'BD', 'B', 'D', 'seed', 'bestFound', 'worstFound', 
                          'itBestFound', 'timeBestFound', 'bestUnfeasible', 'worstUnfeasible', 'totalTime', 'maxSol', 'improvement'])
        df.loc[0] = [self.lenS, self.lenF, self.B/self.D, self.B, self.D, self.seed, self.bestFound, self.worstFound, 
                     self.itBestFound, self.timeBestFound, self.bestUnfeasible, self.worstUnfeasible, 
                     self.totalTime, self.maxSol, 100*(self.maxSol - self.bestFound)/self.maxSol]
        if os.path.isfile(filename):
            df.to_csv(filename, mode='a', index=None, header=False)
        else:
            df.to_csv(filename, index=None)
        return
    
def toExcelFollower(vars : FVars, model, filename, filenamelog, dd : Data):
    dfpar = pd.DataFrame(columns=['parametro', 'valore'])
    dfy = pd.DataFrame(columns=['j', 'l', 't', 'y'])
    dfx = pd.DataFrame(columns=['j', 't', 'x'])
    dfxbar = pd.DataFrame(columns=['j', 'jp', 't', 'xbar'])
    #dfb = pd.DataFrame(columns=['j', 't', 'b'])
    dfxi = pd.DataFrame(columns=['j', 't', 'xi'])
    dfe = pd.DataFrame(columns=['j', 't', 'e'])
    dfC = pd.DataFrame(columns=['t', 'C'])
    dfv = pd.DataFrame(columns=['j', 'l', 't', 'v'])
    dfz = pd.DataFrame(columns=['j', 'jp', 't', 'z'])
    dfb = pd.DataFrame(columns=['j', 't', 'b'])

    
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for k,v in vars.y.items():
            dfy.loc[len(dfy)] = [k[0], k[1], k[2], v.x]
        dfy.sort_values(by=['j', 'l', 't'], inplace=True)
        dfy.to_excel(writer, sheet_name='y', index=False)

        for k,v in vars.x.items():
            dfx.loc[len(dfx)] = [k[0], k[1], v.x]
        dfx.to_excel(writer, sheet_name='x', index=False)

        for k,v in vars.xbar.items():
            dfxbar.loc[len(dfxbar)] = [k[0], k[1], k[2], v.x]
        dfxbar.to_excel(writer, sheet_name='xbar', index=False)

        # for k,v in vars.b.items():
        #     dfb.loc[len(dfb)] = [k[0], k[1], v.x]
        # dfb.to_excel(writer, sheet_name='b', index=False)

        for k,v in vars.xi.items():
            dfxi.loc[len(dfxi)] = [k[0], k[1], v.x]
        dfxi.to_excel(writer, sheet_name='xi', index=False)

        for k,v in vars.e.items():
            dfe.loc[len(dfe)] = [k[0], k[1], v.x]
        dfe.to_excel(writer, sheet_name='e', index=False)

        for k,v in vars.C.items():
            dfC.loc[len(dfC)] = [k, v.x]
        dfC.to_excel(writer, sheet_name='C', index=False)

        for k,v in vars.v.items():
            dfv.loc[len(dfv)] = [k[0], k[1], k[2], v.x]
        dfv.to_excel(writer, sheet_name='v', index=False)
        for k,v in vars.z.items():
            dfz.loc[len(dfz)] = [k[0], k[1], k[2], v.x]
        dfz.to_excel(writer, sheet_name='z', index=False)
        for k,v in vars.b.items():
            dfb.loc[len(dfb)] = [k[0], k[1], v.x]
        dfb.to_excel(writer, sheet_name='b', index=False)

        dfpar.loc[len(dfpar)] = ['InstanceID', 0]
        dfpar.loc[len(dfpar)] = ['ModelType', 'Follower']
        dfpar.loc[len(dfpar)] = ['S', dd.LS]
        dfpar.loc[len(dfpar)] = ['F', dd.LF]
        dfpar.loc[len(dfpar)] = ['T', dd.LT]
        dfpar.loc[len(dfpar)] = ['L', dd.LL]
        dfpar.loc[len(dfpar)] = ['D', str(dd.D)]
        dfpar.loc[len(dfpar)] = ['B', dd.B]
        dfpar.loc[len(dfpar)] = ['CF', str(dd.CF)]
        dfpar.loc[len(dfpar)] = ['CS', str(dd.CS)]
        dfpar.loc[len(dfpar)] = ['e_bar', str(dd.e_bar)]
        dfpar.loc[len(dfpar)] = ['e_hat', str(dd.e_hat)]
        dfpar.loc[len(dfpar)] = ['c_bar', str(dd.c_bar)]
        dfpar.loc[len(dfpar)] = ['c_hat', str(dd.c_hat)]
        dfpar.loc[len(dfpar)] = ['tc', dd.tc]
        dfpar.loc[len(dfpar)] = ['delta', dd.delta]
        dfpar.loc[len(dfpar)] = ['alfa', dd.alfa]
        dfpar.loc[len(dfpar)] = ['alfa_prima', dd.alfa_prime]
        dfpar.loc[len(dfpar)] = ['k', str(dd.kappa)]
        dfpar.loc[len(dfpar)] = ['E0', dd.maxCO2()]

        dfpar.loc[len(dfpar)] = ['ObjVal', model.ObjVal]
        dfpar.loc[len(dfpar)] = ['Z1', vars.Z1.getValue()]
        dfpar.loc[len(dfpar)] = ['Z2', vars.Z2.getValue()]
        dfpar.loc[len(dfpar)] = ['MIPGap',  model.MIPGap]
        dfpar.loc[len(dfpar)] = ['RunTime', model.Runtime]

        dfpar.to_excel(writer, sheet_name='parameters', index=False)
        #writer.save()


    #writer = pd.ExcelWriter(path + '/' + filename, engine='xlsxwriter')
    df = pd.DataFrame(columns=['S', 'F', 'T', 'B', 'D', 'Z1', 'Z2',
                        'ObjVal', 'time', 'GAP'])
    z1Val = vars.Z1.getValue()
    z2Val = vars.Z2.getValue()
    objval =  model.ObjVal
    runtime =  model.Runtime
    df.loc[0] = [dd.LS, dd.LF, dd.LT, dd.B, dd.D, z1Val, z2Val, 
                        objval, runtime, model.MIPGap]

    if os.path.isfile(filenamelog):
            df.to_csv(filenamelog, mode='a', index=None, header=False)
    else:
        df.to_csv(filenamelog, index=None)

def toExcelLeader(lmodel : LModel, filename, filenamelog, dd : Data):
    
    dfpar = pd.DataFrame(columns=['parameter', 'valore'])
    dfB0 = pd.DataFrame(columns=['j', 'B0'])
    dfE = pd.DataFrame(columns=['t', 'E'])
    dfC = pd.DataFrame(columns=['t', 'C'])
   
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    for k,v in lmodel.B0.items():
        dfB0.loc[len(dfB0)] = [k, v]
    dfB0.to_excel(writer, sheet_name='B0', index=False)

    for k,v in lmodel.E.items():
        dfE.loc[len(dfE)] = [k, v]
    dfE.to_excel(writer, sheet_name='E', index=False)

    for k,v in lmodel.C.items():
        dfC.loc[len(dfC)] = [k, v]
    dfC.to_excel(writer, sheet_name='C', index=False)

    dfpar.loc[len(dfpar)] = ['InstanceID', 0]
    dfpar.loc[len(dfpar)] = ['ModelType', 'Leader']
    dfpar.loc[len(dfpar)] = ['S', dd.LS]
    dfpar.loc[len(dfpar)] = ['F', dd.LF]
    dfpar.loc[len(dfpar)] = ['T', dd.LT]
    dfpar.loc[len(dfpar)] = ['L', dd.LL]
    dfpar.loc[len(dfpar)] = ['D', str(dd.D)]
    dfpar.loc[len(dfpar)] = ['B', dd.B]
    dfpar.loc[len(dfpar)] = ['CF', str(dd.CF)]
    dfpar.loc[len(dfpar)] = ['CS', str(dd.CS)]
    dfpar.loc[len(dfpar)] = ['e_bar', str(dd.e_bar)]
    dfpar.loc[len(dfpar)] = ['e_hat', str(dd.e_hat)]
    dfpar.loc[len(dfpar)] = ['c_bar', str(dd.c_bar)]
    dfpar.loc[len(dfpar)] = ['c_hat', str(dd.c_hat)]
    dfpar.loc[len(dfpar)] = ['tc', dd.tc]
    dfpar.loc[len(dfpar)] = ['delta', dd.delta]
    dfpar.loc[len(dfpar)] = ['alfa', dd.alfa]
    dfpar.loc[len(dfpar)] = ['alfa_prima', dd.alfa_prime]
    dfpar.loc[len(dfpar)] = ['k', str(dd.kappa)]
    dfpar.loc[len(dfpar)] = ['E0', dd.maxCO2()]

    dfpar.loc[len(dfpar)] = ['ObjVal', lmodel.Z1 + lmodel.Z2]
    dfpar.loc[len(dfpar)] = ['Z1', lmodel.Z1]
    dfpar.loc[len(dfpar)] = ['Z2', lmodel.Z2]
 
    dfpar.to_excel(writer, sheet_name='parameters', index=False)
    writer.close()
    """
    df = pd.DataFrame(columns=['S', 'F', 'T', 'B', 'D', 'Z1', 'Z2',
                          'ObjVal', 'time', 'GAP'])
    z1Val = vars.Z1.getValue()
    z2Val = vars.Z2.getValue()
    objval =  model.ObjVal
    runtime =  model.Runtime
    df.loc[0] = [LS, LF, LT, B, D, z1Val, z2Val, 
                           objval, runtime, model.MIPGap]

    if os.path.isfile(path + '/' + filenamelog):
            df.to_csv(path + '/' + filenamelog, mode='a', index=None, header=False)
    else:
        df.to_csv(path + '/' + filenamelog, index=None)
    """

def getFollowerModel(lModel : LModel, dd : Data, checkfeasibility = False):
    model = gb.Model('cap-and-trade')
    vars = FVars()
    # VARIABLES
    y = {(j,l,t): model.addVar(vtype = gb.GRB.BINARY, name='y_{}_{}_{}'.format(j,l,t)) for j in dd.F for l in dd.L for t in dd.T0}
    x = {(j, t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='x_{}_{}'.format(j,t)) for j in dd.F for t in dd.T}
    xbar = {(s, j, t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='xbar_{}_{}_{}'.format(s,j,t)) for s in dd.S for j in dd.F for t in dd.T}
    z = {(j,jp,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='z_{}_{}_{}'.format(j,jp,t)) for j in dd.F for jp in dd.F for t in dd.T}
    b = {(j,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='b_{}_{}'.format(j,t)) for j in dd.F for t in dd.T0}
    #E = {t: model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name= 'E_{}'.format(t)) for t in T}
    xi = {(j,t) : model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name= 'k_{}_{}'.format(j,t)) for j in dd.F for t in dd.T}
    e = {(j,t) : model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name= 'e_{}_{}'.format(j,t)) for j in dd.F for t in dd.T}
    C = {t: model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name= 'C_{}'.format(t)) for t in dd.T}
    v = {(j,l,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name = 'v_{}_{}_{}'.format(j,l,t)) for j in dd.F for l in dd.L for t in dd.T}

    # OBJECTIVE FUNCTION
    Z1 = gb.quicksum(  dd.c_bar[j,l] * v[j,l,t] for j in dd.F for l in dd.L for t in dd.T) 
    Z2 = gb.quicksum(dd.c_hat[s,j]*xbar[s,j,t] for s in dd.S for j in dd.F for t in dd.T) #\
        #+ gb.quicksum(dd.alfa*dd.tc* gb.quicksum(z[jp,j,t] for jp in dd.F if j!=jp) for j in dd.F for t in dd.T)
    Z = Z1 + Z2
    if not checkfeasibility:
        model.setObjective(Z, gb.GRB.MINIMIZE)
    else:
        model.setObjective(0, gb.GRB.MINIMIZE)

    # SET leader vars
    for t in dd.T:
        C[t].ub = lModel.C[t]
        C[t].lb = lModel.C[t]
        #E[t].ub = lVars.E[t]
        #E[t].lb = lVars.E[t]
        
    for j in dd.F:
        b[j,0].ub = lModel.B0[j]
        b[j,0].lb = lModel.B0[j]

    #model.addConstr( y[1,2,1] == 1, name = 'ctY1F' )

    # CONSTRAINTS
    #model.addConstr( gb.quicksum(x[j,t] for j in dd.F for t in dd.T) == gb.quicksum(dd.D[t] for t in dd.T), name = 'ctDem'  )
    model.addConstrs( (gb.quicksum(x[j,t] for j in dd.F) == dd.D[t] for t in dd.T), name = 'ctDem'  )
    
    #model.addConstrs( (gb.quicksum(y[j,lp,t] for lp in range(1,l+2))
    #                  >= gb.quicksum(y[j,lp,t-1] for lp in range(1,l+1))
    #                    for j in dd.F for l in dd.L for t in dd.T if l <  dd.LF), name = 'ctY1')
    
    model.addConstrs((gb.quicksum(l*y[j,l,t] for l in dd.L) >= gb.quicksum(l*y[j,l,t-1] for l in dd.L) for j in dd.F for t in dd.T), name = 'ctY1')

    model.addConstrs(( gb.quicksum(y[j,l,t] for l in dd.L) == 1 for j in dd.F for t in dd.T0), name = 'ctY2')

    #model.addConstrs(( y[j,l,0] == (1 if l <= (j%dd.LL + 1) else 0) for l in dd.L  for j in dd.F ), name = 'ctY3' )
    model.addConstrs(( y[j,1,0] == 1  for j in dd.F), name = 'ctY3' )
    #model.addConstr( y[3,3,0] == 1  , name = 'ctY3f1' )

    model.addConstrs( ( e[j,t] == gb.quicksum(dd.e_bar[j,l]*v[j,l,t] for l in dd.L) 
                       - gb.quicksum(z[jp,j,t] for jp in dd.F if jp != j)
                        + gb.quicksum(z[j,jp,t] for jp in dd.F if jp != j) 
                        for j in dd.F for t in dd.T ) , name = 'cte')
    
    model.addConstrs((z[j,j,t] == 0 for j in dd.F for t in dd.T), name= 'ct_z_self')
    
    #model.addConstrs( (E[t] == gb.quicksum(e[j,t] for j in F) 
    #                   + gb.quicksum(e_hat[s,j,t]*xbar[s,j,t] for s in S for j in F) for t in T), name = 'ctE')

    model.addConstrs( (xi[j,t] == gb.quicksum(dd.kappa[j,l]*y[j,l,t] for l in dd.L)
                       - gb.quicksum(dd.kappa[j,lp]*y[j,lp,t-1] for lp in dd.L) for j in dd.F for t in dd.T ), name = 'ctk')
    model.addConstrs( (b[j,t] == b[j,t-1] - xi[j,t] - gb.quicksum(dd.tc*z[jp,j,t] for jp in dd.F) 
                       + gb.quicksum((1 - dd.alfa - dd.alfa_prime)*dd.tc*z[j,jp,t] for jp in dd.F ) for t in dd.T for j in dd.F ), name = 'ct_b' )
    model.addConstrs( (gb.quicksum(xi[j,tau] for tau in range(t+1, dd.LT+1)) >=
                        gb.quicksum(dd.alfa_prime*dd.tc*z[j,jp,t] for jp in dd.F) for j in dd.F for t in dd.T ),  name = 'ct_xi2')
    model.addConstrs( (x[j,t] == gb.quicksum(xbar[s,j,t] for s in dd.S) for j in dd.F for t in dd.T), name = 'ct_x')
    model.addConstrs((gb.quicksum(xbar[s,j,t] for j in dd.F) <= dd.CS[s,t] for s in dd.S for t in dd.T), name = 'ct_xbar')
    model.addConstrs((x[j,t] <= dd.CF[j,t] for j in dd.F for t in dd.T), name = 'ct_maxCJ')
    model.addConstrs( (e[j,t]  <= C[t] for j in dd.F for t in dd.T), name = 'ct_max_e-C' )
                    #+ gb.quicksum(dd.e_hat[s,j]*xbar[s,j,t] for s in dd.S)
    model.addConstrs((1 - y[j,l,t] <= 1 - v[j,l,t]/dd.CF[j,t] for j in dd.F for l in dd.L for t in dd.T), name= 'ct_v_1')
    model.addConstrs((v[j,l,t] <= x[j,t] for j in dd.F for l in dd.L for t in dd.T), name = 'ct_v_2')
    model.addConstrs((v[j,l,t] >= -dd.CF[j,t]*(1 - y[j,l,t]) + x[j,t] for j in dd.F for l in dd.L for t in dd.T), name = 'ct_v_2bis')

    # Load solution generated by lower bound model and set variables x, z, y to solution values

#    if LOADSOLUTION:
#        x_load, z_load, y_load, _, _ = loadSolExcel('results/' + flienameToLoad.format( "True", str(BINARY),
#                                            "False", str(LS), str(LF), str(LT), str(B), str(D), str(phi_hat_0), 
#                                            str(phi_bar_0), str(rho_bar_0), str(eta), str(alpha), '1'))
#        for k, v in z_load.items():
#            z[k[0], k[1]].lb = v
#            z[k[0], k[1]].ub = v

    model.update()
    vars.initVars(y,x,xbar,z,b,xi,e,C,v,Z1,Z2)
    return vars, model
