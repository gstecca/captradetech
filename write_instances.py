# Copyright (c) 2025 Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Authors: Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Email: giuseppe.stecca@gmail.com

#from algorithms.bilevel_params import *
import pandas as pd
import json


def write_instences():

    df = pd.read_csv('instances_bilevel.csv', index_col='id', delimiter=',')
    for index, row in df.iterrows():
        id = index
        print(f"writing to excel instance {id}")
        dd = Data(id, False, 'instances_bilevel.csv')
        if row['DeltaRandom']:
            dd.randomize()
        dd.to_excel(f"instances/{id}.xlsx")

def generate_big_instances():
            instance_file = ""
            inst_id_template = "IR_B{}_D{}_F{}_L{}_T_{}_seed_{}"
            Bs = [3000, 5000, 7000, 10000] # budget
            LFs = [ 3, 6, 9, 12] # number of plants
            LLs = [4, 8, 12] # number of technologies
            LTs = [5, 10, 15] # time horizon
            for B, LF, LL, LT, seed in itertools.product(Bs, LFs, LLs, LTs, range(0, 5)):
                random.seed(seed)
                D_avg = int(B / LF)
                insT_id = inst_id_template.format(B, D_avg, LF, LL, LT, seed)
                dd = Data()
                dd.ins_id = insT_id
                dd.DeltaRandom = 0.1
                dd.seed = seed
                dd.DDecrease = False
                dd.delta = 0.7  # decrease of cap over time

                dd.eta = 0.5
                dd.Lmin = 1
                dd.Bmin = 1
                dd.LF = LF # number of plants
                dd.LT = LT # time horizon
                dd.LL = LL # number of technologies
                dd.LS = LF # number of suppliers
                dd.T = [t for t in range(1,dd.LT+1)] # Time horizon
                dd.T0 = [t for t in range(0,dd.LT+1)] # Time horizon
                dd.S = [i for i in range(1,dd.LS+1)]
                dd.F = [j for j in range(1,dd.LF+1)]
                dd.L = [l for l in range(1, dd.LL+1)]
                dd.A = [(i,j) for (i,j) in itertools.product(dd.S,dd.F) ]
                
                #Pint('cbar ', c_bar)
                dd.c_hat = {(s,j): 5 for s in dd.S for j in dd.F}
                dd.tc = 0.05
                dd.D = {t:B/LF for t in dd.T}
                dd.B = B   #1000*dd.LF # kEur
                dd.alfa = 0.2
                dd.alfa_prime = 0.5

                kaverage = 2*B/(LF*LT) # average installation cost for each technology
                #dd.kappa = {(j,l): (dd.B/(dd.LF * dd.LL))*(1 + math.log(l)) for j in dd.F for l in dd.L }
                dd.kappa = {(j,l): kaverage for j in dd.F for l in dd.L } # average installation cost for each technology
                dd.kappa.update({(j, 1): 0 for j in dd.F})  # no installation cost for technology 1


                #TODO aggiornare CF
                dd.CF = {(j,t) : int((dd.D[t]/(dd.LF))*1.3) for j in dd.F for t in dd.T}
                dd.CS = {(s,t) : int((dd.D[t]/dd.LS)*2) for s in dd.S for t in dd.T}
                print('###### CF #####')
                print(dd.CF)#PARAMETERS
                #dd.e_bar = {(j,l): 10 - (2/dd.LL)*math.log(l) for j in dd.F for l in dd.L}
                #dd.c_bar = {(j,l): 20 + 10*math.log(l) for j in dd.F for l in dd.L}

                e_bar_max = 200
                e_bar_min = 0
                e_hat_avg = 1
                c_bar_max = 100
                c_bar_min = 10
                c_hat_avg = 5

                dd.e_bar = {(j,l): int((LL - l)*((e_bar_max - e_bar_min)/(LL-1)) + e_bar_min) for j in dd.F for l in dd.L} # e_bar_max per la tecnologia 1 e 0 per la tecnologia LL
                #dd.c_bar = {(j,l): 20 + 10*math.log(l) for j in dd.F for l in dd.L}
                #print('ebar ', e_bar)
                dd.e_hat = {(s,j): e_hat_avg for s in dd.S for j in dd.F}
                dd.c_bar = {(j,l): int((l - l)*((c_bar_max - c_bar_min)/(LL-1)) + c_bar_min) for j in dd.F for l in dd.L}  # c_bar_min per la tecnologia 1 e c_bar_max per la tecnologia LL
                dd.c_hat = {(s,j): c_hat_avg for s in dd.S for j in dd.F}

                ####
                #### RANDOMIZING PARAMETERS
                ####
                # For each facility 
                #    with a probability 0.25 increase the installation cost k of the technology by shift%, increase the operating cost cbar by shift%
                #    with a probability 0.25 decrease the cost of the technology by shift%, decrease the operating cost cbar by shift%
                shift = 0.3
                for j in dd.F:
                    for l in dd.L:
                        if random.random() < 0.25:
                            if l != 1:
                                dd.kappa[(j,l)] = int(dd.kappa[(j,l)] * (1 + shift))
                            dd.c_bar[(j,l)] = int(dd.c_bar[(j,l)] * (1 + shift))
                        elif random.random() < 0.5:
                            if l != 1:
                                dd.kappa[(j,l)] = int(dd.kappa[(j,l)] * (1 - shift))
                            dd.c_bar[(j,l)] = int(dd.c_bar[(j,l)] * (1 - shift))

                #randomize demand, budget, installation cost kappa, operating cost cbar, emissions e_bar, by rand_factor 
                rand_factor = 0.15
                #dd.B *= (1 + random.uniform(-rand_factor, rand_factor))
                for t in dd.T:
                        dd.D[t] = int(dd.D[t] * (1 + random.uniform(-rand_factor, rand_factor)))
                for j in dd.F:

                    for l in dd.L:
                        dd.kappa[(j,l)] = int(dd.kappa[(j,l)] * (1 + random.uniform(-rand_factor, rand_factor)))
                        dd.c_bar[(j,l)] = int(dd.c_bar[(j,l)] * (1 + random.uniform(-rand_factor, rand_factor)))
                        dd.e_bar[(j,l)] = int(dd.e_bar[(j,l)] * (1 + random.uniform(-rand_factor, rand_factor)))
                        
                    for t in dd.T:
                        dd.CF[(j,t)] = int(dd.CF[(j,t)] * (1 + random.uniform(-rand_factor, rand_factor)))

                for s in dd.S:
                    for t in dd.T:
                        dd.CS[(s,t)] = int(dd.CS[(s,t)] * (1 + random.uniform(-rand_factor, rand_factor)))
                    for j in dd.F:
                        dd.c_hat[(s,j)] = int(dd.c_hat[(s,j)] * (1 + random.uniform(-rand_factor, rand_factor)))

                print(f"writing to excel instance {insT_id}")
                dd.to_excel(f"instances_IR/{insT_id}.xlsx")
if __name__ == "__main__":
     generate_big_instances()
            #pr
