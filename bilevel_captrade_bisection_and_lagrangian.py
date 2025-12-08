# Copyright (c) 2025 Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Authors: Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Email: giuseppe.stecca@gmail.com
import gurobipy as gb
import numpy as np
import random
from numpy.linalg import norm
import sys
import json
import time
import pandas as pd
import os
import math
from bilevel_params import *
from ngsaii import *

def loadSol(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        x = {(a['i'],a['j']):a['v'] for a  in data['solution'][0]['x'] }
        z = {a['i']:a['v'] for a  in data['solution'][1]['z'] }
        return x, z

def loadSolExcel(filename):

    dfx = pd.read_excel(filename, sheet_name='x')
    dfz = pd.read_excel(filename, sheet_name='z')
    dfphi = pd.read_excel(filename, sheet_name='phi')
    dfrho = pd.read_excel(filename, sheet_name='rho')
    dfy = pd.read_excel(filename, sheet_name='y')


    x_load = {(row['k'], row['j'], row['t']): row['x'] for index, row in dfx.iterrows() }
    z_load = {(row['j'], row['t']): row['z'] for index, row in dfz.iterrows() }
    phi_load = {(row['j'], row['t']): row['phi'] for index, row in dfphi.iterrows() }
    rho_load = {(row['t']): row['rho'] for index, row in dfrho.iterrows() }
    y_load = {(row['j'], row['t']): row['y'] for index, row in dfy.iterrows() }
    return x_load, z_load, y_load, phi_load, rho_load 

def toJson(x, z, filename):
    solout = {'solution':[]}
    solout['solution'].append( {'x': [ {'i':k[0], 'j':k[1], 'v':v} for k,v in x.items() ] } )
    solout['solution'].append( {'z' : [ {'i': k, 'v':v} for k,v in z.items()] } )
    with open (filename, 'w') as xout:
        json.dump(solout, xout, indent=4)
        xout.close()

def toExcelLagrangian(vars : FVars, model, filename, filenamelog, dd : Data):
    dfpar = pd.DataFrame(columns=['parametro', 'valore'])
    dfy = pd.DataFrame(columns=['j', 'l', 't', 'y'])
    dfx = pd.DataFrame(columns=['j', 't', 'x'])
    dfxbar = pd.DataFrame(columns=['j', 'jp', 't', 'xbar'])
    #dfb = pd.DataFrame(columns=['j', 't', 'b'])
    dfxi = pd.DataFrame(columns=['j', 't', 'xi'])
    dfe = pd.DataFrame(columns=['j', 't', 'e'])
    dfC = pd.DataFrame(columns=['t', 'C'])
    dfE = pd.DataFrame(columns=['t', 'E'])
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

        for k,v in vars.E.items():
            dfE.loc[len(dfE)] = [k, v.x]
        dfE.to_excel(writer, sheet_name='EE', index=False)

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
        dfpar.loc[len(dfpar)] = ['Zlgr', vars.Z_LB.getValue()]
        dfpar.loc[len(dfpar)] = ['MIPGap',  model.MIPGap]
        dfpar.loc[len(dfpar)] = ['RunTime', model.Runtime]
        dfpar.loc[len(dfpar)] = ['lambdastar', vars.lambdastar]

        dfpar.to_excel(writer, sheet_name='parameters', index=False)
        #writer.save()


    #writer = pd.ExcelWriter(path + '/' + filename, engine='xlsxwriter')
    #logfile
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

def get_budget_patterns(dd, epsilon = 0):
    #* Generate budget patterns for the leader's budget allocation.
    # epsilon = 0
    #     #1. Equal distribution of budget across all followers.
    #     #2. Random distribution of budget across all followers.
    #     #3. Random distribution of budget across followers, with decreasing budget for each subsequent follower.
    #     #4. Random distribution of budget across followers, with increasing budget for each subsequent follower.
    # epsilon > 0
    #     #5. Random distribution of budget across followers, with decreasing budget for each subsequent follower, and a minimum budget for the first follower.
   
    patterns = { i : [] for i in range(1, 6) } 
    for p in range(1, 6):
        # static pattern
        if p == 1:
            patterns[p] = [dd.B / dd.LF for i in dd.F]
        # random pattern
        elif p == 2:
            b = [0 for i in dd.F]
            for j in dd.F:
                b0j = random.random()*(dd.B / dd.LF)
                sumbj = sum(b) + b0j
                if sumbj > dd.B:
                    b0j = dd.B - sum(b)
                b[j-1] = b0j
            # Alcuni budget potrebbero essere negativi, in tal caso li mettiamo a zero
            b = [max(0, b[j-1]) for j in dd.F]
            sumb = sum(b)
            if sumb < dd.B:
                for j in dd.F:
                    b[j-1] = b[j-1] + (dd.B - sumb) / dd.LF
            patterns[p] = b
        # decreasing pattern
        elif p == 3:
            b = [0 for i in dd.F]
            for j in dd.F:
                b0j = random.random()*(dd.B / (dd.LF-j+1))
                sumbj = sum(b) + b0j
                if sumbj > dd.B:
                    b0j = dd.B - sum(b)
                b[j-1] = b0j
            # Alcuni budget potrebbero essere negativi, in tal caso li mettiamo a zero
            b = [max(0, b[j-1]) for j in dd.F]
            sumb = sum(b)
            if sumb < dd.B:
                for j in dd.F:
                    b[j-1] = b[j-1] + (dd.B - sumb) / dd.LF
            patterns[p] = b
        # increasing pattern
        elif p == 4:
            b = [0 for i in dd.F]
            for j in dd.F:
                b0j = random.random()*(dd.B / (j+1))
                sumbj = sum(b) + b0j
                if sumbj > dd.B:
                    b0j = dd.B - sum(b)
                b[j-1] = b0j
            patterns[p] = b
            # Alcuni budget potrebbero essere negativi, in tal caso li mettiamo a zero
            b = [max(0, b[j-1]) for j in dd.F]
            sumb = sum(b)
            if sumb < dd.B:
                for j in dd.F:
                    b[j-1] = b[j-1] + (dd.B - sumb) / dd.LF
            patterns[p] = b
        # decreasing pattern with epsilon as delta budget to add to the current budget after ordering
        elif p == 5:
            b = []
            for j in dd.F:
                db = (epsilon /(dd.LF -1)) * ( -2*j + dd.LF + 1)
                b.append(db)
            patterns[p] = b
    return patterns

def get_shake_pattern(dd, currentBudget : dict, epsilon = 0.1):
    #* Generate shake pattern for the leader's budget allocation.
    
    deltab = [0 for i in dd.F]
    for j in dd.F:
        db = (2*(random.random() - 0.5))
        db = db* epsilon*(dd.B / dd.LF)
        deltab[j-1] = db
    sumb = 0
    for j in dd.F:
        currentBudget[j] = currentBudget[j] + deltab[j-1]
        if currentBudget[j] < 0:
            currentBudget[j] = - currentBudget[j]
        sumb += currentBudget[j]
        # check if the sum of the budgets is less than the total budget
        if sumb > dd.B:
            currentBudget[j] = dd.B - sumb
    # Alcuni budget potrebbero essere negativi, in tal caso li mettiamo a zero
    currentBudget = {j: max(0, currentBudget[j]) for j in dd.F}
    sumb = sum(currentBudget.values())
    if sumb < dd.B:
        for j in dd.F:
            currentBudget[j] = currentBudget[j] + (dd.B - sumb) / dd.LF
    return currentBudget

def bisectionHeuristics(params : dict):
    """
    
    Heuristic approach based on budget patterns and local search.
    The leader generates budget patterns and solves the follower problem.
    In the paper the heuristic is named IRIS
    @params: dictionary of parameters
    @return: best solution found
    """
    TLIM = params.get('maxtime')
    idI = params.get('instancename') #'I_0' # is instance

    #max_c_e_iniz = dd.B /dd.E_min_sum
    #max_c_cong_iniz = dd.B / dd.G_sat_min_sum
    dd = Data(params, idI, params.get('INSTANCE_FROM_FILE'), 'instances_bilevel.csv')
    if params.get('params_from_params_file'):
        dd.delta = params.get('delta')
        dd.B = params.get('B')
    E0 = dd.maxCO2()
    Emax = E0
    Emin = dd.minCO2()
    #DeltaB = [1, 1.25, 1.5, 1.75, 2] #[0.5, 0.75, 1, 1.25, 1.5,5]
    #DeltaB = [5,10,15,20]
    #DeltaB = [1]

    list_lObj = {0: float('inf')}
    list_fObj = {0: float('inf')}
    list_lZ1s = {0: float('inf')}
    list_lZ2s = {0: float('inf')}
    list_runtime = {0: float('inf')}
    list_fgap = {0: float('inf')}
    iteration = 0
    iterationE = 0 # iteration for the bisection on emissions
    maxIterationsE = 5 # max iterations for the bisection on emissions
    maxIterations = params.get('maxIterations')
    epsilon_ratio = params.get('epsilon_ratio')
    epsilon = epsilon_ratio * dd.B / dd.LF
    deltaE = [0 for j in dd.F]
    currentE = [0 for j in dd.F]
    bestB0 = {j:0 for j in dd.F}
    currentB0 = {j:0 for j in dd.F}
    best_it = 0
    n_it_not_improved = 0
    max_it_not_improved = int(maxIterations / 5)
    best_lobjval = float('inf')
    improvementHeuristic = 0 # improvement of the euristic solution with respect to the initial solution
    firstSolutionValue = 0
    patterns = get_budget_patterns(dd, epsilon)
    df_track_results = pd.DataFrame(columns=['iteration', 'bestlobjval', 'lobjval', 'fobjval', 'lz1', 'lz2', 'tot_runtime', 'gapf'])
    tot_runtime = time.time()
    maxtimetotal = params.get('maxtimetotal')

    Ecurrent = Emin
    E0 = Emax
    lObjValBest = float('inf')
    followerFeasible = False
    #maxIterationsE = 0 #debug
    patterns_ok = False
    while True: 
        while iterationE < maxIterationsE:
            iterationE += 1
            print("###############################")
            print("##########EMISSIONS BISECTION ITERATION=", iterationE)
            print("###############################")
            # set E0 on bisection
            pattern = patterns[1]
            lmodel = LModel() # initialize leader model
            lmodel.B0 = {j : pattern[j-1] for j in dd.F}
            lmodel.C = {t: (Ecurrent/dd.LF) *(1-(dd.delta/(dd.LT-1))*(t-1)) for t in dd.T}

            fVars, fmodel = getFollowerModel(lmodel, dd, checkfeasibility=True)

            fmodel.params.TimeLimit = 100

            fmodel.optimize()
            if fmodel.Status not in [2,9,11] or fmodel.ObjVal == float('inf'):
                followerFeasible = False
                print(f'######EMISSION BISECTION: FOLLOWER MODEL CANNOT BE SOLVED######__iteration {iteration} pattern {pattern} Status {fmodel.Status} ######')
                Ecurrent = (Emax + Ecurrent)/2
            else:
                followerFeasible = True
                lmodel.solve(fVars,dd)
                lfeasible = lmodel.feasible(fVars, dd)
                if not lfeasible:
                    print(f'######EMISSION BISECTION: LEADER SOLUTION INFEASIBLE######__iteration {iteration} pattern {pattern} Status {fmodel.Status} ######')
                    Ecurrent = (Emax + Ecurrent)/2
                else:
                    # leader and follower feasible -> save E0, search for a lower E0
                    if lmodel.ObjVal < lObjValBest:
                        lObjValBest = lmodel.ObjVal
                        E0 = Ecurrent
                        print(f'######EMISSION BISECTION: NEW BEST LEADER OBJVAL FOUND {lObjValBest}, E0 {E0} ######__iteration {iteration} pattern {pattern} Status {fmodel.Status} ######')

                    Ecurrent = (Ecurrent + Emin)/2
                    print(f'######EMISSION BISECTION: FOLLOWER MODEL SOLVED######__iteration {iteration} pattern {pattern} Status {fmodel.Status} ######')
                    print('######LEADER FEASIBILITY######')
                    print(lfeasible)
                    print('######OTIMAL LEADER OBJECTIVE FUNCTION VALUE######')
                    print(lmodel.ObjVal)
        if not followerFeasible and not patterns_ok:
            print('######NO FEASIBLE SOLUTION FOUND BY EMISSIONS BISECTION######')
            E0 = Emax
            print("updating patterns[1] in order to allocate all budget to half of the followers")
            randomset = random.sample(range(1, dd.LF + 1), dd.LF // 2)
            patterns[1] = [dd.B / (dd.LF/2) if  i in randomset else 0 for i in dd.F if i <= dd.LF]
            print(f'######NEW PATTERN[1]: {patterns[1]} ######')
            iterationE = 0
            patterns_ok = True
        else:
            patterns_ok = True
            break
        
    print(f'######FINAL E0 SET FOR HEURISTIC: {E0} ######')
    print(f'######FINAL BEST LEADER OBJVAL: {lObjValBest} ######')
    print("################IMPROVEMENT HEURISTIC STARTS#######################")
    while iteration < maxIterations and (time.time() - tot_runtime) < maxtimetotal:
        iteration += 1
        n_it_not_improved += 1
        lmodel = LModel() # initialize leader model
        if iteration < 5:
            pattern = patterns[iteration]
            lmodel.B0 = {j : pattern[j-1] for j in dd.F}

        else:
            if n_it_not_improved >= max_it_not_improved:
                n_it_not_improved = 0
                # shake pattern
                currentB0 = get_shake_pattern(dd, currentB0, 0.1)
                lmodel.B0 = currentB0
            else:
                pattern = patterns[5]
                # sort followers by deltaE. The one with the highest decrease in emissions will be the on to receive the highest budget
                deltaE_dict = {j : deltaE[j-1] for j in dd.F}
                sorted_deltaE = sorted(deltaE_dict.items(), key=lambda item: item[1])
                sumb = 0
                epsilon_index = 0
                lmodel.B0 = currentB0
                for kv in sorted_deltaE:
                    j = kv[0]
                    lmodel.B0[j] = lmodel.B0[j] + pattern[epsilon_index]
                    if lmodel.B0[j] < 0:
                        lmodel.B0[j] = - lmodel.B0[j]
                    epsilon_index += 1
                    sumb += lmodel.B0[j]
                    # check if the sum of the budgets is less than the total budget
                    if sumb > dd.B:
                        lmodel.B0[j] = dd.B - sumb
                # Alcuni budget potrebbero essere negativi, in tal caso li mettiamo a zero
                lmodel.B0 = {j: max(0, lmodel.B0[j]) for j in dd.F}
                sumb = sum(lmodel.B0.values())
                if sumb < dd.B:
                    for j in dd.F:
                        lmodel.B0[j] = lmodel.B0[j] + (dd.B - sumb) / dd.LF

                    

            
        print("###############################")
        print(f"##########ITERATION pattern= {iteration}  -- time elapsed = {time.time() - tot_runtime} / {maxtimetotal} ##########")
        print("###############################")
        print("pattern=", pattern)
        print("###############################")
        

        # SET Leader Variables

        lmodel.C = {t: E0 *(1-(dd.delta/(dd.LT-1))*(t-1)) for t in dd.T}

        fVars, fmodel = getFollowerModel(lmodel, dd, False)

        fmodel.params.TimeLimit = TLIM

        #fmodel.write('model_follower_'+sdB+'.lp')
        path = params.get('folder_output')
        fmodel.optimize()
        if fmodel.Status not in [2,9,11] or fmodel.MIPGap == float('inf'):
            print(f'######FOLLOWER MODEL CANNOT BE SOLVED######__iteration {iteration} pattern {pattern} Status {fmodel.Status} ######')
            #fmodel.write(f' {}_{}.lp'.format(idI, kp))
            #fmodel.computeIIS()
            #fmodel.write(f'{path}/followerModel_{idI}_{iteration}_delta{dd.delta}.ilp')
            print(lmodel.B0)
            print(dd.B)
            continue

        print(f'######FOLLOWER MODEL SOLVED######__iteration {iteration} pattern {pattern} Status {fmodel.Status} ######')
        print('######OTIMAL FOLLOWER OBJECTIVE FUNCTION VALUE######')
        print(fmodel.ObjVal)
        print(f"runtime = {fmodel.Runtime}")
        print(f"gap = {fmodel.MIPGap}")
        list_fObj[iteration] = fmodel.ObjVal
        list_runtime[iteration] = fmodel.Runtime
        list_fgap[iteration] = fmodel.MIPGap

        lmodel.solve(fVars,dd)
        print('######OTIMAL LEADER OBJECTIVE FUNCTION VALUE######')
        print(lmodel.ObjVal)
        list_lObj[iteration] = lmodel.ObjVal
        list_lZ1s[iteration] = lmodel.Z1
        list_lZ2s[iteration] = lmodel.Z2

        # updating deltaE for keep track of the increase / decrease of emissions for each follower
        for j in dd.F:
            _e = [fVars.e[j, t].x for t in dd.T]
            sumE = sum(_e)
            deltaE[j-1] = sumE - currentE[j-1]
            currentE[j-1] = sumE
        currentB0 = lmodel.B0

        if iteration == 1:
            firstSolutionValue = lmodel.ObjVal

        if lmodel.ObjVal < best_lobjval:
            print(f'###### ITERATION {iteration} - NEW BEST LEADER OBJECTIVE FUNCTION VALUE {lmodel.ObjVal} ######')
            best_lobjval = lmodel.ObjVal
            best_it = iteration
            n_it_not_improved = 0
            bestB0 = lmodel.B0

            
            path = params.get('folder_output')
            print('writing follower results to file...')
            toExcelFollower(fVars, fmodel, f'{path}/follower_results_{idI}_{iteration}_delta{dd.delta}.xlsx', f'{path}/follower_log_{idI}_{iteration}.csv', dd)
            print('...done.')
            print('writing leader results to file...')
            toExcelLeader(lmodel, f"{path}/leader_results_{idI}_{iteration}.xlsx", f"leader_log_{idI}_{iteration}.xlsx", dd)
            print('...done.')

        print('...done iteration ', iteration)
        df_track_results.loc[len(df_track_results)] = [iteration, best_lobjval, lmodel.ObjVal, fmodel.ObjVal, lmodel.Z1, lmodel.Z2, fmodel.Runtime, fmodel.MIPGap]

    tot_runtime = time.time() - tot_runtime
    try:
        improvementHeuristic = 100*(firstSolutionValue - best_lobjval) / firstSolutionValue
    except ZeroDivisionError:
        print("First solution value is zero (NOT FEASIBLE), cannot compute improvement.")
        improvementHeuristic = -1
    print('Total runtime = ', tot_runtime)
    print(f'BEST IT ----------->{best_it}.---------------All iterations completed.')
    df_track_results.to_csv(f"{path}/results_track_it_{idI}.csv", index=None)

    

    results = {'lobjval' : best_lobjval, 'fobjval' : list_fObj[best_it], 'lz1' : list_lZ1s[best_it], 'lz2' : list_lZ2s[best_it], 'tot_runtime' : tot_runtime,
               'gapf': list_fgap[best_it], 'best_it': best_it, 'improvementHeuristic' : improvementHeuristic}
               #'list_lObj' : list_lObj, 'list_fObj' :list_fObj, 'list_lZ1s' : list_lZ1s, 'list_lZ2s': list_lZ2s, 'list_runtime' : list_runtime, 'improvementHeuristic' : improvementHeuristic}

    dfall = pd.DataFrame(results.items(), columns = ["attr", "value"])
    dfall.to_csv(f"{path}/results_all_{idI}.csv", index=None)
    return results

def getLagrangianModel(dd : Data, lgr_lambda: float = 1000.0, upper_bound : bool = False):
    model = gb.Model('cap-and-trade-lagrangian')
    vars = FVars()
    # VARIABLES

    E = {t: model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name= 'E_{}'.format(t)) for t in dd.T}
    C = {t: model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name= 'C_{}'.format(t)) for t in dd.T}


    y = {(j,l,t): model.addVar(vtype = gb.GRB.BINARY, name='y_{}_{}_{}'.format(j,l,t)) for j in dd.F for l in dd.L for t in dd.T0}
    x = {(j, t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='x_{}_{}'.format(j,t)) for j in dd.F for t in dd.T}
    xbar = {(s, j, t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='xbar_{}_{}_{}'.format(s,j,t)) for s in dd.S for j in dd.F for t in dd.T}
    z = {(j,jp,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='z_{}_{}_{}'.format(j,jp,t)) for j in dd.F for jp in dd.F for t in dd.T}
    b = {(j,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='b_{}_{}'.format(j,t)) for j in dd.F for t in dd.T0}
    xi = {(j,t) : model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name= 'k_{}_{}'.format(j,t)) for j in dd.F for t in dd.T}
    e = {(j,t) : model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name= 'e_{}_{}'.format(j,t)) for j in dd.F for t in dd.T}
    v = {(j,l,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name = 'v_{}_{}_{}'.format(j,l,t)) for j in dd.F for l in dd.L for t in dd.T}

    Z_LB = lgr_lambda * (gb.quicksum(b[j,0] for j in dd.F) - dd.B )
    if upper_bound:
        # OBJECTIVE FUNCTION Follower
        Z1 = gb.quicksum(  dd.c_bar[j,l] * v[j,l,t] for j in dd.F for l in dd.L for t in dd.T) 
        Z2 = gb.quicksum(dd.c_hat[s,j]*xbar[s,j,t] for s in dd.S for j in dd.F for t in dd.T) #\
        #+ gb.quicksum(dd.alfa*dd.tc* gb.quicksum(z[jp,j,t] for jp in dd.F if j!=jp) for j in dd.F for t in dd.T)
        Z = Z1 + Z2
    else:
        # OBJECTIVE FUNCTION Leader
        Z1 = gb.quicksum(E[t] for t in dd.T)
        Z2 = gb.quicksum(b[j,0] - sum (dd.alfa * dd.tc * z[j, jp, t] for jp in dd.F for t in dd.T) for j in dd.F)
        Z = Z1 + Z2 + Z_LB

    model.setObjective(Z, gb.GRB.MINIMIZE)


    # LEADER CONSTRAINTS
    model.addConstrs(( E[t] == 
                      gb.quicksum(e[j,t] for j in dd.F) for t in dd.T), name = 'ctE') 
            #+ gb.quicksum(dd.e_hat[s,j]*xbar[s,j,t] for s in dd.S for j in dd.F) 
    model.addConstrs(( E[t] <= C[t] * dd.LF for t in dd.T), name = 'ctC-E')
    model.addConstr(C[dd.T[-1]] <= (1-dd.delta)*C[dd.T[0]], name = 'ctCdelta')  #T[0] should be 1, while T[-1] is T #TODO check if correct
    model.addConstrs((C[t] <= C[t-1] for t in dd.T if t != 1), name = 'Ct_Ct-1')


    # FOLLOWER CONSTRAINTS
    #model.addConstr( gb.quicksum(x[j,t] for j in dd.F for t in dd.T) == gb.quicksum(dd.D[t] for t in dd.T), name = 'ctDem'  )
    model.addConstrs( (gb.quicksum(x[j,t] for j in dd.F) == dd.D[t] for t in dd.T), name = 'ctDem'  )
    
    #model.addConstrs( (gb.quicksum(y[j,lp,t] for lp in range(1,l+2))
    #                  >= gb.quicksum(y[j,lp,t-1] for lp in range(1,l+1))
    #                    for j in dd.F for l in dd.L for t in dd.T if l <  dd.LF), name = 'ctY1')

    model.addConstr(gb.quicksum(b[j,0] for j in dd.F) <= dd.B, name = 'ctBudgetTotal')
    
    model.addConstrs((gb.quicksum(l*y[j,l,t] for l in dd.L) >= gb.quicksum(l*y[j,l,t-1] for l in dd.L) for j in dd.F for t in dd.T), name = 'ctY1')

    model.addConstrs(( gb.quicksum(y[j,l,t] for l in dd.L) == 1 for j in dd.F for t in dd.T0), name = 'ctY2')

    #model.addConstrs(( y[j,l,0] == (1 if l <= (j%dd.LL + 1) else 0) for l in dd.L  for j in dd.F ), name = 'ctY3' )
    model.addConstrs(( y[j,1,0] == 1  for j in dd.F), name = 'ctY3' )
    #model.addConstr( y[3,3,0] == 1  , name = 'ctY3f1' )


    model.addConstrs( ( e[j,t] == gb.quicksum(dd.e_bar[j,l]*v[j,l,t] for l in dd.L) 
                       - gb.quicksum(z[jp,j,t] for jp in dd.F if jp != j)
                        + gb.quicksum(z[j,jp,t] for jp in dd.F if jp != j) 
                         for j in dd.F for t in dd.T ) , name = 'cte')
    
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
    model.addConstrs( (e[j,t] <= C[t] for j in dd.F for t in dd.T), name = 'ct_max_e-C' )
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
    vars.Z_LB = Z_LB
    vars.E = E
    return vars, model

def upper_bound(params):
    return lagrangian(params, upper_bound = True)

def lower_bound_improved(params):
    """
    # Calcolo del Lower Bound Improved

    Calcola \(\varepsilon\) con:
    \[
    \varepsilon = \frac{\text{UB}^{\text{RIS}} - \text{LB}^0}{N}
    \]

    dove \(N\) è il numero di iterazioni massime.

    \[
    \text{LB}^i = \text{LB}^0 + \varepsilon
    \]

    ---

    ### Definizioni:
    - \(f_L\): funzione obiettivo leader  
    - \(V_L\): vincoli leader  
    - \(f_F\): funzione obiettivo follower  
    - \(V_F\): vincoli follower  
    - \(f_F\): valore funzione obiettivo follower in corrispondenza di \(\text{LB}^0\)

    ---

    ### Problema leader all'iterazione \(i\):

    \[
    \min f_L
    \]

    soggetto a:
    \[
    \begin{cases}
    V_L \\
    V_F \\
    f_F \geq \text{LB}^i
    \end{cases}
    \]

    ---

    Aggiornamento del lower bound:
    \[
    \text{LB}^i = \text{LB}^{i-1} + \varepsilon
    \]
    """

    
# Estrai parametri
    UB = params.get("UB")
    LB0 = params.get("LB0")
    maxIterations = params.get('maxIterations') #N

    # Calcolo epsilon
    epsilon = (UB - LB0) / maxIterations

    # Aggiornamento LB
    LB_i = LB0 + epsilon

    # Placeholder per calcolo funzione obiettivo leader
    # f_L = calcola_funzione_leader(params)

    # Placeholder per vincoli leader e follower
    # V_L = verifica_vincoli_leader(params)
    # V_F = verifica_vincoli_follower(params)


    TLIM = params.get('maxtime')
    total_runtime = params.get('maxtimetotal')
    idI = params.get('instancename') #'I_0' # is instance

    #max_c_e_iniz = dd.B /dd.E_min_sum
    #max_c_cong_iniz = dd.B / dd.G_sat_min_sum
    dd = Data(params, idI, params.get('INSTANCE_FROM_FILE'), 'instances_bilevel.csv')
    if params.get('params_from_params_file'):
        dd.delta = params.get('delta')
        dd.B = params.get('B')
    

    start_time = time.time()
    lgr_lambda = 0
    iteration = 0

    while iteration < maxIterations:# debut + 4:
        iteration += 1
        if iteration >= maxIterations:
            print(f"DEBUGGGGGGGGG Maximum number of iterations reached: it {iteration}.")

        fVars, lgrmodel = getLagrangianModel(dd, lgr_lambda, False)

        # add / update cut constraint f_F >= LB_i
        lgrmodel.addConstr(fVars.Z1 + fVars.Z2 >= LB_i, name='ctLBimproved')
        lgrmodel.update()

        lgrmodel.params.TimeLimit = TLIM

        #fmodel.write('model_follower_'+sdB+'.lp')
        path = params.get('folder_output')
        lgrmodel.optimize()
        if lgrmodel.Status not in [2,9,11]:
            print(f'######LAGRANGIAN MODEL CANNOT BE SOLVED: Status { lgrmodel.Status}, instance: {idI} ######')
            #fmodel.write(f' {}_{}.lp'.format(idI, kp))
            #lgrmodel.computeIIS()
            #lgrmodel.write(f'{path}/lagrangianModel_{idI}_{iteration}_delta{dd.delta}.ilp')
            print("b: ", [(k,v) for (k,v) in fVars.b.items()])
            print(dd.B)
            break
        print('######OTIMAL LEADER OBJECTIVE FUNCTION VALUE######')
        LB = lgrmodel.ObjVal
        print(LB)
        print(f"runtime = {lgrmodel.Runtime}")
        gap = lgrmodel.MIPGap
        LB_i += epsilon  # update LB for next iteration
        if (time.time() - start_time) >= total_runtime:
            print("Maximum total runtime reached.")
            break
        
    process_time = time.time() - start_time

    
    results = {'LB' : LB_i, 'runtime' : process_time, 'iterations' : iteration}

    return results



def lagrangian(params, upper_bound : bool = False):
    
    TLIM = params.get('maxtime')
    idI = params.get('instancename') #'I_0' # is instance

    #max_c_e_iniz = dd.B /dd.E_min_sum
    #max_c_cong_iniz = dd.B / dd.G_sat_min_sum
    dd = Data(params, idI, params.get('INSTANCE_FROM_FILE'), 'instances_bilevel.csv')
    if params.get('params_from_params_file'):
        dd.delta = params.get('delta')
        dd.B = params.get('B')
    E0 = dd.maxCO2()
    #DeltaB = [1, 1.25, 1.5, 1.75, 2] #[0.5, 0.75, 1, 1.25, 1.5,5]
    #DeltaB = [5,10,15,20]
    #DeltaB = [1]

    list_lObj = []
    list_fObj = []
    list_lZ1s = []
    list_lZ2s = []
    list_runtime = []
    list_fgap = []
    iteration = 0
    maxIterations = params.get('maxIterations')
    epsilon_ratio = params.get('epsilon_ratio')
    epsilon = epsilon_ratio * dd.B / dd.LF
    deltaE = [0 for j in dd.F]
    currentE = [0 for j in dd.F]
    bestB0 = {j:0 for j in dd.F}
    currentB0 = {j:0 for j in dd.F}
    best_it = 0
    n_it_not_improved = 0
    max_it_not_improved = 15
    best_lobjval = float('inf')
    improvementHeuristic = 0 # improvement of the euristic solution with respect to the initial solution
    firstSolutionValue = -1
    df_track_results = pd.DataFrame(columns=['iteration', 'bestlobjval', 'lobjval', 'fobjval', 'lz1', 'lz2', 'tot_runtime', 'gapf'])
    tot_runtime = time.time()
    lgr_lambda = 0
    while iteration < maxIterations:
        iteration += 1
        n_it_not_improved += 1

        fVars, lgrmodel = getLagrangianModel(dd, lgr_lambda, upper_bound)
        lgrmodel.params.TimeLimit = TLIM

        #fmodel.write('model_follower_'+sdB+'.lp')
        path = params.get('folder_output')
        lgrmodel.optimize()
        if lgrmodel.Status not in [2,9,11]:
            print(f'######LAGRANGIAN MODEL CANNOT BE SOLVED: Status { lgrmodel.Status}, instance: {idI} ######')
            #fmodel.write(f' {}_{}.lp'.format(idI, kp))
            #lgrmodel.computeIIS()
            #lgrmodel.write(f'{path}/lagrangianModel_{idI}_{iteration}_delta{dd.delta}.ilp')
            print("b: ", [(k,v) for (k,v) in fVars.b.items()])
            print(dd.B)
            continue
        print('######OTIMAL LEADER OBJECTIVE FUNCTION VALUE######')
        print(fVars.calcLeaderObj(dd) if upper_bound else lgrmodel.ObjVal)
        print(f"runtime = {lgrmodel.Runtime}")
        gap = lgrmodel.MIPGap
        lgrObjVal = float('inf')
        Z1f = float('inf')
        Z2f = float('inf')
        Zf = float('inf')
        lgrZ1 = float('inf')
        lgrZ2 = float('inf')
        print(f"gap = {gap}")
        if  not math.isinf(gap):
            Z1f, Z2f = fVars.calcFollowerObj(dd)
            Zf = Z1f + Z2f
            lgrObjVal = fVars.calcLeaderObj(dd) if upper_bound else lgrmodel.ObjVal
            lgrZ1 = fVars.calcZ1Leader(dd) if upper_bound else  fVars.Z1.getValue()
            lgrZ2 = fVars.calcZ2Leader(dd) if upper_bound else  fVars.Z2.getValue()
            
        print('######CALCULATED FOLLOWER OBJECTIVE FUNCTION VALUE######')
        print(Zf)
        list_fObj.append(Zf)
        list_runtime.append(lgrmodel.Runtime)
        list_fgap.append(lgrmodel.MIPGap)


        list_lObj.append(lgrObjVal)
        list_lZ1s.append(lgrZ1)
        list_lZ2s.append(lgrZ2)

        # updating deltaE for keep track of the increase / decrease of emissions for each follower
        #for j in dd.F:
        #    _e = [fVars.e[j, t].x for t in dd.T]
        #    sumE = sum(_e)
        #    deltaE[j-1] = sumE - currentE[j-1]
        #    currentE[j-1] = sumE
        #currentB0 = {j:fVars.b[j,0].x for j in dd.F}

        if iteration == 1:
            firstSolutionValue = lgrObjVal

        if lgrObjVal < best_lobjval:
            print(f'###### ITERATION {iteration} - NEW BEST LEADER OBJECTIVE FUNCTION VALUE {lgrObjVal} ######')
            best_lobjval = lgrObjVal
            best_it = iteration
            n_it_not_improved = 0
            
            path = params.get('folder_output')
            print(f'writing lagrangian results to in directory{path}')
            toExcelLagrangian(fVars, lgrmodel, f'{path}/lgrn_results_{idI}_{iteration}_delta{dd.delta}.xlsx', f'{path}/log_lgrn_{idI}_{iteration}.csv', dd)
            print('...done.')

        print('...done iteration ', iteration)
        df_track_results.loc[len(df_track_results)] = [iteration, best_lobjval, lgrObjVal, Z1f+Z2f, lgrZ1, lgrZ2, lgrmodel.Runtime, lgrmodel.MIPGap]

    tot_runtime = time.time() - tot_runtime
    improvementHeuristic = 100*(firstSolutionValue - best_lobjval) / firstSolutionValue
    print('Total runtime = ', tot_runtime)
    print(f'BEST IT ----------->{best_it}.---------------All iterations completed.')
    df_track_results.to_csv(f"{path}/track_it_{idI}.csv", index=None)

    
    try:
        results = {'lobjval' : best_lobjval, 'fobjval' : list_fObj[best_it-1], 'lz1' : list_lZ1s[best_it-1], 'lz2' : list_lZ2s[best_it-1], 'tot_runtime' : tot_runtime,
               'gapf': list_fgap[best_it-1], 'best_it': best_it, 
               'list_lObj' : list_lObj, 'list_fObj' :list_fObj, 'list_lZ1s' : list_lZ1s, 'list_lZ2s': list_lZ2s, 'list_runtime' : list_runtime, 'improvementHeuristic' : improvementHeuristic}
    except IndexError:
        results = {'lobjval' : best_lobjval, 'fobjval' : -1, 'lz1' : -1, 'lz2' : -1, 'tot_runtime' : tot_runtime,
               'gapf': -1, 'best_it': best_it, 
               'list_lObj' : list_lObj, 'list_fObj' :list_fObj, 'list_lZ1s' : list_lZ1s, 'list_lZ2s': list_lZ2s, 'list_runtime' : list_runtime, 'improvementHeuristic' : improvementHeuristic}

    dfall = pd.DataFrame(data = results)
    dfall.to_csv(f"{path}/results_all_{idI}.csv", index=None)
    return results

def check_follower_feasibility(fVars, dd):
    """
    Rebuild a leader model from the budget values present in fVars and
    solve a follower model with checkfeasibility=True to test feasibility.

    Returns True if the follower model is solvable (status in [2,9,11]), False otherwise.
    TimeLimit is set to 100 seconds.
    """
    # Reconstruct leader model using B0 values from fVars.b (use lb if fixed, otherwise ub)
    lmodel = LModel()
    B0 = {}
    for j in dd.F:
        bj = fVars.b.get((j, 0))
        if bj is None:
            B0[j] = 0.0
            continue
        # prefer lb (it should be fixed when constructed), fallback to ub
        val = getattr(bj, "lb", None)
        if val is None:
            val = getattr(bj, "ub", None)
        if val is None:
            # fallback to 0.0 if neither bound is available
            val = 0.0
        B0[j] = val
    lmodel.B0 = B0

    # Reconstruct C using dd.E0 if present, otherwise use maxCO2()
    E0 = getattr(dd, "E0", None)
    if E0 is None:
        E0 = dd.maxCO2()
    lmodel.C = {t: E0 * (1 - (dd.delta / (dd.LT - 1)) * (t - 1)) for t in dd.T}

    # Build follower model in feasibility-check mode and solve with a short time limit
    _, fmodel_check = getFollowerModel(lmodel, dd, checkfeasibility=True)
    fmodel_check.params.TimeLimit = 100
    fmodel_check.optimize()

    # Treat statuses 2, 9, 11 as acceptable (consistent with other code in the file)
    return fmodel_check.Status in [2, 9, 11]

def read_params():
    with open('PARAMS.json', 'r') as f:
        params = json.load(f)
    return params

def run(params : dict):
    result = None
    algotype = params.get('algotype')
    if algotype == 'bisection':
        result = bisectionHeuristics(params)
    elif algotype == 'LAGRANGIAN': #TODO change name to LB (Lower Bound)
        result = lagrangian(params) 
    elif algotype == 'LBI':  #Lower Bound Improved
        result = lower_bound_improved(params)
    elif algotype == 'UPPER_BOUND':
        result = upper_bound(params)
    elif algotype == 'NSGAII':
        result = NSGAII(params)
    return result

if __name__=="__main__":
    params = read_params()
    algotype = params.get('algotype')

    if algotype == 'bisection':
        bisectionHeuristics(params)
