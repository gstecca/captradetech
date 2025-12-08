# Copyright (c) 2025 Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Authors: Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Email: giuseppe.stecca@gmail.com
import os
# Set Gurobi license file path
#os.environ['GRB_LICENSE_FILE'] = r'C:/U/annalaura/gurobi.lic'
#import itertools
#import numpy as np
import random
from numpy.linalg import norm
#import sys
import json
import time
import pandas as pd
import os
from bilevel_params import *
import random
import os
import json
import time
import pandas as pd
import multiprocessing
from gurobipy import GRB, GurobiError
# --- Modifica della classe soluzione per includere E0 ---
class LSol:
    def __init__(self, E0: float, B0: dict) -> None:
        # E0: emissioni iniziali candidate (leader decision variable indipendente)
        # B0: dizionario budget follower {j: valore}
        self.E0 = E0
        self.B0 = B0
        self.Z1 = 0.0
        self.Z2 = 0.0
        self.fObj = 0.0
        self.lObj = 0.0
        self.DominatedBySolutions = []
        self.lm = None

    def to_vector(self, dd):
        # se serve una rappresentazione vettoriale ordinata:
        # [E0, b_f1, b_f2, ...] con ordine su dd.F
        return [self.E0] + [self.B0[j] for j in dd.F]

# --- Manteniamo max_B0_iniz per generare i massimi budget  ---
def max_B0_iniz(LF, B) -> dict:
    seq = random.sample(range(1, LF+1), LF)
    max_B0_iniz = {j: 0 for j in seq}
    # Generate random weights
    r = {i: random.random() for i in seq}
    # Normalize weights to sum to 1
    total_r = sum(r.values())
    r = {i: v/total_r for i, v in r.items()}
    # Allocate exact portions of B
    for i in seq:
        max_B0_iniz[i] = r[i] * B
    return max_B0_iniz

# --- Inizializzazione popolazione: genera E0 casuale fra E0_min e E0_max e B0 per follower ---
def getInitPop(N, max_B0, dd, E0_min=None, E0_max=None) -> list:
    if E0_min is None:
        E0_min = dd.minCO2()
    if E0_max is None:
        E0_max = dd.maxCO2()

    min_B0 = {j: 0 for j in dd.F}
    parentPop = []
    print("Creating initial population...")
    print(f"N = {N}")
    print(f"max_B0 = {max_B0}")
    print(f"dd.F = {dd.F}")
    for i in range(N):
        # genera E0 casuale nello spazio ammissibile
        E0 = random.random() * (E0_max - E0_min) + E0_min
        B0 = {}
        for j in dd.F:
            B0[j] = max(0.0, random.random() * (max_B0[j] - min_B0[j]) + min_B0[j])
        aSol = LSol(E0, B0)
        parentPop.append(aSol)
        print(f"Solution {i} E0 = {E0:.4f}, B0 = {B0}")
    print(f"Final parentPop length = {len(parentPop)}")
    return parentPop

# --- SBX crossover: applicata sia su E0 sia su ciascun B0[j] ---
def SBX(p1: LSol, p2: LSol, eta_c, dd, E0_min=None, E0_max=None):
    if E0_min is None:
        E0_min = dd.minCO2()
    if E0_max is None:
        E0_max = dd.maxCO2()

    rho_b = {}
    sigma_b = {}
    # crossover per E0
    rho_e = random.random()
    if rho_e <= 0.5:
        sigma_e = 2 * (rho_e ** (1.0/(eta_c + 1.0)))
    else:
        sigma_e = 1.0 / ((2.0 * (1.0 - rho_e)) ** (1.0/(eta_c + 1.0)))
    E0_1 = 0.5 * (1 - sigma_e) * p1.E0 + 0.5 * (1 + sigma_e) * p2.E0
    E0_2 = 0.5 * (1 + sigma_e) * p1.E0 + 0.5 * (1 - sigma_e) * p2.E0
    # clamp E0 nei bounds
    E0_1 = max(E0_min, min(E0_max, E0_1))
    E0_2 = max(E0_min, min(E0_max, E0_2))

    B0_1 = {}
    B0_2 = {}
    for j in dd.F:
        rho_b[j] = random.random()
        if rho_b[j] <= 0.5:
            sigma_b[j] = 2 * (rho_b[j] ** (1.0/(eta_c + 1.0)))
        else:
            sigma_b[j] = 1.0 / ((2.0 * (1.0 - rho_b[j])) ** (1.0/(eta_c + 1.0)))
        B0_1[j] = 0.5 * (1 - sigma_b[j]) * p1.B0[j] + 0.5 * (1 + sigma_b[j]) * p2.B0[j]
        B0_2[j] = 0.5 * (1 + sigma_b[j]) * p1.B0[j] + 0.5 * (1 - sigma_b[j]) * p2.B0[j]
        # assicurati non negativi (o applica altri vincoli se esistono)
        B0_1[j] = max(0.0, B0_1[j])
        B0_2[j] = max(0.0, B0_2[j])

    c1 = LSol(E0_1, B0_1)
    c2 = LSol(E0_2, B0_2)
    return c1, c2

# --- find max/min su popolazione: ora anche per E0 ---
def find_max_min(parentPop, dd):
    print("In find_max_min:")
    print(f"parentPop type = {type(parentPop)}")
    print(f"parentPop length = {len(parentPop)}")
    print(f"First element type = {type(parentPop[0])}")

    max_B0 = {}
    min_B0 = {}
    for j in dd.F:
        try:
            max_B0[j] = max(p.B0[j] for p in parentPop)
            min_B0[j] = min(p.B0[j] for p in parentPop)
        except AttributeError as e:
            print(f"Error: elemento in parentPop non ha attributo B0")
            if hasattr(parentPop[0], '__dict__'):
                print(f"Attributi disponibili: {parentPop[0].__dict__}")
            raise
    # anche max/min di E0
    max_E0 = max(p.E0 for p in parentPop)
    min_E0 = min(p.E0 for p in parentPop)
    return max_B0, min_B0, max_E0, min_E0

# --- Mutazione polinomiale: applicata sia su E0 sia su B0[j] ---
def PolMutation(rho_b, eta_m, p: LSol, max_B0, min_B0, dd, max_E0=None, min_E0=None):
    # Mutazione E0
    if max_E0 is None:
        max_E0 = dd.maxCO2()
    if min_E0 is None:
        min_E0 = dd.minCO2()

    rho_e = random.random()
    if rho_e < 0.5:
        sigma_e = 2 * (rho_e ** (1.0/(eta_m + 1.0))) - 1.0
    else:
        sigma_e = 1.0 - ((2 * (1.0 - rho_e)) ** (1.0/(eta_m + 1.0)))
    E0_new = p.E0 + (max_E0 - min_E0) * sigma_e
    E0_new = max(min_E0, min(max_E0, E0_new))

    # Mutazione budgets
    B0_new = {}
    for j in dd.F:
        # usa rho_b[j] già generato (se non lo è, fallo qui)
        r = rho_b.get(j, random.random())
        if r < 0.5:
            sigma_b = 2 * (r ** (1.0/(eta_m + 1.0))) - 1.0
        else:
            sigma_b = 1.0 - ((2.0 * (1.0 - r)) ** (1.0/(eta_m + 1.0)))
        # usa la differenza tra max e min per scalare la variazione
        delta = max_B0[j] - min_B0[j] if j in max_B0 and j in min_B0 else 0.0
        B0_new[j] = max(0.0, p.B0[j] + delta * sigma_b)

    mutated_solution = LSol(E0_new, B0_new)
    return mutated_solution

# --- createOffspring: genera figli con SBX (incluso E0) e poi muta (incluso E0) ---
def createOffspring(parentPop, eta_c, eta_m, max_b, min_b, dd, E0_min=None, E0_max=None):
    offspringPop = []
    # crossover a coppie
    for i in range(0, len(parentPop) - 1, 2):
        p1 = parentPop[i]
        p2 = parentPop[i + 1]
        c1, c2 = SBX(p1, p2, eta_c, dd, E0_min=E0_min, E0_max=E0_max)
        offspringPop.append(c1)
        offspringPop.append(c2)

    mutatedOffspringPop = []
    for offspring in offspringPop:
        rho_b = {}
        for j in dd.F:
            rho_b[j] = random.random()
        mutated_offspring = PolMutation(rho_b, eta_m, offspring, max_b, min_b, dd, max_E0=E0_max, min_E0=E0_min)
        mutatedOffspringPop.append(mutated_offspring)

    return mutatedOffspringPop

# --- union parent/offspring invariato ---
def union_parent_offspring(parentPop: list, offspringPop: list):
    U = parentPop + offspringPop
    return U

# --- fast non dominated sort: ho lasciato la logica ma attenzione a DominatedBySolutions ---
def fast_non_dominated_sort(U: list):
    """
    Per il caso mono-obiettivo (Z = Z1 + Z2) ritorna un'unica lista (fronte)
    con le soluzioni ordinate in modo crescente per Z.
    """
    print("\nInizio fast_non_dominated_sort (single objective Z = Z1 + Z2)")
    if not U:
        return [[]]   # manteniamo la struttura a lista di fronti

    # Calcola Z e assicura che gli attributi esistano
    for u in U:
        # assumo che Z1 e Z2 esistano e siano numerici; se no fallirà esplicitamente
        u.Z = u.Z1 + u.Z2

    # Ordine crescente su Z (stable sort)
    sorted_pop = sorted(U, key=lambda s: s.Z)
    print(f"Popolazione ordinata per Z. Prima Z = {sorted_pop[0].Z}, ultima Z = {sorted_pop[-1].Z}")
    # Restituisco un solo "fronte" contenente tutti gli individui ordinati
    return [sorted_pop]


def combine_and_select(N: int, sorted_fronts: list):
    """
    Seleziona i primi N individui leggendo i fronti in ordine.
    In questo contesto c'è un solo fronte ordinato per Z, quindi
    si prendono semplicemente i primi N.
    """
    print(f"\nCombine and select. N = {N}")
    if not sorted_fronts:
        return []

    combined = []
    for front in sorted_fronts:
        for sol in front:
            combined.append(sol)
            if len(combined) >= N:
                break
        if len(combined) >= N:
            break

    print(f"Selezionate {len(combined)} soluzioni (richieste {N}).")
    return combined[:N]

# --- NSGAII: integrazione E0 durante la valutazione dei modelli ---
def GA_old(params):
    TLIM = params.get('timelimit')
    maxIterations = params.get('maxIterations')
    maxtimetotal = 600
    N = 2
    eta_m = 20
    eta_c = 10
    idI = params.get('instancename')
    dd = Data(params=params, inst_id=idI, INSTANCE_FROM_FILE=params.get('INSTANCE_FROM_FILE'), instance_file='instances_bilevel.csv')
    E0_max = dd.maxCO2()
    E0_min = dd.minCO2()
    path = params.get('folder_output', '.')
    os.makedirs(path, exist_ok=True)
    it = 1
    dfall = pd.DataFrame(columns=[
        'Iteration',
        'E0',
        'b',
        'FollowerObjVal',
        'LeaderObjVal',
        'LeaderZ1',
        'LeaderZ2',
        'Gap',
        'Runtime'
    ])
    start_total_runtime = time.time()
    best_solution = None
    best_obj_val = float('inf')
    # total_runtime sarà calcolato solo alla fine

    while it < maxIterations:
        total_runtime = time.time() - start_total_runtime
        if total_runtime >= maxtimetotal:
            print(f"Tempo totale limite ({maxtimetotal}s) raggiunto all'inizio dell'iterazione {it}: {total_runtime:.2f}s")
            break

        max_B0_iniziale = max_B0_iniz(dd.LF, dd.B)
        initPop = getInitPop(N, max_B0_iniziale, dd, E0_min=E0_min, E0_max=E0_max)
        max_B0, min_B0, max_E0_pop, min_E0_pop = find_max_min(initPop, dd)
        offspringPop = createOffspring(initPop, eta_c, eta_m, max_B0, min_B0, dd, E0_min=min_E0_pop, E0_max=max_E0_pop)
        U = union_parent_offspring(initPop, offspringPop)

        feasible_found = False
        evaluated_individuals = []   # raccolgo qui solo gli individui valutati
        ind = 0
        for individual in U:
            ind += 1
            print("###############################")
            print("it=", it)
            print("individual=", ind)
            print("E0 =", individual.E0)
            print("B0", individual.B0)
            print("###############################")

            # imposta E0 nel problema (usalo quando costruisci il modello leader/follower)
            dd.E0 = individual.E0
            lmodel = LModel()
            lmodel.B0 = individual.B0
            # usa individual.E0 per costruire i costi C (era E0 non definito nel tuo snippet originale)
            lmodel.C = {t: individual.E0 * (1 - (dd.delta / (dd.LT - 1)) * (t - 1)) for t in dd.T}
            fVars, fmodel = getFollowerModel(lmodel, dd)
            fmodel.params.TimeLimit = TLIM
            start_time = time.time()
            fmodel.optimize()
            runtime = time.time() - start_time

            # controlla lo stato del follower
            if fmodel.Status not in [2, 9, 11]:
                individual.Z1 = float('inf')
                individual.Z2 = float('inf')
                continue

            feasible_found = True
            folder_out = os.path.join(path, f"results_GA/{idI}")
            os.makedirs(folder_out, exist_ok=True)
            # file_xlsx = os.path.join(folder_out, f"follower_model_it{it}_ind{ind}_{idI}.xlsx")
            # file_log = os.path.join(folder_out, f"follower_log_{idI}.csv")
            # toExcelFollower(fVars, fmodel, file_xlsx, file_log, dd)
            lmodel.solve(fVars, dd)
            lmodel.feasible(fVars, dd)
            individual.Z1 = float(lmodel.Z1)
            individual.Z2 = float(lmodel.Z2)
            individual.lm = lmodel
            individual.fObj = float(fVars.Z1.getValue() + fVars.Z2.getValue())
            individual.lObj = float(lmodel.ObjVal)

            if lmodel.ObjVal < best_obj_val:
                best_obj_val = float(lmodel.ObjVal)
                best_solution = {
                    'lobjval': float(lmodel.ObjVal),
                    'fobjval': float(fVars.Z1.getValue() + fVars.Z2.getValue()),
                    'lz1': float(individual.Z1),
                    'lz2': float(individual.Z2),
                    'tot_runtime': float(total_runtime),
                    'gapf': float(fmodel.MIPGap) if hasattr(fmodel, 'MIPGap') else float('inf'),
                    'best_it': it,
                    'best_individual': individual,
                    'best_E0': individual.E0,
                    'best_B0': dict(individual.B0) if hasattr(individual, 'B0') else None
                }
                folder_out = os.path.join(path, f"results_GA/{idI}")
                os.makedirs(folder_out, exist_ok=True)
                #salva json
                try:
                    best_json = {
                        'lobjval': best_solution['lobjval'],
                        'fobjval': best_solution['fobjval'],
                        'lz1': best_solution['lz1'],
                        'lz2': best_solution['lz2'],
                        'best_it': best_solution['best_it'],
                        'best_E0': best_solution['best_E0'],
                        'best_B0': best_solution['best_B0']
                    }
                    out_json = os.path.join(path, f"best_solution_{idI}.json")
                    with open(out_json, 'w') as bf:
                        json.dump(best_json, bf, indent=2)
                    print(f"Best solution JSON written: {out_json}")
                except Exception as e:
                    print('Errore nel salvataggio della best_solution su disco:', e)
                try:
                    out_xlsx = os.path.join(path, f"leader_results_best_{idI}.xlsx")
                    out_log = os.path.join(path, f"leader_log_best_{idI}.csv")
                    toExcelLeader(individual.lm, out_xlsx, out_log, dd)
                    print(f"Leader Excel written: {out_xlsx}")
                except Exception as e:
                    print('Impossibile scrivere leader_results_best file:', e)
            dfall.loc[len(dfall)] = [
                it,
                individual.E0,
                individual.B0,
                float(individual.fObj),
                float(individual.lObj),
                float(individual.Z1),
                float(individual.Z2),
                float(fmodel.MIPGap) if hasattr(fmodel, 'MIPGap') else float('inf'),
                float(runtime)
            ]

        # aggiungo l'individuo alla lista degli effettivamente valutati
            evaluated_individuals.append(individual)

        # Se non è stato trovato nessun individuo feasible in tutti quelli valutati
        if not feasible_found:
            print("Nessuna soluzione feasible trovata tra gli individui valutati in questa iterazione")
            # Non esco, passo all'iterazione successiva
            it += 1
            continue

        # ORDINA I FRONTI usando solo gli individui che sono stati valutati (come richiesto)
        sorted_fronts = fast_non_dominated_sort(evaluated_individuals)

        # Seleziona la nuova popolazione (combine_and_select dovrebbe accettare i fronti)
        newPop = combine_and_select(N, sorted_fronts)
        initPop = newPop

        # altrimenti avanza all'iterazione successiva
        it += 1

    print('All iterations completed.')
    total_runtime = time.time() - start_total_runtime
    if best_solution is None:
        best_solution = {
            'lobjval': float('inf'),
            'fobjval': float('inf'),
            'lz1': float('inf'),
            'lz2': float('inf'),
            'tot_runtime': float(total_runtime),
            'gapf': float('inf'),
            'best_it': -1
        }
    file_all = os.path.join(folder_out, "results_all_GA.xlsx")
    dfall.to_excel(file_all, index=None)
    return best_solution

def GA(params):
    TLIM = params.get('timelimit')
    maxIterations = params.get('maxIterations')
    maxtimetotal = 600
    N = 2
    eta_m = 20
    eta_c = 10
    idI = params.get('instancename')
    path = params.get('folder_output', '.')
    os.makedirs(path, exist_ok=True)
    
    dd = Data(params=params, inst_id=idI, INSTANCE_FROM_FILE=params.get('INSTANCE_FROM_FILE'), instance_file='instances_bilevel.csv')
    E0_max = dd.maxCO2()
    E0_min = dd.minCO2()

    folder_out = os.path.join(path, f"results_GA/{idI}")
    os.makedirs(folder_out, exist_ok=True)

    it = 1
    dfall = pd.DataFrame(columns=[
        'Iteration',
        'E0',
        'b',
        'FollowerObjVal',
        'LeaderObjVal',
        'LeaderZ1',
        'LeaderZ2',
        'Gap',
        'Runtime'
    ])
    start_total_runtime = time.time()
    best_solution = None
    best_obj_val = float('inf')

    while it < maxIterations:
        total_runtime = time.time() - start_total_runtime
        if total_runtime >= maxtimetotal:
            print(f"Tempo totale limite ({maxtimetotal}s) raggiunto all'inizio dell'iterazione {it}: {total_runtime:.2f}s")
            break

        max_B0_iniziale = max_B0_iniz(dd.LF, dd.B)
        initPop = getInitPop(N, max_B0_iniziale, dd, E0_min=E0_min, E0_max=E0_max)
        max_B0, min_B0, max_E0_pop, min_E0_pop = find_max_min(initPop, dd)
        offspringPop = createOffspring(initPop, eta_c, eta_m, max_B0, min_B0, dd, E0_min=min_E0_pop, E0_max=max_E0_pop)
        U = union_parent_offspring(initPop, offspringPop)

        feasible_found = False
        evaluated_individuals = []   # raccolgo qui solo gli individui valutati
        ind = 0
        for individual in U:
            ind += 1
            print("###############################")
            print("it=", it)
            print("individual=", ind)
            print("E0 =", getattr(individual, 'E0', None))
            print("B0", getattr(individual, 'B0', None))
            print("###############################")

            # imposta E0 nel problema
            dd.E0 = individual.E0
            lmodel = LModel()
            lmodel.B0 = individual.B0
            # costruisci i costi C usando individual.E0
            lmodel.C = {t: individual.E0 * (1 - (dd.delta / (dd.LT - 1)) * (t - 1)) for t in dd.T}
            fVars, fmodel = getFollowerModel(lmodel, dd)
            fmodel.params.TimeLimit = TLIM
            start_time = time.time()
            fmodel.reset()  # cancella tutte le soluzioni precedenti
            fmodel.update()
            fmodel.optimize()
            runtime = time.time() - start_time
            status = getattr(fmodel, 'Status', None)
            mipgap = getattr(fmodel, 'MIPGap', getattr(fmodel, 'mipgap', None))
            print(f"[DEBUG] follower status={status}, MIPGap={mipgap}, runtime={runtime:.2f}s")
            # controlla lo stato del follower: 2 = OPTIMAL (Gurobi), 9 = INTERRUPTED/??, 11 = TIME_LIMIT? (adatta al tuo solver)
            if fmodel.SolCount > 0:
                fObj = float(fVars.Z1.getValue() + fVars.Z2.getValue())
                has_incumbent = True
            else:
                fObj = float('inf')
                has_incumbent = False

            if not has_incumbent:
                print(f"Follower non ha soluzione reale. Skip leader.")
                individual.Z1 = float('inf')
                individual.Z2 = float('inf')
                individual.fObj = float('inf')
                individual.lObj = float('inf')
                dfall.loc[len(dfall)] = [
                    it, individual.E0, individual.B0, individual.fObj, individual.lObj,
                    individual.Z1, individual.Z2,
                    float(mipgap) if mipgap is not None else float('inf'),
                    float(runtime)
                ]
                evaluated_individuals.append(individual)
                continue

            # se arriviamo qui il follower ha dato uno stato valido
            feasible_found = True

            # risolvo il leader usando i risultati del follower
            lmodel.solve(fVars, dd)
            lmodel.feasible(fVars, dd)
            individual.Z1 = float(lmodel.Z1)
            individual.Z2 = float(lmodel.Z2)
            individual.lm = lmodel
            individual.fObj = float(fVars.Z1.getValue() + fVars.Z2.getValue())
            individual.lObj = float(lmodel.ObjVal)

            # aggiorno best
            if lmodel.ObjVal < best_obj_val:
                best_obj_val = float(lmodel.ObjVal)
                best_solution = {
                    'lobjval': float(lmodel.ObjVal),
                    'fobjval': float(fVars.Z1.getValue() + fVars.Z2.getValue()),
                    'lz1': float(individual.Z1),
                    'lz2': float(individual.Z2),
                    'tot_runtime': float(total_runtime),
                    'gapf': float(getattr(fmodel, 'MIPGap', float('inf'))),
                    'best_it': it,
                    'best_individual': individual,
                    'best_E0': individual.E0,
                    'best_B0': dict(individual.B0) if hasattr(individual, 'B0') else None
                }
                # salva json (cartella già creata in alto)
                try:
                    best_json = {
                        'lobjval': best_solution['lobjval'],
                        'fobjval': best_solution['fobjval'],
                        'lz1': best_solution['lz1'],
                        'lz2': best_solution['lz2'],
                        'best_it': best_solution['best_it'],
                        'best_E0': best_solution['best_E0'],
                        'best_B0': best_solution['best_B0']
                    }
                    out_json = os.path.join(folder_out, f"best_solution_{idI}.json")
                    with open(out_json, 'w') as bf:
                        json.dump(best_json, bf, indent=2)
                    print(f"Best solution JSON written: {out_json}")
                except Exception as e:
                    print('Errore nel salvataggio della best_solution su disco:', e)
                # try:
                #     out_xlsx = os.path.join(folder_out, f"leader_results_best_{idI}.xlsx")
                #     out_log = os.path.join(folder_out, f"leader_log_best_{idI}.csv")
                #     toExcelLeader(individual.lm, out_xlsx, out_log, dd)
                #     print(f"Leader Excel written: {out_xlsx}")
                # except Exception as e:
                #     print('Impossibile scrivere leader_results_best file:', e)

            # salva su dataframe i risultati del singolo individuo (valutato)
            dfall.loc[len(dfall)] = [
                it,
                individual.E0,
                individual.B0,
                float(individual.fObj),
                float(individual.lObj),
                float(individual.Z1),
                float(individual.Z2),
                float(getattr(fmodel, 'MIPGap', float('inf'))),
                float(runtime)
            ]

            evaluated_individuals.append(individual)

        # fine for U

        # Se non è stato trovato nessun individuo feasible in tutti quelli valutati
        if not feasible_found:
            print(f"Nessuna soluzione feasible trovata tra gli individui valutati in questa iterazione {it}. Proseguo.")
            it += 1
            continue

        # ORDINA I FRONTI usando solo gli individui che sono stati valutati
        # assicurati che fast_non_dominated_sort gestisca lista vuota (ma qui non lo è)
        sorted_fronts = fast_non_dominated_sort(evaluated_individuals)

        # Seleziona la nuova popolazione (combine_and_select dovrebbe accettare i fronti)
        newPop = combine_and_select(N, sorted_fronts)
        initPop = newPop

        it += 1

    # fine while
    print('All iterations completed.')
    total_runtime = time.time() - start_total_runtime
    if best_solution is None:
        best_solution = {
            'lobjval': float('inf'),
            'fobjval': float('inf'),
            'lz1': float('inf'),
            'lz2': float('inf'),
            'tot_runtime': float(total_runtime),
            'gapf': float('inf'),
            'best_it': -1
        }

    # salva il dataframe aggregato (uso folder_out già creato)
    try:
        file_all = os.path.join(folder_out, "results_all_GA.xlsx")
        dfall.to_excel(file_all, index=None)
        print(f"Saved results all: {file_all}")
    except Exception as e:
        print("Errore durante il salvataggio di results_all_GA.xlsx:", e)

    return best_solution

# def read_params():
#     with open('PARAMS_GA.json', 'r') as f:
#         params = json.load(f)
#     return params

def run(params : dict):
    result = GA(params)
    return result

# if __name__=="__main__":
#     params = read_params()
#     best_solution = GA(params)
