import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from core.fitness.fitness_functions import objective_function
from core.algorithms.genetic_algorithm import GeneticAlgorithm
from core.algorithms.particle_swarm import ParticleSwarm
from core.operators.selection import roulette_selection
from core.operators.crossover import one_point_crossover, blx_alpha_crossover
from core.operators.mutation import uniform_mutation

NUM_RUNS = 5  # Quantas vezes rodar cada configuração para tirar a média
OUTPUT_FILE = "resultados_experimento.csv"

# --- FUNÇÕES AUXILIARES ---

def calculate_metrics(history_best, final_population, best_pos, lower, upper):
    # 1. Ajustar Fitness (desfazer a negação da minimização)
    real_history = [-val for val in history_best]
    best_fitness = real_history[-1]

    # 2. Iteração de Convergência (Estabilização)
    # Considera estabilizado se a variação for menor que 1e-6 em relação ao final
    stabilization_iter = len(real_history)
    for i, val in enumerate(real_history):
        if abs(val - best_fitness) < 1e-6:
            stabilization_iter = i
            break
    
    # 3. % Vizinhança
    # Para o GA, a população precisa ser escalada antes se estiver em [0,1]
    # Assumimos que 'final_population' já chega aqui na escala REAL do problema
    radius = 10.0
    pop_arr = np.array(final_population)
    best_arr = np.array(best_pos)
    
    neigh_pct = 0.0
    if pop_arr.size > 0:
        dists = np.linalg.norm(pop_arr - best_arr, axis=1)
        neigh_count = np.count_nonzero(dists <= radius)
        neigh_pct = (neigh_count / len(pop_arr)) * 100.0
        
    return best_fitness, stabilization_iter, neigh_pct

# --- DEFINIÇÃO DOS CENÁRIOS DE TESTE ---

# Cenários para o GA (Variando Tamanho da População e Mutação)
ga_scenarios = [
    {"name": "GA_Padrao_OnePoint", "pop": 40, "gen": 200, "mut": 0.1, "cross": 0.7, "type": "one-point"},
    {"name": "GA_Pop_Baixa_OnePoint", "pop": 20, "gen": 200, "mut": 0.1, "cross": 0.7, "type": "one-point"},
    {"name": "GA_Pop_Alta_OnePoint", "pop": 80, "gen": 200, "mut": 0.1, "cross": 0.7, "type": "one-point"},
    {"name": "GA_Mut_Alta_OnePoint", "pop": 40, "gen": 200, "mut": 0.2, "cross": 0.7, "type": "one-point"},
    # BLX-α variants
    {"name": "GA_Padrao_BLX", "pop": 40, "gen": 200, "mut": 0.1, "cross": 0.7, "type": "blx-alpha", "alpha": 0.3},
    {"name": "GA_Mut_Alta_BLX", "pop": 40, "gen": 200, "mut": 0.2, "cross": 0.7, "type": "blx-alpha", "alpha": 0.3},
    {"name": "GA_Pop_Baixa_BLX", "pop": 20, "gen": 200, "mut": 0.1, "cross": 0.7, "type": "blx-alpha", "alpha": 0.3},
]

# Cenários para o PSO (Variando Inércia e Partículas)
pso_scenarios = [
    {"name": "PSO_Padrao", "part": 40, "iter": 200, "w": 0.7, "c1": 1.5, "c2": 1.5},
    {"name": "PSO_Inercia_Baixa", "part": 40, "iter": 200, "w": 0.4, "c1": 1.5, "c2": 1.5},
    {"name": "PSO_Inercia_Alta", "part": 40, "iter": 200, "w": 0.9, "c1": 1.5, "c2": 1.5},
    {"name": "PSO_Pop_Alta", "part": 80, "iter": 200, "w": 0.7, "c1": 1.5, "c2": 1.5},
]

results_data = []

print(f"=== INICIANDO BATERIA DE TESTES AUTOMATIZADOS ===")
print(f"Repetições por cenário: {NUM_RUNS}\n")

# --- EXECUÇÃO DO GA ---
for scenario in ga_scenarios:
    print(f"Rodando: {scenario['name']}...")
    
    for run in range(NUM_RUNS):
        # Configuração do Wrapper de Fitness (igual ao seu Worker)
        # O GA opera em [0,1], então precisamos escalar para [-100, 100]
        # x_real = x_norm * 200 - 100
        fitness_wrapper = lambda pop: -objective_function(pop * 200 - 100)
        
        # Escolhe operador de crossover
        if scenario.get("type") == "blx-alpha":
            alpha = float(scenario.get("alpha", 0.3))
            def crossover_fn(pop, rate, rng):
                return blx_alpha_crossover(pop, rate, alpha, rng, lower_bounds=0.0, upper_bounds=1.0)
        else:
            crossover_fn = one_point_crossover

        ga = GeneticAlgorithm(
            fitness_func=fitness_wrapper,
            population_size=scenario["pop"],
            num_generations=scenario["gen"],
            crossover_rate=scenario["cross"],
            mutation_rate=scenario["mut"],
            chromosome_length=2,
            selection_op=roulette_selection, 
            crossover_op=crossover_fn, 
            mutation_op=uniform_mutation
        )
        
        # Executa
        best_ind, best_fit_val = ga.optimize()
        
        hist = getattr(ga, "history_best", [-best_fit_val]*scenario["gen"]) 
        
        final_pop_scaled = ga.population * 200 - 100
        best_pos_scaled = best_ind * 200 - 100
        
        bf, stab_iter, neigh = calculate_metrics(hist, final_pop_scaled, best_pos_scaled, -100, 100)
        
        results_data.append({
            "Algoritmo": "GA",
            "Cenario": scenario["name"],
            "Execucao": run + 1,
            "Melhor_Fitness": bf,
            "Iter_Convergencia": stab_iter,
            "Vizinhanca_Pct": neigh,
            "Params": str(scenario)
        })

for scenario in pso_scenarios:
    print(f"Rodando: {scenario['name']}...")
    
    for run in range(NUM_RUNS):
        fitness_wrapper = lambda pos: -objective_function(pos)
        
        pso = ParticleSwarm(
            fitness_func=fitness_wrapper,
            num_particles=scenario["part"],
            num_iterations=scenario["iter"],
            dimension=2,
            inertia=scenario["w"],
            cognitive=scenario["c1"],
            social=scenario["c2"],
            lower_bound=-100,
            upper_bound=100
        )
        
        best_pos, best_fit_val = pso.optimize()
        
        hist = getattr(pso, "history_best", [-best_fit_val]*scenario["iter"])
        
        bf, stab_iter, neigh = calculate_metrics(hist, pso.positions, best_pos, -100, 100)
        
        results_data.append({
            "Algoritmo": "PSO",
            "Cenario": scenario["name"],
            "Execucao": run + 1,
            "Melhor_Fitness": bf,
            "Iter_Convergencia": stab_iter,
            "Vizinhanca_Pct": neigh,
            "Params": str(scenario)
        })

df = pd.DataFrame(results_data)

summary = df.groupby(["Algoritmo", "Cenario"])[["Melhor_Fitness", "Iter_Convergencia", "Vizinhanca_Pct"]].mean()

print("\n=== RESUMO DOS RESULTADOS (MÉDIA) ===")
print(summary)

df.to_csv(OUTPUT_FILE, index=False, sep=";")
print(f"\nRelatório detalhado salvo em: {OUTPUT_FILE}")

summary.to_csv("resumo_media.csv", sep=";")