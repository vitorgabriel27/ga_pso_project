# Otimização de Sistemas — PSO e Algoritmo Genético (GA)

Este projeto implementa e compara dois meta-heurísticos de otimização para funções em duas dimensões: Particle Swarm Optimization (PSO) e Algoritmo Genético (GA). Inclui uma interface gráfica em PyQt6 com visualização em gráficos 2D/3D, geração de GIFs da evolução e tabelas de resultados por iteração.

## Visão Geral

- Interface com abas para execução do PSO e GA, e uma aba de Visualização que mostra GIFs da evolução (2D/3D) e o melhor ponto nas superfícies.
- Execução em threads (`QThread`) com workers (`PSOWorker` e `GAWorker`) que acumulam histórico completo por iteração e sinalizam o término.
- Visualização pós-processada: sem atualização frame-a-frame durante a otimização; os GIFs e as tabelas são preenchidos após o término, evitando travamentos.
- Configuração aleatória com critérios: botões dedicados para rodar cada algoritmo com hiperparâmetros escolhidos aleatoriamente em faixas razoáveis; histórico das últimas execuções persistido.

## Estrutura do Projeto

```
project/
  app.py
  core/
    algorithms/
      genetic_algorithm.py
      particle_swarm.py
    fitness/
      fitness_functions.py
    operators/
      crossover.py
      mutation.py
      selection.py
  experiments/
    experiment_runner.py
  ui/
    main_window.py
    pso_worker.py
    ga_worker.py
    charts.py
    function_plot.py
    function_plot_3d.py
    animation.py
```


Arquivos principais:
- `ui/main_window.py`: janela principal, abas, botões e visualizações. Mostra tabelas por iteração e GIFs.
- `ui/pso_worker.py` e `ui/ga_worker.py`: lógica de execução em thread; acumulam histórico de posições/populações e melhor valor.
- `core/algorithms/particle_swarm.py`: implementação do PSO.
- `core/algorithms/genetic_algorithm.py`: implementação do GA.
- `core/operators/*.py`: operadores de seleção, crossover e mutação para o GA.
- `ui/animation.py`: geração de GIFs a partir dos históricos.
- `ui/function_plot*.py`: visualizações 2D/3D da função objetivo.

## Função Objetivo

A função utilizada por padrão é uma combinação de termos oscilatórios (tipo Schaffer/Schwefel-like) e um termo quadrático estilo Rosenbrock, definida para vetor de pontos `pos = [[x1,y1], [x2,y2], ...]`:

- Termo oscilatório:

$$
z = -x \sin(\sqrt{|x|}) \;-\; y \sin(\sqrt{|y|})
$$

- Normalização auxiliar:

$$
x_{\text{norm}} = \frac{x}{250},
\qquad
y_{\text{norm}} = \frac{y}{250}
$$

- Termo tipo Rosenbrock:

$$
r = 100\left( y_{\text{norm}} - 2x_{\text{norm}} \right)^2
    + \left( 1 - x_{\text{norm}} \right)^2
$$

- Função final:

$$
f(x,y) = r - z
$$

Na interface, minimizamos:

$$
f(x,y)
$$

Para uniformizar sinais entre algoritmos, os workers usam:

$$
-f
$$

internamente (maximizam o negativo) e invertem o sinal para exibir resultados.

## Particle Swarm Optimization (PSO)

### Ideia Matemática

- Inicialização de \(N\) partículas com posição \(\mathbf{x}_i\) e velocidade \(\mathbf{v}_i\) dentro de limites \([L, U]\).
- Atualização por iteração \(t\):

Velocidade:

\(\mathbf{v}_i^{t+1} = \omega \mathbf{v}_i^{t} + c_1 r_1 \left( \mathbf{p}_i - \mathbf{x}_i^t \right) + c_2 r_2 \left( \mathbf{g} - \mathbf{x}_i^t \right)\)

Parâmetros:
- \(\omega\): inércia  
- \(c_1\): componente cognitiva  
- \(c_2\): componente social  
- \(r_1, r_2 \sim U(0,1)\)  
- \(\mathbf{p}_i\): melhor posição pessoal  
- \(\mathbf{g}\): melhor posição global  

- Restrições de posição e velocidade aplicadas para manter limites.

### Implementação

- Parâmetros: `num_particles`, `num_iterations`, `dimension`, `inertia`, `cognitive`, `social`, `lower_bound`, `upper_bound`, `rng_seed`.
- Histórico guardado: `positions_history`, `history_best`, `history_best_pos`.
- Na UI, o `% de vizinhança` é a fração de partículas dentro de raio fixo (10) em torno do melhor ponto.

## Algoritmo Genético (GA)

### Ideia Matemática

- Representação: cromossomos reais em \([0,1]\) por dimensão; mapeados para \([-100, 100]\):

$$
\mathbf{x} = 200\mathbf{c} - 100
$$

Ciclo por geração:

1. Avaliação via aptidão (usamos \(-f\) internamente).
2. Seleção por roleta: probabilidade proporcional à aptidão.
3. Crossover de um ponto.
4. Mutação uniforme com taxa \(\mu\).
5. Elitismo opcional.

### Implementação

- Parâmetros: `population_size`, `generations`, `chromosome_length` (2), `mutation_rate`, `crossover_rate`, `rng_seed`.
- Histórico: `population_history`, `history_best`, `history_best_pos`.
- `% de vizinhança`: fração de indivíduos dentro do raio do melhor ponto.

## Visualização e GIFs

- GIFs 2D/3D são gerados com Matplotlib em `ui/animation.py`.
- Cada frame mostra partículas/indivíduos da iteração.
- Na UI, exibidos com `QMovie`, reiniciam ao trocar sub-aba.

## Configuração Aleatória e Histórico

- PSO: partículas, iterações, \(\omega\), \(c_1\), \(c_2\), limites.
- GA: população (par), gerações, mutação, crossover.

Histórico salvo em: `project/ui/output/run_history.json`.

## Execução

Pré-requisitos:
- Python 3.12
- Dependências em `requirements.txt`

Criar venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Instalar as dependências dentro do venv e executar o app:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python project\app.py
```

Observações:

- O diretório `.venv/` não é versionado; cada máquina cria o seu.
- Para desativar o venv: `Deactivate`
- Se preferir outro nome de venv, ajuste os caminhos de ativação conforme necessário.


Na janela:

- Aba `PSO`: clique `Rodar PSO` ou `Rodar PSO (config aleatória)`.
- Aba `Algoritmo Genético (GA)`: clique `Rodar GA` ou `Rodar GA (config aleatória)`.
- Aba `Visualização`: veja GIF 2D/3D com contagem de iteração; ao trocar a sub-aba o GIF reinicia.
- Tabelas mostram resultados por iteração/geração: Iteração, x, y, f(x,y), % Vizinhança.

## Aspectos Técnicos Importantes

- Sinal da aptidão: internamente os algoritmos usam \\(-f\\) para compatibilizar minimização; exibição e tabelas usam \\(f\\).
- Desempenho da UI: todos os históricos são acumulados nos workers e renderizados ao final, evitando travamentos.
- Compatibilidade Matplotlib + Qt: captura de frames via `buffer_rgba()` para robustez com `FigureCanvasQTAgg`.
- GA — população par: garantido na configuração aleatória para evitar erros de pareamento no crossover.
- Vizinhança: raio fixo (10 unidades) para o cálculo de proximidade; pode ser tornado ajustável.

## Extensões e Ideias Futuras

- Controle do raio de vizinhança na UI e reflexão nas tabelas.
- Exportação das tabelas para CSV/Excel e reexecução de configurações do histórico.
- Suporte a outras funções objetivo (ex.: Rastrigin, Ackley) via `core/fitness` e seleção na UI.
- Métricas adicionais nas tabelas: distância média/mediana, diversidade populacional.

## Licença

Uso acadêmico/educacional.
