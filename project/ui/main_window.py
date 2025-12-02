from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QPushButton,
    QVBoxLayout, QLabel, QTabWidget, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QMessageBox, QComboBox, QLineEdit
)
from PyQt6.QtCore import QThread, Qt
from PyQt6.QtGui import QMovie
import numpy as np
import os
import json

from core.fitness.fitness_functions import objective_function

from ui.ga_worker import GAWorker
from ui.pso_worker import PSOWorker
from ui.charts import HistoryChart
from ui.function_plot import FunctionPlot
from ui.function_plot_3d import FunctionPlot3D


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GA/PSO Optimizer GUI")
        self.resize(1100, 650)

        self.tabs = QTabWidget()
        self.ga_tab = self.create_ga_tab()
        self.pso_tab = self.create_pso_tab()
        self.visual_tab = self.create_visualization_tab()
        self.tested_params_tab = self.create_tested_params_tab()

        self.tabs.addTab(self.ga_tab, "Algoritmo Genético (GA)")
        self.tabs.addTab(self.pso_tab, "PSO")
        self.tabs.addTab(self.visual_tab, "Visualização")
        self.tabs.addTab(self.tested_params_tab, "Parâmetros Testados")
        
        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        container.setLayout(layout)
        self.setCentralWidget(container)

    def create_pso_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        self.pso_status = QLabel("Status: idle")
        self.btn_pso = QPushButton("Rodar PSO")
        self.btn_pso_random = QPushButton("Rodar PSO (config aleatória)")
        self.btn_pso.clicked.connect(self.run_pso)
        self.btn_pso_random.clicked.connect(lambda: self.run_pso(randomize=True))
        self.pso_cfg_label = QLabel("Configuração: padrão")
        self.pso_chart = HistoryChart()
        # Result table for PSO (all iterations)
        self.pso_table = QTableWidget(0, 5)
        self.pso_table.setHorizontalHeaderLabels(["Iteração", "x", "y", "f(x,y)", "% Vizinhança"])
        self.pso_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.pso_status)
        layout.addWidget(self.btn_pso)
        layout.addWidget(self.btn_pso_random)
        layout.addWidget(self.pso_cfg_label)
        layout.addWidget(self.pso_chart)
        layout.addWidget(self.pso_table)
        tab.setLayout(layout)
        return tab

    def create_ga_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        self.ga_status = QLabel("Status do GA: idle")
        layout.addWidget(self.ga_status)
        self.btn_ga = QPushButton("Rodar GA")
        self.btn_ga_random = QPushButton("Rodar GA (config aleatória)")
        self.btn_ga.clicked.connect(self.run_ga)
        self.btn_ga_random.clicked.connect(lambda: self.run_ga(randomize=True))
        self.ga_cfg_label = QLabel("Configuração: padrão")
        # Crossover config controls
        cfg_row = QHBoxLayout()
        lbl_type = QLabel("Crossover:")
        self.ga_crossover_combo = QComboBox()
        self.ga_crossover_combo.addItems(["one-point", "blx-alpha"])
        lbl_alpha = QLabel("α:")
        self.ga_blx_alpha_input = QLineEdit()
        self.ga_blx_alpha_input.setPlaceholderText("0.3")
        self.ga_blx_alpha_input.setFixedWidth(80)
        cfg_row.addWidget(lbl_type)
        cfg_row.addWidget(self.ga_crossover_combo)
        cfg_row.addWidget(lbl_alpha)
        cfg_row.addWidget(self.ga_blx_alpha_input)
        cfg_row.addStretch()
        layout.addWidget(self.btn_ga)
        layout.addWidget(self.btn_ga_random)
        layout.addLayout(cfg_row)
        layout.addWidget(self.ga_cfg_label)
        self.ga_chart = HistoryChart()
        # Result table for GA (all iterations)
        self.ga_table = QTableWidget(0, 5)
        self.ga_table.setHorizontalHeaderLabels(["Iteração", "x", "y", "f(x,y)", "% Vizinhança"])
        self.ga_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.ga_chart)
        layout.addWidget(self.ga_table)
        tab.setLayout(layout)
        return tab

    def create_visualization_tab(self):
        tab = QWidget()
        layout = QHBoxLayout()
        # Animation tabs
        self.animation_tabs = QTabWidget()
        # Iteration labels
        self.animation_iter_label_2d = QLabel("Iteração: -/-")
        self.animation_iter_label_3d = QLabel("Iteração: -/-")
        self.animation_label_2d = QLabel("Sem animação ainda")
        self.animation_label_2d.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.animation_label_3d = QLabel("Sem animação ainda")
        self.animation_label_3d.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c2d = QWidget(); l2d = QVBoxLayout(); l2d.addWidget(self.animation_iter_label_2d); l2d.addWidget(self.animation_label_2d); c2d.setLayout(l2d)
        c3d = QWidget(); l3d = QVBoxLayout(); l3d.addWidget(self.animation_iter_label_3d); l3d.addWidget(self.animation_label_3d); c3d.setLayout(l3d)
        self.animation_tabs.addTab(c2d, "2D")
        self.animation_tabs.addTab(c3d, "3D")
        # Graph tabs
        self.graph_tabs = QTabWidget()
        self.vis_function_plot_2d = FunctionPlot(
            fitness_func=objective_function, lower=-100, upper=100, resolution=150
        )
        self.vis_function_plot_3d = FunctionPlot3D(
            fitness_func=objective_function, lower=-100, upper=100, resolution=60
        )
        g2d = QWidget(); g2d_l = QVBoxLayout(); g2d_l.addWidget(self.vis_function_plot_2d); g2d.setLayout(g2d_l)
        g3d = QWidget(); g3d_l = QVBoxLayout(); g3d_l.addWidget(self.vis_function_plot_3d); g3d.setLayout(g3d_l)
        self.graph_tabs.addTab(g2d, "2D")
        self.graph_tabs.addTab(g3d, "3D")
        layout.addWidget(self.animation_tabs)
        layout.addWidget(self.graph_tabs)
        tab.setLayout(layout)
        # Restart GIF when switching 2D/3D tabs
        self.animation_tabs.currentChanged.connect(self._restart_current_animation)
        return tab
    
    def create_tested_params_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Histórico de Configurações Testadas")
        layout.addWidget(title)

        # Tabela
        self.params_table = QTableWidget(0, 3)
        self.params_table.setHorizontalHeaderLabels(["Algoritmo", "Configuração", "Timestamp"])
        self.params_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.params_table)

        # Botão limpar histórico
        btn_clear = QPushButton("Limpar Histórico")
        btn_clear.clicked.connect(self._clear_tested_history)
        layout.addWidget(btn_clear)

        # Carrega JSON ao iniciar
        self._load_tested_history()

        tab.setLayout(layout)
        return tab


    def run_pso(self, randomize: bool = False):
        self.pso_status.setText("Rodando PSO...")
        self.pso_thread = QThread()
        cfg = self._random_pso_config() if randomize else {
            "num_particles": 40,
            "num_iterations": 200,
            "dimension": 2,
            "inertia": 0.7,
            "cognitive": 1.5,
            "social": 1.5,
            "lower_bound": -100,
            "upper_bound": 100,
            "rng_seed": 42,
        }
        # Show chosen config in UI
        self._show_config("pso", cfg, randomize)
        self._append_history("pso", cfg)
        self.pso_worker = PSOWorker(
            fitness_func=lambda pos: -objective_function(pos),
            **cfg,
        )
        self.pso_worker.moveToThread(self.pso_thread)
        self.pso_thread.started.connect(self.pso_worker.run)
        self.pso_worker.finished_signal.connect(self.on_pso_finished)
        self.pso_thread.start()

    def on_pso_update(self, state):
        pass

    def on_pso_finished(self, result):
        best_neg = result["best"]
        self.pso_status.setText(f"Concluído! melhor = {-best_neg:.6f}")
        hist = result.get("history_best", [])
        hist_real = [-v for v in hist]
        if hist_real:
            self.pso_chart.set_series(hist_real)
        positions_history = result.get("positions_history", [])
        # Compute neighborhood percentage for PSO based on last positions
        best_pos = np.array(result.get("best_pos", [0, 0]))
        neigh_pct = 0.0
        if positions_history:
            last_positions = np.array(positions_history[-1])
            if last_positions.size:
                dists = np.linalg.norm(last_positions - best_pos, axis=1)
                radius = 10.0
                neigh_pct = float(np.mean(dists <= radius) * 100.0)
        # Fill full table across all iterations
        self._fill_pso_table(positions_history)
        try:
            from ui.animation import generate_gif_2d, generate_gif_3d
            gif_2d_meta = generate_gif_2d(positions_history, objective_function, -100, 100, "project/ui/output/pso_anim_2d.gif")
            gif_3d_meta = generate_gif_3d(positions_history, objective_function, -100, 100, "project/ui/output/pso_anim_3d.gif")
            self._set_gif(self.animation_label_2d, gif_2d_meta)
            self._set_gif(self.animation_label_3d, gif_3d_meta)
        except Exception as e:
            self.animation_label_2d.setText(f"Falha gerar GIF: {e}")
            self.animation_label_3d.setText(f"Falha gerar GIF: {e}")
        self.vis_function_plot_2d.update_best_point(best_pos)
        self.vis_function_plot_3d.update_best_point(best_pos)
        hist_raw = result.get("history_best", [])
        positions_history = result.get("positions_history", [])
        best_pos = result.get("best_pos", [0,0])
        
        if positions_history:
            last_iteration_pop = positions_history[-1]
            
            self.show_execution_report(
                algo_name="PSO",
                history_best=hist_raw,
                final_population=last_iteration_pop,
                best_position=best_pos,
                lower=-100, upper=100
            )
    
        self.pso_thread.quit()
        self.pso_thread.wait()

    def run_ga(self, randomize: bool = False):
        self.ga_status.setText("Rodando GA...")
        self.ga_thread = QThread()
        cfg = self._random_ga_config() if randomize else {
            "population_size": 40,
            "generations": 200,
            "chromosome_length": 2,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "rng_seed": 42,
        }
        # Include crossover type and alpha from UI
        if randomize:
            # Randomize crossover choice and alpha moderately
            rng = np.random.default_rng()
            cfg["crossover_type"] = ["one-point", "blx-alpha"][int(rng.integers(0,2))]
            cfg["blx_alpha"] = float(rng.uniform(0.1, 0.5))
        else:
            cfg["crossover_type"] = self.ga_crossover_combo.currentText()
            alpha_text = self.ga_blx_alpha_input.text().strip()
            cfg["blx_alpha"] = float(alpha_text) if alpha_text else 0.3
        # Show chosen config in UI
        self._show_config("ga", cfg, randomize)
        self._append_history("ga", cfg)
        self.ga_worker = GAWorker(
            fitness_func=lambda pop: -objective_function(pop * 200 - 100),
            **cfg,
        )
        self.ga_worker.moveToThread(self.ga_thread)
        self.ga_thread.started.connect(self.ga_worker.run)
        self.ga_worker.finished_signal.connect(self.on_ga_finished)
        self.ga_thread.start()

    def on_ga_update(self, state):
        pass

    def on_ga_finished(self, result):
        best_neg = result["best"]
        self.ga_status.setText(f"Concluído! melhor = {-best_neg:.6f}")
        hist = result.get("history_best", [])
        hist_real = [-v for v in hist]
        if hist_real:
            self.ga_chart.set_series(hist_real)
        best_pos_raw = np.array(result.get("best_pos", [0, 0]))
        best_pos_scaled = best_pos_raw * 200 - 100
        population_history = result.get("population_history", [])
        scaled_history = [(np.array(p) * 200 - 100).tolist() for p in population_history]
        # Compute neighborhood percentage for GA based on last population (scaled)
        neigh_pct = 0.0
        if scaled_history:
            last_pop = np.array(scaled_history[-1])
            if last_pop.size:
                dists = np.linalg.norm(last_pop - best_pos_scaled, axis=1)
                radius = 10.0
                neigh_pct = float(np.mean(dists <= radius) * 100.0)
        # Fill full table across all iterations
        self._fill_ga_table(scaled_history)
        try:
            from ui.animation import generate_gif_2d, generate_gif_3d
            gif_2d_meta = generate_gif_2d(scaled_history, objective_function, -100, 100, "project/ui/output/ga_anim_2d.gif")
            gif_3d_meta = generate_gif_3d(scaled_history, objective_function, -100, 100, "project/ui/output/ga_anim_3d.gif")
            self._set_gif(self.animation_label_2d, gif_2d_meta)
            self._set_gif(self.animation_label_3d, gif_3d_meta)
        except Exception as e:
            self.animation_label_2d.setText(f"Falha gerar GIF: {e}")
            self.animation_label_3d.setText(f"Falha gerar GIF: {e}")
        self.vis_function_plot_2d.update_best_point(best_pos_scaled)
        self.vis_function_plot_3d.update_best_point(best_pos_scaled)
        
        hist_raw = result.get("history_best", [])
        population_history = result.get("population_history", [])
        
        best_pos_raw = np.array(result.get("best_pos", [0, 0]))
        best_pos_scaled = best_pos_raw * 200 - 100
        
        if population_history:
            last_pop_raw = np.array(population_history[-1])
            last_pop_scaled = last_pop_raw * 200 - 100
            
            self.show_execution_report(
                algo_name="Algoritmo Genético",
                history_best=hist_raw,
                final_population=last_pop_scaled,
                best_position=best_pos_scaled,
                lower=-100, upper=100
            )

        self.ga_thread.quit()
        self.ga_thread.wait()
        self.ga_thread.quit()
        self.ga_thread.wait()
        
    def show_execution_report(self, algo_name, history_best, final_population, best_position, lower, upper):
        """
        Gera um popup com os dados calculados para o relatório.
        """
        # Melhor Fitness Final
        real_history = [-val for val in history_best]
        best_fitness = real_history[-1]

        # Iteração de Estabilização
        # Critério: Primeira iteração onde o fitness chegou muito próximo (1e-6) do valor final
        stabilization_iter = 0
        for i, val in enumerate(real_history):
            if abs(val - best_fitness) < 1e-6:
                stabilization_iter = i
                break
        
        # 3. % Vizinhança Final
        # Recalcula a porcentagem da população que está próxima do melhor indivíduo
        radius = 10.0 # O mesmo raio usado na tabela
        pop_arr = np.array(final_population)
        best_arr = np.array(best_position)
        
        neigh_pct = 0.0
        if pop_arr.size > 0:
            dists = np.linalg.norm(pop_arr - best_arr, axis=1)
            neigh_count = np.count_nonzero(dists <= radius)
            neigh_pct = (neigh_count / len(pop_arr)) * 100.0

        # Monta a mensagem
        msg = (
            f"=== RELATÓRIO FINAL: {algo_name} ===\n\n"
            f"Melhor f(x,y): {best_fitness:.6f}\n"
            f"Iteração de Estabilização: {stabilization_iter} (de {len(real_history)})\n"
            f"% Vizinhança Final: {neigh_pct:.2f}%\n"
            f"Melhor Posição: [{best_position[0]:.4f}, {best_position[1]:.4f}]\n"
        )

        # Exibe o Popup
        QMessageBox.information(self, f"Resultado {algo_name}", msg)
        
        # Opcional: Imprime no console para facilitar copiar
        print(msg)

    def _set_gif(self, label: QLabel, meta):
        # meta may be dict from animation with keys path, index_map, total_original
        if isinstance(meta, dict):
            path = meta.get("path")
            index_map = meta.get("index_map", [])
            total_original = meta.get("total_original", 0)
        else:
            path = meta
            index_map = []
            total_original = 0
        if not path or not isinstance(path, str):
            return
        movie = QMovie(path)
        if not movie.isValid():
            label.setText("GIF inválido")
            return
        label.setMovie(movie)
        movie.start()
        # Store reference & connect frame change
        if label is self.animation_label_2d:
            self.movie_2d = movie
            self.movie_2d_index_map = index_map
            self.movie_2d_total_original = total_original
            self._connect_movie(movie, self.animation_iter_label_2d, index_map, total_original)
        elif label is self.animation_label_3d:
            self.movie_3d = movie
            self.movie_3d_index_map = index_map
            self.movie_3d_total_original = total_original
            self._connect_movie(movie, self.animation_iter_label_3d, index_map, total_original)

    def _connect_movie(self, movie: QMovie, iter_label: QLabel, index_map, total_original):
        total_frames = movie.frameCount() if movie.frameCount() > 0 else 0
        # Show original total instead of downsampled count
        iter_label.setText(f"Iteração: 0/{total_original}")
        movie.frameChanged.connect(lambda f, m=movie, lbl=iter_label, im=index_map, to=total_original: self._on_movie_frame_changed(f, m, lbl, im, to))

    def _on_movie_frame_changed(self, frame_index: int, movie: QMovie, iter_label: QLabel, index_map, total_original):
        # Map sampled frame to original iteration index
        if index_map and frame_index < len(index_map):
            orig_iter = index_map[frame_index] + 1
        else:
            orig_iter = frame_index + 1
        iter_label.setText(f"Iteração: {orig_iter}/{total_original}")

    def _restart_current_animation(self, index: int):
        if index == 0 and hasattr(self, 'movie_2d'):
            self.movie_2d.stop(); self.movie_2d.start()
        elif index == 1 and hasattr(self, 'movie_3d'):
            self.movie_3d.stop(); self.movie_3d.start()

    # Helpers: fill result tables for all iterations
    def _fill_pso_table(self, positions_history):
        self.pso_table.setRowCount(0)
        radius = 10.0
        for i, pts in enumerate(positions_history, start=1):
            arr = np.array(pts)
            if arr.size == 0:
                continue
            # Best by objective (minimize)
            fvals = objective_function(arr)
            idx_best = int(np.argmin(fvals))
            best = arr[idx_best]
            fx = float(fvals[idx_best])
            dists = np.linalg.norm(arr - best, axis=1)
            neigh_pct = float(np.mean(dists <= radius) * 100.0)
            row = self.pso_table.rowCount()
            self.pso_table.insertRow(row)
            self.pso_table.setItem(row, 0, QTableWidgetItem(str(i)))
            self.pso_table.setItem(row, 1, QTableWidgetItem(f"{best[0]:.4f}"))
            self.pso_table.setItem(row, 2, QTableWidgetItem(f"{best[1]:.4f}"))
            self.pso_table.setItem(row, 3, QTableWidgetItem(f"{fx:.6f}"))
            self.pso_table.setItem(row, 4, QTableWidgetItem(f"{neigh_pct:.2f}%"))

    def _fill_ga_table(self, scaled_history):
        self.ga_table.setRowCount(0)
        radius = 10.0
        for i, pop in enumerate(scaled_history, start=1):
            arr = np.array(pop)
            if arr.size == 0:
                continue
            fvals = objective_function(arr)
            idx_best = int(np.argmin(fvals))
            best = arr[idx_best]
            fx = float(fvals[idx_best])
            dists = np.linalg.norm(arr - best, axis=1)
            neigh_pct = float(np.mean(dists <= radius) * 100.0)
            row = self.ga_table.rowCount()
            self.ga_table.insertRow(row)
            self.ga_table.setItem(row, 0, QTableWidgetItem(str(i)))
            self.ga_table.setItem(row, 1, QTableWidgetItem(f"{best[0]:.4f}"))
            self.ga_table.setItem(row, 2, QTableWidgetItem(f"{best[1]:.4f}"))
            self.ga_table.setItem(row, 3, QTableWidgetItem(f"{fx:.6f}"))
            self.ga_table.setItem(row, 4, QTableWidgetItem(f"{neigh_pct:.2f}%"))

    # Config randomization and history
    def _random_pso_config(self):
        rng = np.random.default_rng()
        return {
            "num_particles": int(rng.integers(20, 80)),
            "num_iterations": int(rng.integers(100, 300)),
            "dimension": 2,
            "inertia": float(rng.uniform(0.4, 0.9)),
            "cognitive": float(rng.uniform(1.0, 2.5)),
            "social": float(rng.uniform(1.0, 2.5)),
            "lower_bound": -100,
            "upper_bound": 100,
            "rng_seed": int(rng.integers(1, 10000)),
        }

    def _random_ga_config(self):
        rng = np.random.default_rng()
        pop = int(rng.integers(20, 80))
        if pop % 2 == 1:
            pop += 1
        return {
            "population_size": pop,
            "generations": int(rng.integers(100, 300)),
            "chromosome_length": 2,
            "mutation_rate": float(rng.uniform(0.02, 0.2)),
            "crossover_rate": float(rng.uniform(0.5, 0.9)),
            "rng_seed": int(rng.integers(1, 10000)),
        }

    def _append_history(self, algo: str, cfg: dict):
        try:
            out_dir = os.path.join("project", "ui", "output")
            os.makedirs(out_dir, exist_ok=True)
            hist_path = os.path.join(out_dir, "run_history.json")
            data = []
            if os.path.exists(hist_path):
                with open(hist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            entry = {"algo": algo, "config": cfg}
            data.append(entry)
            # Keep last 50
            data = data[-50:]
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)  
            # Also persist tested parameters for the UI table with timestamp
            tested_path = os.path.join(out_dir, "tested_parameters.json")
            tested_data = []
            if os.path.exists(tested_path):
                try:
                    with open(tested_path, "r", encoding="utf-8") as tf:
                        tested_data = json.load(tf)
                except Exception:
                    tested_data = []
            ui_entry = {
                "algorithm": algo.upper(),
                "config": cfg,
                "timestamp": self._current_timestamp(),
            }
            tested_data.append(ui_entry)
            tested_data = tested_data[-100:]
            with open(tested_path, "w", encoding="utf-8") as tf:
                json.dump(tested_data, tf, indent=2)
            self._add_row_to_params_table(ui_entry)
        except Exception:
            # Silent fail for history
            pass

    def _show_config(self, algo: str, cfg: dict, randomized: bool):
        text = "Configuração: " + ("aleatória" if randomized else "padrão") + " | "
        parts = []
        for k, v in cfg.items():
            parts.append(f"{k}={v}")
        text += ", ".join(parts)
        if algo == "pso":
            self.pso_cfg_label.setText(text)
        elif algo == "ga":
            self.ga_cfg_label.setText(text)
    
    def _load_tested_history(self):
        path = "project/ui/output/tested_parameters.json"
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except:
            return

        for entry in data:
            self._add_row_to_params_table(entry)
            
    def _add_row_to_params_table(self, entry):
        row = self.params_table.rowCount()
        self.params_table.insertRow(row)

        algo = entry.get("algorithm", "")
        cfg = json.dumps(entry.get("config", {}), indent=2)
        timestamp = entry.get("timestamp", "")

        self.params_table.setItem(row, 0, QTableWidgetItem(algo))
        self.params_table.setItem(row, 1, QTableWidgetItem(cfg))
        self.params_table.setItem(row, 2, QTableWidgetItem(timestamp))
    
    def _clear_tested_history(self):
        path = "project/ui/output/tested_parameters.json"
        if os.path.exists(path):
            os.remove(path)

        self.params_table.setRowCount(0)
        
    def _current_timestamp(self):
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


