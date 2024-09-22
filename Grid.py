import numpy as np
from pathlib import Path
import os

C_C = 1
C_D = 2
D_D = 3
D_C = 4

COOPERATORS = [C_C, D_C]
DEFECTORS = [C_D, D_D]

RESULTS_PATH = Path('./results')

class Grid:
    def __init__(self, options: dict, metadata: dict) -> None:
        self.state: np.ndarray = np.array(options.get('state'))
        self.cost: int = options.get('cost')
        self.benefit: int = options.get('benefit')
        self.save_folder: str = options.get('save_folder')

        self.metadata: dict = metadata
        self.num_rows: int = metadata.get('num_rows')
        self.num_cols: int = metadata.get('num_cols')

        self.iteration_count: int = 0
        self.short_term_history: list[np.ndarray] = []
        self.stable = False
        self.save()

    @staticmethod
    def generate_state(num_cols:int, num_rows:int, intial_cooperation: float) -> np.ndarray:
        state = np.full((num_rows, num_cols), D_D, dtype=int)
        total_elements = num_rows * num_cols
        num_cooperators = int(total_elements * intial_cooperation)
        cooperator_indices = np.random.choice(total_elements, num_cooperators, replace=False)
        state.flat[cooperator_indices] = C_C
        return state

    def save(self) -> None:
        # save as results/save_folder/trajectories/ccc-bbb/num_iteration.csv
        save_folder_path = RESULTS_PATH / self.save_folder
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)
            os.mkdir(save_folder_path / 'trajectories')
            os.mkdir(save_folder_path / 'renders')
            os.mkdir(save_folder_path / 'frames')


        ccc_bbb = f'{self.cost}'.rjust(3, '0') + '_' + f'{self.benefit}'.rjust(3, '0')
        ccc_bbb_path = RESULTS_PATH / self.save_folder / 'trajectories' / ccc_bbb
        if not os.path.exists(ccc_bbb_path):
            os.mkdir(ccc_bbb_path)

        save_file_path = ccc_bbb_path / f'{self.iteration_count}.csv'.rjust(3 + 4, '0')
        np.savetxt(save_file_path, self.state, delimiter=',', fmt='%d')

    def calculate_payoff(self, cell_strat: int, neighbour_strat: int) -> int:
        if cell_strat in COOPERATORS:
            if neighbour_strat in COOPERATORS:
                return self.benefit - self.cost
            if neighbour_strat in DEFECTORS:
                return -1 * self.cost

        if cell_strat in DEFECTORS:
            if neighbour_strat in COOPERATORS:
                return self.benefit
            if neighbour_strat in DEFECTORS:
                return 0

    def is_cooperator(self, strat: int) -> bool:
        return strat in COOPERATORS

    def iterate(self) -> None:

        if self.stable:
            return

        payoff_mat = np.zeros((self.num_rows, self.num_cols), dtype=int)

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                cell_strat = self.state[i, j]
                total_payoff: int = 0

                if (i > 0):
                    total_payoff += self.calculate_payoff(cell_strat, self.state[i - 1, j])

                if (i < self.num_rows - 1):
                    total_payoff += self.calculate_payoff(cell_strat, self.state[i + 1, j])

                if (j > 0):
                    total_payoff += self.calculate_payoff(cell_strat, self.state[i, j - 1])

                if (j < self.num_cols - 1):
                    total_payoff += self.calculate_payoff(cell_strat, self.state[i, j + 1])

                payoff_mat[i, j] = total_payoff

        copy_state = np.array(self.state)
        if (len(self.short_term_history) >= 5):
            self.short_term_history.pop(0)
        self.short_term_history.append(np.array(self.state))

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                current_strat = copy_state[i, j]
                local_payoffs = []
                local_strategies = []

                local_payoffs.append(payoff_mat[i, j])
                local_strategies.append(copy_state[i, j])

                if (i > 0):
                    local_payoffs.append(payoff_mat[i - 1, j])
                    local_strategies.append(copy_state[i - 1, j])

                if (i < self.num_rows - 1):
                    local_payoffs.append(payoff_mat[i + 1, j])
                    local_strategies.append(copy_state[i + 1, j])

                if (j > 0):
                    local_payoffs.append(payoff_mat[i, j - 1])
                    local_strategies.append(copy_state[i, j - 1])

                if (j < self.num_cols - 1):
                    local_payoffs.append(payoff_mat[i, j + 1])
                    local_strategies.append(copy_state[i, j + 1])

                max_payoff = max(local_payoffs)
                best_indices = [index for index, value in enumerate(local_payoffs) if value == max_payoff]

                if len(best_indices) > 1:
                    for index in best_indices:
                        neighbour_strat = local_strategies[index]
                        if (
                            (self.is_cooperator(current_strat) and self.is_cooperator(neighbour_strat)) or
                            (not self.is_cooperator(current_strat) and not self.is_cooperator(neighbour_strat))
                        ):
                            will_cooperate_next = self.is_cooperator(neighbour_strat)
                            break
                        else:
                            will_cooperate_next = self.is_cooperator(neighbour_strat)
                else:
                    # pick the singular best strategy
                    will_cooperate_next = self.is_cooperator(local_strategies[best_indices[0]])

                if will_cooperate_next:
                    # will evolve into cooperator
                    if current_strat in COOPERATORS:
                        # currently a cooperator
                        next_strat = C_C
                    elif current_strat in DEFECTORS:
                        # currently a defector
                        next_strat = D_C
                else:
                    # will evolve into defector
                    if current_strat in COOPERATORS:
                        # currently a cooperator
                        next_strat = C_D
                    elif current_strat in DEFECTORS:
                        # currently a defector
                        next_strat = D_D
                self.state[i, j] = next_strat

        self.iteration_count += 1
        self.save()
        for history_entry in self.short_term_history:
            if (history_entry == self.state).all():
                self.stable = True
                break

if __name__ == "__main__":
    ROWS = 25
    COLS = 25
    COOP = 0.99
    COST = 1
    BENEFIT = 100

    state = [
        [1, 1, 1, 1, 1,],
        [1, 3, 3, 3, 1,],
        [1, 3, 3, 3, 1,],
        [1, 3, 3, 3, 1,],
        [1, 1, 1, 1, 1,],
    ]

    initial_state = Grid.generate_state(ROWS, COLS, COOP)
    options = {
        # 'state': np.array(state),
        'state': initial_state,
        'cost': COST,
        'benefit': BENEFIT,
        'save_folder': 'test',
    }
    metadata = {
        'num_cols': COLS,
        'num_rows': ROWS,
        'initial_cooperation': COOP,
    }
    grid = Grid(options, metadata)
    for i in range(50):
        grid.iterate()