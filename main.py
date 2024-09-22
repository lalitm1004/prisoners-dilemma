import math
import numpy as np
from tqdm import tqdm

from Grid import Grid
from Render import Renderer

C_C = 1
C_D = 2
D_D = 3
D_C = 4

ROWS = 51
COLS = 51
INITAL_COOPERATION = 0.5
MAX_BENEFIT = 100
SAVE_NAME = '51x51-single-defector'

# initial_state = Grid.generate_state(ROWS, COLS, INITAL_COOPERATION)
initial_state = np.full((ROWS, COLS), C_C)
initial_state[math.ceil(ROWS // 2), math.ceil(COLS // 2)] = D_D

for curr_cost in tqdm(range(MAX_BENEFIT + 1), desc='generating trajectories'):
    options = {
        'state': np.array(initial_state),
        'cost': curr_cost,
        'benefit': MAX_BENEFIT,
        'save_folder': SAVE_NAME,
    }
    metadata = {
        'num_rows': ROWS,
        'num_cols': COLS,
    }
    grid = Grid(options, metadata)
    for i in range(100):
        grid.iterate()

renderer = Renderer()
renderer.render(SAVE_NAME)