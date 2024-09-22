import csv
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageColor
from tqdm import tqdm

color_map = {
    1: ("#6488EA", "C_C"),
    2: ("#FFF078", "C_D"),
    3: ("#FF2C2C", "D_D"),
    4: ("#6FC276", "D_C"),
}

output_size = (2000, 2000)
matrix_size = (1600, 1600)

RESULTS_PATH = Path('./results')

class Renderer:
    def __init__(self) -> None:
        pass

    def csv_to_img(self, file_path: Path, trajectory_name: str, iteration_num: int) -> Image:
        mat = np.loadtxt(file_path, delimiter=',', dtype=int)
        height, width = mat.shape
        tile_size = matrix_size[0] // mat.shape[0]

        img = Image.new('RGB', output_size, color='white')
        draw = ImageDraw.Draw(img)

        start_x = (output_size[0] - matrix_size[0]) // 2
        start_y = (output_size[1] - matrix_size[1]) // 2

        for y in range(height):
            for x in range(width):
                color = ImageColor.getrgb(color_map[mat[y, x]][0])
                draw.rectangle(
                    [
                        start_x + x * tile_size,
                        start_y + y * tile_size,
                        start_x + (x + 1) * tile_size,
                        start_y + (y + 1) * tile_size,
                    ],
                    fill=color,
                )

        font_size = 60
        font = ImageFont.truetype('arial.ttf', font_size)
        text_color = (0, 0, 0)
        draw.text((200, 100), f'cost_benefit: {trajectory_name}', font=font, fill=text_color)
        draw.text((output_size[0] - 200, output_size[1] - 100), f'{iteration_num}', font=font, fill=text_color)

        return img

    def render(self, folder_name: Path) -> None:
        trajectories_path = RESULTS_PATH / folder_name / 'trajectories'
        renders_path = RESULTS_PATH / folder_name / 'renders'
        all_trajectories = os.listdir(trajectories_path)
        current_index = 0
        for trajectory in all_trajectories:
            trajectory_path = trajectories_path / trajectory

            frames = []
            
            iteration_num = 0
            for filename in tqdm(os.listdir(trajectory_path), f'rendering {trajectory}'):
                if not filename.endswith('.csv'):
                    continue

                file_path = trajectory_path / filename
                img = self.csv_to_img(file_path, trajectory, iteration_num)
                frames.append(img)
                iteration_num += 1

            output_file = renders_path / f'{trajectory}.gif'
            frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=100, loop=0)
            current_index += 1

    def render_frame(self, folder: Path, ccc_bbb:Path, iteration_num: int) -> None:
        csv_path = RESULTS_PATH / folder / 'trajectories' / ccc_bbb / f'{iteration_num}.csv'.rjust(3 + 4, '0')
        save_folder = RESULTS_PATH / folder/ 'frames' / ccc_bbb 
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        filename = folder + '_' + ccc_bbb + '_' +  f'{iteration_num}.png'.rjust(3 + 4, '0')
        save_path = save_folder / filename
        img = self.csv_to_img(csv_path, ccc_bbb, iteration_num)
        img.save(save_path)
    
if __name__ == '__main__':
    renderer = Renderer()
    # renderer.render('test')
    for i in range(20,21):
        renderer.render_frame('51x51-single-defector', '051_100', i)