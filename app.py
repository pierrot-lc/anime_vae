import yaml
import torch
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from torch.utils.data import DataLoader

from src.vae import VAE
from src.dataset import load_datasets
from src.pca import PCAComponents


class App(tk.Tk):
    def __init__(
        self,
        model: VAE,
        pca: PCAComponents,
        size: int = 800,
        nscales: int = 10,
    ):
        super().__init__()

        self.model = model
        self.pca = pca

        self.width = size
        self.height = size // 2
        self.nscales = nscales

        self.title('AnimeVAE')
        self.geometry(f'{self.width}x{self.height}')

        self._init_z()
        self._build_scale_frame()
        self._build_image_frame()
        self.image_frame.pack(side=tk.RIGHT)
        self.scale_frame.pack(side=tk.LEFT)

    def _init_z(self):
        self.z = torch.randn(self.nscales)

    def _build_scale_frame(self):
        self.scale_frame = tk.Frame(self)

        # Scales
        frame1 = tk.Frame(self.scale_frame)
        self.scales_var = [
            tk.DoubleVar(value=self.z[i].item())
            for i in range(self.nscales)
        ]

        scales = [
            tk.Scale(
                frame1,
                from_=-2,
                to=2,
                resolution=0.1,
                variable=v,
                command=lambda _: self.produce_image(),
            )
            for v in self.scales_var
        ]

        for i, s in enumerate(scales):
            s.grid(row=i//6, column=i%6)

        # Random button
        frame2 = tk.Frame(self.scale_frame)
        tk.Button(
            frame2,
            command=self.random_image,
            text='Randomize',
            width=30,
            height=5,
        ).pack()

        frame1.pack(side=tk.TOP)
        frame2.pack(side=tk.BOTTOM, pady=40)

    def _build_image_frame(self):
        self.image_frame = tk.Frame(self)
        self.image_canvas = tk.Canvas(
            self.image_frame,
            width=self.height,
            height=self.height,
        )
        self.image_canvas.pack()

        self.produce_image()

    @torch.no_grad()
    def produce_image(self):
        for i, v in enumerate(self.scales_var):
            self.z[i] = v.get()

        z = self.pca.compute_latent(self.z)
        image = self.model.generate_from_z(z)[0]

        image = image.permute(1, 2, 0).numpy()  # To numpy
        image = np.uint8(image * 255)
        image = Image.fromarray(image)  # To PIL
        image = image.resize((self.height, self.height))
        image = ImageTk.PhotoImage(image=image)  # To ImageTk

        self.current_image = image
        self.image_canvas.create_image(
            0,
            0,
            anchor='nw',
            image=self.current_image,
        )

    def random_image(self):
        self._init_z()
        for i, v in enumerate(self.scales_var):
            v.set(self.z[i].item())
        self.produce_image()


def main(model_path: str, config_path: str):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    model = VAE.load_from_config(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    train_set, _ = load_datasets(
        config['data']['path_dir'],
        config['data']['image_size'],
        config['train']['seed'],
    )
    loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True)
    batch = next(iter(loader))

    pca = PCAComponents()
    pca.compute_components(model, batch)

    app = App(model, pca)
    app.mainloop()

if __name__ == '__main__':
    model_path = 'models/vae.pth'
    config_path = 'models/vae.yaml'
    main(model_path, config_path)
