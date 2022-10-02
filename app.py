import random

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
    """Interact with the latent components of a VAE model
    and see the resulting decoded image.
    """
    def __init__(
        self,
        model: VAE,
        pca: PCAComponents,
        device: str = 'cpu',
        size: int = 800,
        nscales: int = 12,
        vscales: int = 3,
    ):
        super().__init__()

        self.model = model
        self.pca = pca
        self.device = device

        self.model.to(self.device)

        self.width = size
        self.height = size // 2
        self.nscales = nscales
        self.vscales = vscales
        # self.tk.call('tk', 'scaling', 4.0)

        self.title('AnimeVAE')
        self.geometry(f'{self.width}x{self.height}')

        self._init_w()
        self._build_scale_frame()
        self._build_image_frame()
        self.image_frame.pack(side=tk.RIGHT)
        self.scale_frame.pack(side=tk.LEFT)

        self.move_directions = [
            1 if random.random() > 0.5 else -1
            for _ in range(nscales)
        ]
        self.is_animating = False

    def _init_w(self):
        """Instanciate a new weight `w` of shape [nscales,].
        """
        self.w = torch.randn(self.nscales)

    def _build_scale_frame(self):
        self.scale_frame = tk.Frame(self)

        # Scales
        frame1 = tk.Frame(self.scale_frame)
        self.scales_var = [
            tk.DoubleVar(value=self.w[i].item())
            for i in range(self.nscales)
        ]

        scales = [
            tk.Scale(
                frame1,
                from_=-self.vscales,
                to=self.vscales,
                resolution=0.1,
                variable=v,
                command=lambda _: self.produce_image(),
            )
            for v in self.scales_var
        ]

        for i, s in enumerate(scales):
            s.grid(row=i//6, column=i%6)

        # Buttons
        frame2 = tk.Frame(self.scale_frame)
        tk.Button(
            frame2,
            command=self.random_image,
            text='Randomize',
            width=15,
            height=5,
        ).pack(side=tk.LEFT)

        tk.Button(
            frame2,
            command=self.toggle_animation,
            text='Animate',
            width=15,
            height=5,
        ).pack(side=tk.RIGHT, padx=5)

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
        """Update each weight of `w` according the the scales and then
        compute the decoded image of the corresponding latent vector.
        """
        for i, v in enumerate(self.scales_var):
            self.w[i] = v.get()

        z = self.pca.compute_latent(self.w).to(self.device)
        image = self.model.generate_from_z(z)[0]

        image = image.permute(1, 2, 0).cpu().numpy()  # To numpy
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
        """Init a new `w` and update the image.
        """
        self._init_w()
        for i, v in enumerate(self.scales_var):
            v.set(self.w[i].item())
        self.produce_image()

    def move_around(self):
        """Move values of `w` randomly, to a local neighbour.
        """
        for i, v in enumerate(self.scales_var):
            w = v.get() + self.move_directions[i] * 0.05
            w = min(max(w, -self.vscales), self.vscales)
            v.set(w)

            if abs(w) == self.vscales:  # If we're at a frontier
                self.move_directions[i] *= -1  # Inverse the direction of future moves
        self.produce_image()

    def animation(self):
        """Repeatedly calls the `self.move_around` method.
        """
        self.move_around()
        if self.is_animating:  # Keep going while this variable is true
            self.after(10, lambda: self.animation())

    def toggle_animation(self):
        """Toggle on/off the animation and launches it if necessary.
        """
        self.is_animating = not self.is_animating
        if self.is_animating:
            self.animation()



def main(model_path: str, config_path: str, device: str):
    """Load the model, compute the PCA components and launch the app.
    """
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

    app = App(model, pca, device)
    app.mainloop()


if __name__ == '__main__':
    model_path = 'models/vae.pth'
    config_path = 'models/vae.yaml'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(model_path, config_path, device)
