import tkinter as tk
from PIL import Image, ImageTk

import numpy as np
import torch

from VAE import load_model


class App(tk.Tk):
    def __init__(self, model, components, size=800, nscales=6):
        super().__init__()

        self.model = model
        self.components = components

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
        self.z = torch.randn(self.model.latent_size) * 10

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
                from_=-10,
                to=10,
                resolution=0.1,
                variable=v,
                command=lambda val: self.produce_image(),
            )
            for v in self.scales_var
        ]

        for i, s in enumerate(scales):
            s.grid(row=0, column=i%6)

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

    def produce_image(self):
        for i, v in enumerate(self.scales_var):
            self.z[i] = v.get()

        z_latent = 0
        for scale, component in zip(self.z, self.components):
            z_latent += scale * component

        with torch.no_grad():
            z = z_latent.unsqueeze(0)  # Batch dim
            image = self.model.decode(z)[0]
            image = torch.sigmoid(image)

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


if __name__ == '__main__':
    config = {
        'nc': 3,
        'nfilters': 64,
        'latent_size': 128,
        'nlayers': 3,
        'res_layers': 6,
    }
    model = load_model(config, 'hopeful_planet_112.pt')
    components = np.load('components.npy')
    components = torch.FloatTensor(components)

    app = App(model, components)
    app.mainloop()
