import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import torch


def decode_latent_space(model, latent_vector):
    model.eval()
    latent_vector = torch.from_numpy(latent_vector).float()
    latent_vector = latent_vector.unsqueeze(0)
    decoded_image = model.decoder(latent_vector)
    decoded_image = decoded_image.detach().numpy()
    decoded_image = decoded_image.squeeze(0)
    decoded_image = decoded_image.transpose(1, 2, 0)
    decoded_image = decoded_image * 255
    decoded_image = decoded_image.astype(np.uint8)
    return decoded_image


class LatentSpaceExplorer:
    def __init__(self, model=None, latent_dim=50):
        self.latent_dim = latent_dim
        self.latent_vector = np.zeros(latent_dim)
        self.model = model

        self.root = tk.Tk()
        self.root.title("Latent Space Explorer")

        self.create_gui()

    def create_gui(self):
        # Create sliders for each dimension organized in a 5x10 grid
        rows = 10
        cols = 5
        self.sliders = []
        for i in range(rows):
            for j in range(cols):
                index = i * cols + j
                slider = ttk.Scale(self.root, from_=-3, to=3, orient="horizontal", length=100,
                                   command=lambda val, index=index: self.update_latent_vector(val, index))
                slider.grid(row=i, column=j, padx=5, pady=5)
                self.sliders.append(slider)

        # Create canvas for displaying decoded image with increased size
        self.canvas = tk.Canvas(self.root, width=500, height=500)
        self.canvas.grid(row=0, column=cols, rowspan=rows, padx=10, pady=10)

        # Initial update of the canvas
        self.update_canvas()

    def update_latent_vector(self, value, index):
        self.latent_vector[index] = float(value)
        self.update_canvas()

    def update_canvas(self):
        decoded_image = decode_latent_space(self.model, self.latent_vector)
        self.display_image(decoded_image)

    def display_image(self, image_array):
        # Convert NumPy array to PhotoImage
        img = Image.fromarray(image_array.astype('uint8'))
        img = ImageTk.PhotoImage(img.resize((500, 500), Image.LANCZOS))

        # Update canvas with the new image
        self.canvas.config(width=img.width(), height=img.height())
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self.canvas.image = img

    def run(self):
        self.root.mainloop()