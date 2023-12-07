import torch
import numpy as np
from matplotlib import animation as animation
import matplotlib.pyplot as plt


def interpolate(model, points: np.array, step_size, name='interpolation.gif'):
    # interpolate between points in the latent space with a given step size
    interpolated_points = []
    current_point = 0
    num_points = len(points)

    point = points[0]
    vector = points[1] - points[0]
    vector_norm = np.linalg.norm(vector)
    vector_step = (vector / vector_norm) * step_size

    interpolated_points.append(point)
    next_point = point

    while current_point < num_points - 1:
        next_point = next_point + vector_step

        interpolated_points.append(next_point)

        if np.linalg.norm(point - next_point) >= np.linalg.norm(points[current_point + 1] - point):
            current_point += 1
            point = points[current_point]
            if current_point < num_points - 1:
                vector = points[current_point + 1] - points[current_point]
                vector_norm = np.linalg.norm(vector)
                vector_step = (vector / vector_norm) * step_size
            else:
                break

    interpolated_points = torch.Tensor(interpolated_points)

    # decode the interpolated points
    decoded_points = model.decoder(interpolated_points)

    # create a gif
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    images = []

    for i in range(decoded_points.shape[0]):
        images.append([ax.imshow(decoded_points[i, 0].detach().numpy(), cmap='gray', animated=True)])

    anim = animation.ArtistAnimation(fig, images, interval=200, blit=True, repeat_delay=1000)
    anim.save('gifs/' + name, writer='imagemagick')

    return anim