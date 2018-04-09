import numpy as np
from scipy.ndimage import zoom


def generator(images, masks):
    while True:
        x_batch, y_batch = [], []

        for i in range(8):

            s = np.random.randint(images.shape[0])


            img = images[s]
            msk = masks[s]



            x_batch.append(img)
            y_batch.append(msk)

        yield np.array(x_batch), np.array(y_batch)