import numpy as np
from scipy.ndimage import zoom


def generator(images, masks):
    while True:
        x_batch, y_batch = [], []

        for i in range(8):

            s = np.random.randint(images.shape[0])
            z = np.random.uniform(0.5, 1.5)
            size = image.shape[1]

            img = images[s]
            msk = masks[s]

            img_z = zoom(img, [z,1,1,1])
            msk_z = zoom(msk, [z,1,1,1])

            if z > 1:
                img = img_z[:size,...]
                msk = msk_z[:size,...]
            elif z < 1:
                p = img.shape[0] - img_z.shape[0]
                img = np.concatenate((img_z, img_z[:p,...]), axis=0)
                msk = np.concatenate((msk_z, msk_z[:p,...]), axis=0)
            else:
                pass
            
            x_batch.append(img)
            y_batch.append(msk)

        yield np.array(x_batch), np.array(y_batch)