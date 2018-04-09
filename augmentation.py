import numpy as np
from scipy import signal, ndimage


def shift(image, mask):

    new_img = np.copy(image)
    new_msk = np.copy(mask)

    shape = new_img.shape
    max_x = int(shape[1] * 0.2)
    max_y = int(shape[2] * 0.2)

    x = np.random.randint(low=-max_x, high=max_x)
    y = np.random.randint(low=-max_y, high=max_y)

    img = ndimage.interpolation.shift(new_img,shift=[0,x,y])
    msk = ndimage.interpolation.shift(new_msk,shift=[0,x,y])

    return img, msk

def flipx(image, mask):

    new_img = np.copy(image)
    new_msk = np.copy(mask)

    return new_img[:,::-1,:], new_msk[:,::-1,:]

def flipy(image, mask):

    new_img = np.copy(image)
    new_msk = np.copy(mask)

    return new_img[:,:,::-1], new_msk[:,:,::-1]

def rotate(image, mask):

    randnum = np.random.randint(1,360)
    new_img = np.copy(image)
    new_msk = np.copy(mask)

    for i in range(image.shape[0]):
        img[i,...] = ndimage.rotate(new_img[i,...], angle=randnum, reshape=False)
        msk[i,...] = ndimage.rotate(new_msk[i,...], angle=randnum, reshape=False)

    return img, msk


def zoom_xy(image, mask):

    '''apply zoom on axis x and y randomly,
    image.shape = (patch_size, height, width) '''

    shape = image.shape
    half = int(image.shape[1]/2)

    z = np.random.uniform(0.5,1.5)

    image = ndimage.zoom(image, [1,z,z])
    mask = ndimage.zoom(mask, [1,z,z])

    center = (np.array(image.shape) / 2).astype(int)
    if z > 1:
        image = image[:,(center[1]-half):(center[1]+half),(center[2]-half):(center[2]+half)]
        mask = mask[:,(center[1]-half):(center[1]+half),(center[2]-half):(center[2]+half)]

        return image, mask
    elif z < 1:
        img = np.zeros(shape)
        msk = np.zeros(shape)

        img[:,:image.shape[1], :image.shape[2]] = image
        msk[:, :mask.shape[1], :mask.shape[2]]  = mask

        return img, msk
    else:

        return image, mask


def zoom_z(image, mask):

    '''apply zoom on axis z randomly,
    image.shape = (patch_size, height, width) '''

    z = np.random.uniform(0.5, 1.5)
    size = image.shape[0]

    img_z = ndimage.zoom(image, [z,1,1])
    msk_z = ndimage.zoom(mask, [z,1,1])

    if z > 1:
        img = img_z[:size, ...]
        msk = msk_z[:size, ...]
    elif z < 1:
        p = image.shape[0] - img_z.shape[0]
        img = np.concatenate((img_z, img_z[:p, ...]), axis=0)
        msk = np.concatenate((msk_z, msk_z[:p, ...]), axis=0)
    else:
        img = img_z
        msk = msk_z

    return img, msk


def data_augmentation(image, mask):

    '''image.shape = (n_samples, patch_size, height, width, channel)'''


    for i in range(image.shape[0]):

        ops = {
                0: flipx,
                1: shift,
                2: flipy,
                3: rotate,
                4: zoom,
                5: zoom_z
            }

        which_op = np.random.randint(0, 6)

        image[i,...,:], mask[i,...,:]= ops[which_op](image[i,...,:], mask[i,...,:])

    return image, mask