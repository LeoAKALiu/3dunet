import numpy as np
from scipy import signal, ndimage


def shift(image, mask):

    new_img = np.copy(image)
    new_msk = np.copy(mask)

    shape = new_img.shape
    max_x = int(shape[0] * 0.2)
    max_y = int(shape[1] * 0.2)

    x = np.random.randint(low=-max_x, high=max_x)
    y = np.random.randint(low=-max_y, high=max_y)

    img = ndimage.interpolation.shift(new_img,shift=[x,y,0])
    msk = ndimage.interpolation.shift(new_msk,shift=[x,y,0])

    return img, msk

def flipx(image, mask):

    new_img = np.copy(image)
    new_msk = np.copy(mask)

    return new_img[::-1,...], new_msk[::-1,...]

def flipy(image, mask):

    new_img = np.copy(image)
    new_msk = np.copy(mask)

    return new_img[:,::-1,:], new_msk[:,::-1,:]

def rotate(image, mask):

    randnum = np.random.randint(1,360)
    new_img = np.copy(image)
    new_msk = np.copy(mask)

    for i in range(image.shape[2]):
        new_img[...,i] = ndimage.rotate(new_img[...,i], angle=randnum, reshape=False)
        new_msk[...,i] = ndimage.rotate(new_msk[...,i], angle=randnum, reshape=False)

    return new_img, new_msk


def zoom_xy(image, mask):

    '''apply zoom on axis x and y randomly,
    image.shape = (height, width, size) '''

    shape = image.shape
    half = int(image.shape[1]/2)

    z = np.random.uniform(0.5,1.5)

    image = ndimage.zoom(image, [z,z,1])
    mask = ndimage.zoom(mask, [z,z,1])

    center = (np.array(image.shape) / 2).astype(int)
    if z > 1:
        image = image[(center[0]-half):(center[0]+half),(center[1]-half):(center[1]+half),:]
        mask = mask[(center[0]-half):(center[0]+half),(center[1]-half):(center[1]+half),:]

        return image, mask
    elif z < 1:
        img = np.zeros(shape)
        msk = np.zeros(shape)

        img[:image.shape[0], :image.shape[1],:] = image
        msk[:mask.shape[0], :mask.shape[1],:]  = mask

        return img, msk
    else:

        return image, mask


def zoom_z(image, mask):

    '''apply zoom on axis z randomly,
    image.shape = (height, width, size) '''

    z = np.random.uniform(0.5, 1.5)
    size = image.shape[2]

    img_z = ndimage.zoom(image, [1,1,z])
    msk_z = ndimage.zoom(mask, [1,1,z])

    if z > 1:
        img = img_z[...,:size]
        msk = msk_z[...,:size]
    elif z < 1:
        p = image.shape[2] - img_z.shape[2]
        img = np.concatenate((img_z, img_z[...,:p]), axis=-1)
        msk = np.concatenate((msk_z, msk_z[...,:p]), axis=-1)
    else:
        img = img_z
        msk = msk_z

    return img, msk


def data_augmentation(image, mask, size):

    '''image.shape = (height, width, n_samples)'''

    for i in range(0, image.shape[2], size):

        ops = {
                0: flipx,
                1: shift,
                2: flipy,
                3: rotate,
                4: zoom_xy,
                5: zoom_z
            }

        which_op = np.random.randint(0, 6)

        end = min(image.shape[2], i+size)

        image[...,i:end], mask[...,i:end] = ops[which_op](image[...,i:end], mask[...,i:end])

    return image, mask