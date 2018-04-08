import numpy as np
from scipy import signal, ndimage



def shift(image, max_amt=0.2, seed=42):

    new_img = np.copy(image)
    shape = new_img.shape
    max_x = int(shape[0] * max_amt)
    max_y = int(shape[1] * max_amt)
    np.random.seed(seed)
    x = np.random.randint(low=-max_x, high=max_x)
    np.random.seed(seed)
    y = np.random.randint(low=-max_y, high=max_y)

    return ndimage.interpolation.shift(new_img,shift=[x,y])

def flipx(image, seed=42):

    new_img = np.copy(image)

    return new_img[::-1, :]

def flipy(image, seed=42):

    new_img = np.copy(image)

    return new_img[:, ::-1]

def rotate(image, seed=42):
    np.random.seed(seed)
    randnum = np.random.randint(1,360)
    new_image = np.copy(image)
    return ndimage.rotate(new_image, angle=randnum, reshape=False)

# def elastic_distortion(image, kernel_dim=5, sigma=6, alpha=47, seed=42):
#
#     # Returns gaussian kernel in two dimensions
#     # d is the square kernel edge size, it must be an odd number.
#     # i.e. kernel is of the size (d,d)
#     def gaussian_kernel(d, sigma):
#         if d % 2 == 0:
#             raise ValueError("Kernel edge size must be an odd number")
#
#         cols_identifier = np.int32(
#             np.ones((d, d)) * np.array(np.arange(d)))
#         rows_identifier = np.int32(
#             np.ones((d, d)) * np.array(np.arange(d)).reshape(d, 1))
#
#         kernel = np.exp(-1. * ((rows_identifier - d/2)**2 + (cols_identifier - d/2)**2) / (2. * sigma**2))
#         kernel *= 1. / (2. * np.pi * sigma**2)  # normalize
#         return kernel
#
#     np.random.seed(seed)
#     field_x = np.random.uniform(low=-1, high=1, size=(image.shape[1], image.shape[1])) * alpha
#     np.random.seed(seed)
#     field_y = np.random.uniform(low=-1, high=1, size=(image.shape[1], image.shape[1])) * alpha
#
#     kernel = gaussian_kernel(kernel_dim, sigma)
#
#     # Distortion fields convolved with the gaussian kernel
#     # This smoothes the field out.
#     field_x = signal.convolve2d(field_x, kernel, mode="same")
#     field_y = signal.convolve2d(field_y, kernel, mode="same")
#
#     d = image.shape[1]
#     cols_identifier = np.int32(np.ones((d, d))*np.array(np.arange(d)))
#     rows_identifier = np.int32(
#         np.ones((d, d))*np.array(np.arange(d)).reshape(d, 1))
#
#     down_row = np.int32(np.floor(field_x)) + rows_identifier
#     top_row = np.int32(np.ceil(field_x)) + rows_identifier
#     down_col = np.int32(np.floor(field_y)) + cols_identifier
#     top_col = np.int32(np.ceil(field_y)) + cols_identifier
#
#
#     padded_image = np.pad(
#         image, pad_width=d, mode="constant", constant_values=0)
#
#     x1 = down_row.flatten()
#     y1 = down_col.flatten()
#     x2 = top_row.flatten()
#     y2 = top_col.flatten()
#
#     Q11 = padded_image[d+x1, d+y1]
#     Q12 = padded_image[d+x1, d+y2]
#     Q21 = padded_image[d+x2, d+y1]
#     Q22 = padded_image[d+x2, d+y2]
#     x = (rows_identifier + field_x).flatten()
#     y = (cols_identifier + field_y).flatten()
#
#     # Bilinear interpolation algorithm is as described here:
#     # https://en.wikipedia.org/wiki/Bilinear_interpolation#Algorithm
#     distorted_image = (1. / ((x2 - x1) * (y2 - y1)))*(
#         Q11 * (x2 - x) * (y2 - y) +
#         Q21 * (x - x1) * (y2 - y) +
#         Q12 * (x2 - x) * (y - y1) +
#         Q22 * (x - x1) * (y - y1))
#
#     distorted_image = distorted_image.reshape((d, d))
#     return distorted_image

def zoom(image, seed=42):

    shape = image.shape
    half = int(image.shape[1]/2)

    np.random.seed(seed)
    z = np.random.uniform(0.5,1.5)

    image = ndimage.interpolation.zoom(image, z)
    center = (np.array(image.shape) / 2).astype(int)
    if z > 1:
        return image[(center[0]-half):(center[0]+half),(center[1]-half):(center[1]+half)]
    elif z < 1:
        image = np.pad(image, ((0,shape[0]-image.shape[0]),(0,shape[1]-image.shape[1])),
                       mode='constant',constant_values=0)
        return image
    else:
        return image


# def brighter(image, seed=42):
# #
# #     return image+500
# #
# # def darker(image, seed=42):
# #
# #     return image-500

def data_augmentation(image, mask, size=64):


    for i in range(0, image.shape[2], size):

        ops = {
                0: flipx,
                1: shift,
                2: flipy,
                3: rotate,
                4: zoom
            }

        which_op = np.random.randint(0, 5)


        for sample in range(min(64, image.shape[2]-i)):
            image[:, :, i+sample] = ops[which_op](image[:, :, i+sample], seed=i)
            mask[:, :, i+sample] = ops[which_op](mask[:, :, i+sample], seed=i)


    return image, mask