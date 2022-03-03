import cv2
import numpy as np
import pickle
from os.path import exists


class Cartext:
    def __init__(self, image):
        self.image = image

    @staticmethod
    def generateTexture(name, image, sigma=7):

        height, width = image.shape
        size = height * width
        # sigma = 7  # scale parameter

        ksize = int(4 * np.ceil(sigma) + 1)

        kernel = cv2.getGaussianKernel(ksize, sigma, cv2.CV_64F)
        kernel = [i[0] for i in kernel]

        tpGrad = np.zeros((size), dtype=np.float64)

        tpGradAA = np.zeros((image.shape), dtype=np.float64)

        iHeight, iWidth = image.shape

        tpI = np.array([j for sub in image for j in sub], dtype=np.float64)

        print("width", iWidth)
        print("height", iHeight)

        for ih in range(1, iHeight - 1):
            for iw in range(1, iWidth - 1):
                xgrad = tpI[ih * iWidth + iw + 1] - tpI[ih * iWidth + iw - 1]
                ygrad = tpI[(ih-1) * iWidth + iw] - tpI[(ih+1) * iWidth + iw]

                tpGrad[ih * iWidth +
                       iw] = np.hypot(xgrad, ygrad)

        # -------------------------------------------

        ratio1 = Cartext.fiSepConvol(tpGrad, iWidth, iHeight,
                                     kernel, ksize, kernel, ksize)

        niter = 5
        gconvolved = Cartext.low_pass_filter(
            tpI, sigma, niter, width, height)

        # -------------------------------------------

        tpGrad2 = np.zeros((size), dtype=np.float64)

        for ih in range(1, iHeight - 1):
            for iw in range(1, iWidth - 1):
                xgrad = gconvolved[ih * iWidth + iw + 1] - \
                    gconvolved[ih * iWidth + iw - 1]
                ygrad = gconvolved[(ih-1) * iWidth + iw] - \
                    gconvolved[(ih+1) * iWidth + iw]

                tpGrad2[ih * iWidth +
                        iw] = np.hypot(xgrad, ygrad)

        ratio2 = Cartext.fiSepConvol(
            tpGrad2, width, height, kernel, ksize, kernel, ksize)

        out = np.zeros((size), dtype=np.float64)

        for i in range(size):
            weight = Cartext.WeightingFunction(ratio1[i], ratio2[i])
            out[i] = weight * gconvolved[i] + (1.0 - weight) * tpI[i]

        cartoon_v = out
        vLim = 20
        dif_v = np.zeros((size), dtype=np.float64)

        for ii in range(size):
            fValue = tpI[ii] - cartoon_v[ii]
            fValue = (fValue + vLim) * 255.0 / (2.0 * vLim)

            if fValue < 0.0:
                fValue = 0.0
            if fValue > 255.0:
                fValue = 255.0

            dif_v[ii] = fValue

        cartoon_v -= cartoon_v.min()
        cartoon_v /= cartoon_v.max()

        dif_v -= dif_v.min()
        dif_v /= dif_v.max()

        cartoon_v = np.asarray(np.around(cartoon_v * 255), dtype=np.uint8).reshape(
            iHeight, iWidth)

        dif_v = np.asarray(np.around(dif_v * 255),
                           dtype=np.uint8).reshape(iHeight, iWidth)

        with open(name + '_cartoon.pickle', 'wb') as handle:
            pickle.dump(cartoon_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(name + '_texture.pickle', 'wb') as handle:
            pickle.dump(dif_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return cartoon_v, dif_v

    @staticmethod
    def WeightingFunction(r1, r2):
        difference = r1 - r2

        ar1 = np.abs(r1, dtype=np.float64)
        if ar1 > 1.0:
            difference /= ar1
        else:
            difference = 0.0

        weight = None
        cmin = 0.25
        cmax = 0.50

        if difference < cmin:
            weight = 0.0
        elif difference > cmax:
            weight = 1.0
        else:
            weight = (difference - cmin) / (cmax - cmin)

        return weight

    @staticmethod
    def low_pass_filter(input, sigma, niter, width, height):
        size = width*height
        ksize = int(4 * np.ceil(sigma) + 1)

        kernel = cv2.getGaussianKernel(ksize, sigma, cv2.CV_64F)
        kernel = [i[0] for i in kernel]

        imconvolved = np.zeros((size), dtype=np.float64)
        imdifference = np.zeros((size), dtype=np.float64)

        out = Cartext.fiSepConvol(
            input, width, height, kernel, ksize, kernel, ksize)

        imdifference = Cartext.fpCombine(
            input, 1.0, out, -1.0, width*height)

        for i in range(niter):
            imconvolved = Cartext.fiSepConvol(imdifference, width, height,
                                              kernel, ksize, kernel, ksize)

            imdifference = Cartext.fpCombine(
                imdifference, 1.0, imconvolved, -1.0, width*height)

        out = Cartext.fpCombine(input, 1.0, imdifference, -1.0, size)

        return out

    @staticmethod
    def fiFloatHorizontalConvolution(u, width, height, kernel, ksize):
        v = np.zeros((width*height), dtype=np.float64)

        halfsize = int(ksize / 2)
        buffersize = width + ksize

        buffer = np.zeros((buffersize), dtype=np.float64)

        for r in range(height):
            l = r*width
            for i in range(halfsize):
                buffer[i] = u[l + halfsize - 1 - i]

            for i in range(width):
                buffer[halfsize + i] = u[l + i]

            for i in range(halfsize):
                buffer[i + width + halfsize] = u[l + width - 1 - i]

            buffer = Cartext.fiFloatBufferConvolution(
                buffer, kernel, width, ksize)

            for c in range(width):
                v[r*width+c] = buffer[c]

        return v

    @staticmethod
    def fiFloatVerticalVonvolution(u, width, height, kernel, ksize):
        v = np.zeros((width*height), dtype=np.float64)

        halfsize = int(ksize / 2)
        buffersize = height + ksize

        buffer = np.zeros((buffersize), dtype=np.float64)

        for c in range(width):

            for i in range(halfsize):
                buffer[i] = u[(halfsize-i-1)*width + c]

            for i in range(height):
                buffer[halfsize + i] = u[i*width + c]

            for i in range(halfsize):
                buffer[halfsize + height + i] = u[(height - i - 1)*width+c]

            buffer = Cartext.fiFloatBufferConvolution(
                buffer, kernel, height, ksize)

            for r in range(height):
                v[r*width+c] = buffer[r]

        return v

    @staticmethod
    def fiFloatBufferConvolution(buffer, kernel, size, ksize):
        for i in range(size):
            sum = 0.0
            k = 0

            for j in range(0, ksize - 4, 5):
                sum += buffer[i + 0 + j] * kernel[0 + j] + buffer[i + 1 + j] * kernel[1 + j] + buffer[i + 2 + j] * \
                    kernel[2 + j] + buffer[i + 3 + j] * \
                    kernel[3 + j] + buffer[i + 4 + j] * kernel[4 + j]
                k += j

            # for j in range(k, ksize, 5):
            #     sum += buffer[i + j] * (kernel[j])

            buffer[i] = sum

        return buffer

    @staticmethod
    def fiSepConvol(v, width, height, xkernel, xksize, ykernel, yksize):
        v = Cartext.fiFloatHorizontalConvolution(
            v, width, height, xkernel, xksize)

        v = Cartext.fiFloatVerticalVonvolution(
            v, width, height, ykernel, yksize)

        return v

    @staticmethod
    def fpCombine(u, a, v, b, size):
        w = np.zeros((size), dtype=np.float64)
        for i in range(size):
            w[i] = (a * u[i] + b * v[i])

        return w

    @staticmethod
    def loadTexture(name, image, sigma=7):

        path_texture = name + '_texture.pickle'
        path_cartoon = name + '_cartoon.pickle'

        if exists(path_texture) and exists(path_cartoon):
            with open(path_cartoon, 'rb') as handle:
                cartoon = pickle.load(handle)
            with open(path_texture, 'rb') as handle:
                texture = pickle.load(handle)
        else:
            print('Sorry but couldn\'t find cartex pickle files. Regenerating...')
            cartoon, texture = Cartext.generateTexture(
                name=name, image=image, sigma=sigma)

        return cartoon, texture


if __name__ == '__main__':
    image_latent = 'B111.png'
    image_exemplar = '002-06.jp2'

    name = image_latent

    # -----------------------

    path = 'img/' + name
    print(path)

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cartoon_v, dif_v = Cartext.generateTexture(image, sigma=7)

    with open(name + '_cartoon.pickle', 'wb') as handle:
        pickle.dump(cartoon_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(name + '_texture.pickle', 'wb') as handle:
        pickle.dump(dif_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cv2.imshow("cartoon_v", cartoon_v)
    cv2.imshow("texture", dif_v)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
