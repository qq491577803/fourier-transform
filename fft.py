import cv2
import numpy as np
import matplotlib
import matplotlib.pylab as plt
matplotlib.use('TkAgg') 

"""
离散傅里叶变换
"""
class Ft():
    def __init__(self,image):
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]
    def dft(self,image):
        res = np.zeros(shape = [self.height,self.width],dtype=np.complex128)
        for r in range(self.height):
            for c in range(self.width):
                for rs in range(self.height):
                    for cs in range(self.width):
                        res[r][c] = res[r][c] + image[rs][cs] * np.exp(-1.j * 2 *np.pi * (r * rs / self.height + c * cs / self.width))
        
        resAbs = np.abs(res)
        center = np.fft.fftshift(resAbs)

        self.imshow(resAbs,"dft abs")
        self.imshow(center,"dft center")              
        return res
    def idft(self,image):
        resIdft = np.zeros(shape = [self.height,self.width],dtype=np.float64)
        resComples = np.zeros(shape = [self.height,self.width],dtype=np.complex64)

        for r in range(self.height):
            for c in range(self.width):
                for rs in range(self.height):
                    for cs in range(self.width):
                        resComples[r][c] = resComples[r][c] + image[rs][cs] * np.exp(1.j * 2 *np.pi * (r * rs / self.height + c * cs / self.width))
                resIdft[r][c] = np.abs(resComples[r][c] / (self.height * self.height))

        self.imshow(resIdft,"idft")
        return resIdft

    def imshow(self,image,name):
        plt.figure(name)
        plt.imshow(image,cmap="gray")
        plt.axis('on')
        plt.show()

    def run(self):
        dft = self.dft(self.image)
        Idft = self.idft(dft)

"""
快速离散傅里叶变换  递归版本
"""

class dfft():
    def __init__(self):
        pass

    def imshow(self,image,name):
        plt.figure(name)
        plt.imshow(image,cmap = "gray")
        plt.show()

    def _fft1d(self,a, invert=False):
        N = len(a)
        if N == 1:
            return [a[0]]
        elif N & (N - 1) == 0:  # O(nlogn),  2^k
            even = self._fft1d(a[::2], invert)
            odd = self._fft1d(a[1::2], invert)
            i = 2j if invert else -2j
            factor = np.exp(i * np.pi * np.arange(N // 2) / N)
            prod = factor * odd
            return np.concatenate([even + prod, even - prod])
        else:  # O(n^2)
            w = np.arange(N)
            i = 2j if invert else -2j
            m = w.reshape((N, 1)) * w
            W = np.exp(m * i * np.pi / N)
            return np.concatenate(np.dot(W, a.reshape((N, 1))))

    def fft2d(self,image,invert = False):
        result_complex = image.astype(np.complex64)
        for row in range(image.shape[0]):
            result_complex[row,:] = self._fft1d(result_complex[row,:],invert)           
        for col in range(image.shape[1]):
            result_complex[:,col] = self._fft1d(result_complex[:,col],invert)
        return result_complex
    def fft2dshift(self,array):
        #交换1.3 象限和 2.4象限
        rows = image.shape[0]
        cols = image.shape[1]

        rows_m = rows // 2
        cols_m = cols // 2

        shift = np.zeros_like(array,dtype=np.complex64)
        shift[:rows_m,:cols_m] = array[rows_m:,cols_m:]
        shift[rows_m:,cols_m:] = array[:rows_m,:cols_m]

        shift[:rows_m,cols_m:] = array[rows_m:,:cols_m]
        shift[rows_m:,:cols_m] = array[:rows_m,cols_m:]

        return shift
if __name__ == "__main__":
    image = cv2.imread("../image/face.jpeg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,dsize=(256,256))    
    fft = dfft()
    #fft 正变换
    res = fft.fft2d(image,invert=False)
    #fft 中心化
    resShift = fft.fft2dshift(res)
    #fft 反变换

    res = fft.fft2d(resShift,invert=True)
    res = np.abs(res)
    plt.subplot(131)
    plt.imshow(image,cmap = "gray",label = "12")
    plt.title("srcImage")
    plt.subplot(132)
    plt.imshow(np.abs(resShift),cmap = "gray")
    plt.title("fftImage")
    plt.subplot(133)
    plt.imshow(res,cmap = "gray")
    plt.title("ifftImage")
    plt.show()





