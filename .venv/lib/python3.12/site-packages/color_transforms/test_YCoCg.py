import numpy as np
import YCoCg
from skimage import io # pip install scikit-image
from image_IO import image_3 as RGB_image # pip install "image_IO @ git+https://github.com/vicente-gonzalez-ruiz/image_IO"

image_URL = "http://www.hpca.ual.es/~vruiz/images/lena.png"

img = io.imread(image_URL).astype(np.float32)

YCoCg_img = YCoCg.from_RGB(img)
restored_img = YCoCg.to_RGB(YCoCg_img).astype(np.int16)

if not np.all(img == restored_img):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if not np.all(img[y, x] == restored_img[y, x]):
                print(img[y, x], restored_img[y, x])
                break
