import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import zlib

# Load image
img = Image.open(r"C:/Users/pma98/Desktop/SM/images/image.png").convert('L')

# Convert to numpy array and reshape to 2D array of size (height, width)
imgArray = np.array(img).reshape((img.height, img.width))

# Perform DCT on image array
dctArray = dct(dct(imgArray, axis=0), axis=1)

# Divide DCT array into 8x8 blocks
dctBlocks = dctArray.reshape(-1, 8, 8)

# Quantize DCT blocks
quantizationTable = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])

#print(dctBlocks.shape)
#print(dctBlocks.dtype)
#print(quantizationTable.shape)
#print(quantizationTable.dtype)

quantizedBlocks = np.floor(dctBlocks / quantizationTable).astype(quantizationTable.dtype)

# Encode quantized blocks using zlib
encodedBlocks = zlib.compress(quantizedBlocks)

# Save the compressed data and quantization table to a file
with open("test.bin", "wb") as f:
    f.write(encodedBlocks)
    f.write(quantizationTable)

# Read compressed data and quantization table from file
with open("test.bin", "rb") as f:
    encodedBlocks = f.read()
    quantizationMatrix = f.read()

# Decode quantized blocks using zlib
decodedBlocks = zlib.decompress(encodedBlocks)

# Dequantize DCT blocks
dequantizedBlocks = dctBlocks * quantizationTable[np.newaxis, :, :]

# Perform IDCT
idctArray = idct(idct(dequantizedBlocks, axis=0), axis=1)
originalShape = img.size

# Create image from IDCT array and display it
idctArray = idctArray.reshape(originalShape)
# Show reshaped array
#print(idctArray)
imgReconstructed = Image.fromarray(idctArray.astype(np.uint8))
imgReconstructed.show()