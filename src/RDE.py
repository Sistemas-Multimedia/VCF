'''Compute the Rate/Distortion Efficiency (J=R+D).'''

import os
import io
from skimage import io as skimage_io # pip install scikit-image
import argparse
import numpy as np
from PIL import Image
import urllib

def calculate_rmse(image1_path, image2_path):
    """
    Reads two images and computes the Root Mean Square Error (RMSE) 
    between them, assuming they have the same dimensions.
    """
    
    # 1. Load the images

    try:
        img1 = Image.open(image1_path)
    except FileNotFoundError as e:
        req = urllib.request.Request(image1_path, method='HEAD')
        f = urllib.request.urlopen(req)
        input_size = int(f.headers['Content-Length'])
        img1 = skimage_io.imread(image1_path) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread

    img2 = Image.open(image2_path)

    # 2. Convert images to NumPy arrays
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    # 3. Check for matching dimensions
    if arr1.shape != arr2.shape:
        print(f"Error: Image dimensions do not match.")
        print(f"Image 1 shape: {arr1.shape}")
        print(f"Image 2 shape: {arr2.shape}")
        return None

    # 4. Compute the difference
    difference = arr1 - arr2

    # 5. Compute the Squared Error (SE)
    # The **2 operator performs element-wise squaring
    squared_error = difference**2

    # 6. Compute the Mean Squared Error (MSE)
    # The .mean() function calculates the average of all elements in the array
    mse = squared_error.mean()

    # 7. Compute the Root Mean Square Error (RMSE)
    # np.sqrt() calculates the square root
    rmse = np.sqrt(mse)

    return rmse, arr1.shape

def get_file_size(file_path):
    file_size = 0
    try:
        file_size = os.path.getsize(file_path)
    except FileNotFoundError as e:
        req = urllib.request.Request(file_path, method='HEAD')
        f = urllib.request.urlopen(req)
        file_size = int(f.headers['Content-Length'])
    finally:
        return file_size

def main():
    # 1. Setup Argument Parser 
    parser = argparse.ArgumentParser(
        description="Compute the rate/distortion efficiency (J=R+D)."
    )

    parser.add_argument(
        "-o", "--original",
        type=str, help="Original image (default: %(default)s)", default="/tmp/original.png")

    parser.add_argument(
        "-c", "--codestream",
        type=str, help="Codestream (default: %(default)s)", default="/tmp/encoded.tif")

    parser.add_argument(
        "-d", "--decoded",
        type=str, help="Decoded image (default: %(default)s)", default="/tmp/decoded.png")

    args = parser.parse_args()

    RMSE, shape = calculate_rmse(args.original, args.decoded)
    original_bytes = get_file_size(args.original)
    codestream_bytes = get_file_size(args.codestream)
    decoded_bytes = get_file_size(args.decoded)
    number_of_pixels = shape[0]*shape[1]
    original_bpp = original_bytes*8/number_of_pixels
    codestream_bpp = codestream_bytes*8/number_of_pixels
    decoded_bpp = decoded_bytes*8/number_of_pixels
    print("Original image:", args.original, original_bytes, f"bytes ({original_bpp:.2f}) bits/pixel")
    print("Code-stream:", args.codestream, codestream_bytes, f"bytes ({codestream_bpp:.2f}) bits/pixel")
    print("Decoded image:", args.decoded, decoded_bytes, f"bytes ({decoded_bpp:.2f}) bits/pixel")
    
    print("Images shape:", shape)
    print(f"Distortion (RMSE): {RMSE:.2f}")
    #number_of_output_bytes = get_file_size(args.codestream)
    #number_of_output_bits = number_of_output_bytes*8
    #number_of_pixels = shape[0]*shape[1]
    #BPP = number_of_output_bits/number_of_pixels
    #print(f"Rate (bits/pixel): {BPP:.2f}")
    J = codestream_bpp + RMSE
    print(f"J = R + D = {J:.2f}")

if __name__ == "__main__":
    main()
