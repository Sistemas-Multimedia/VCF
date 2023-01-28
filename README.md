# Visual Coding Framework
A programming environment to develop and test image and video compression algorithms.



Please, select one of the following tasks to develop:
# Tema 1: 
Modify VCF to allow the use of Huffman coding as a entropy codec in the compression pipeline. A context-based probabilistic model should be used to minimize the probabilities of the symbols. Complexity 3.
    
    Done with OpenCV and IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY
    To execute it use parameter: "-hm" or "--huffman"

Modify VCF to allow the use of arithmetic coding as a entropy codec in the compression pipeline. Again, a context-based probabilistic model should be used and in this case, to speed up the decoding of the symbols and to increase the compression ratio, it is convenient that they follow a Laplace statistical distribution that can be easily obtained using a spatial predictor. Complexity 3.

    Done with OpenCV and IMWRITE_PNG_STRATEGY_FILTERED
    To execute ir use parameter: "--a" or "--aritmethic"

    
Modify VCF to allow the use of zlib as a entropy codec in the compression pipeline. Again, use spatial predictor to increase the compression ratio. Complexity 2.

    Done with OpenCV and IMWRITE_PNG_COMPRESSION
    

# Tema2:
Modify VCF to allow the use of Lloyd-Max quantization in the compression pipeline. Notice that VCF already implements this quantizer, but the compression pipeline is unfinished. You must implement a complete pipeline (at least for compressing images). Complexity 2.
        
    Is it already done?

        

Modify VCF to allow the use of VQ (applied to the spatial domain) in the compression pipeline. Notice that VQ used in the spatial domain can significantly minimize the advantage of using a spatial transform. For this reason, it can be useful to implement also “fake” spatial transform where no transformation is performed at all. The same happens when VQ is used in the color domain. Complexity 4.

    .
# Tema3:
Modify VCF to allow the use of the (color) DCT in the compression pipeline. Complexity 1.

    .

Modify VCF to allow the use of the YCrCb transform in the compression pipeline. Complexity 1.

    .

Modify VCF to allow the use of VQ (Vector Quantization) (applied to the color domain) in the compression pipeline. Complexity 4

    .

# Tema4:
Tiles are used when the image is made up of very different areas (for example, text and natural images). Tiles are usually rectangular but can have any size, and are usually defined attending to perceptual issues (for example, text is not well compressed by lossy configurations).

    .

Blocks are smaller than tiles and, in most of cases, squared. The block partition can be adaptive and, in this case, should be found using RDO.

    .

# Tema5:
Modify VCF to encode/decode a sequence of images using a III... scheme. Complexity 1.

    .
Modify VCF to encode/decode a sequence of images using a IPP... scheme, without motion compensation. Complexity 2.

    .
Modify VCF to encode/decode a sequence of images using a IPP... scheme, with motion compensation. Complexity 4.

    .
Modify VCF to encode/decode a sequence of images using a IBB... scheme, with motion compensation. Complexity 5.

    .

# Tema6:
Modify the compression pipeline of VCF to exploit the chroma redundancy. Use different quantization step sizes for each color subband. Complexity 2.

    .
Use the default quantization matrices of JPEG in the DCT pipeline of VCF. Complexity 4.

    .
Do the same, but using the DWT. Complexity 4.

    .
The local entropy of the motion vectors can be a good estimation of the motion complexity in a video sequence. In a motion compensated video coding pipeline in VCF, adapt the quantization step size to the local entropy, trying to increase the compression ratios without increasing the perceived distortoin. Compexity 5.

    .
# Tema7:
The L-levels DWT provides L+1 spatial resolution levels of an image. Modify the VCF pipeline to include this functionality. Complexity 2. 

    .
The 2^N x 2^N -DCT domain can be decoded by resolution levels using a inverse 2^Mx2^M-DCT where m =0,1,..., to the lower frequency subbands 0x0 (notice that the inverse 0x0-DCT does not perform any computation). Implement in VCF such image decoder. 

    .
See the notebook Image Compression with YCoCg + 2D-DCT. Complexity 4. 

    .
In video coding, we can obtain spatial scalability if we build a laplacian pyramid of the frames and compress each level of the sequence using a normal video encoder. Notice that we can use the reconstructed sequence at the spatial level l to improve the predictions for the level l-1. Incorporate this functionality to VCF. Complexity 6. 

    .
The spatial resolution level l of the reconstructed video can be used (after interpolation) to estimate the motion at the l-1 level^4, making the transmission of motion fields unnecessary for resolution level l-1. Explore this in VCF. Complexity 7.
