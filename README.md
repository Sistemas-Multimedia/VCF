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

    Done
