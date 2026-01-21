import heapq
from collections import defaultdict, Counter
from bitarray import bitarray

class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = Counter(data)
    heap = [HuffmanNode(value, freq) for value, freq in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    
    return heap[0]

def generate_huffman_codes(node, current_code="", codes={}):
    if node is None:
        return
    if node.value is not None:
        codes[node.value] = current_code
    generate_huffman_codes(node.left, current_code + "0", codes)
    generate_huffman_codes(node.right, current_code + "1", codes)
    return codes

def encode_data(data, codes):
    # Create a single concatenated string of all encoded bits
    encoded_string = ''.join(codes[value] for value in data)
    #print("-------------_", len(data))
    # Convert this string of bits to a bitarray
    encoded_data = bitarray(encoded_string)

    return encoded_data

def decode_data(encoded_data, root):
    data = []
    node = root
    for bit in encoded_data:
        if bit == 0:
            node = node.left
        else:
            node = node.right
        # If it's a leaf node, record the value and reset to root
        if node.left is None and node.right is None:
            data.append(node.value)
            node = root
    #print("-------------_", len(data))
    return data

def print_huffman_tree(root, indent="", is_left=True):
    """Prints the Huffman tree in a visually readable format."""
    if root is not None:
        print(indent, end="")
        if indent:
            print("├─ " if is_left else "└─ ", end="")

        if root.value is not None:
            print(f"'{root.value}' ({root.freq})")
        else:
            print(f"Node ({root.freq})")

        indent += "│  " if is_left else "   "

        print_huffman_tree(root.left, indent, True)
        print_huffman_tree(root.right, indent, False)
