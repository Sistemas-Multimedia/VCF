from collections import OrderedDict


def bits_to_bytes(bits):
    while len(bits) % 8 != 0:
        bits.append(0)
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | b
        result.append(byte)
    return bytes(result)


def bytes_to_bits(data):
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


# Real number out of range 8
def EOF():
    return 256


class FrequencyTable:
    def __init__(self):
        self.alphabet = []
        self.frequencies = OrderedDict()
        self.nsymbols = 0

    def add(self, symbol):
        self.nsymbols += 1
        if symbol in self.frequencies:
            self.frequencies[symbol] += 1
        else:
            self.alphabet.append(symbol)
            self.frequencies[symbol] = 1

    def calculateProbabilities(self):
        probabilities = {}
        for symbol in self.alphabet:
            probabilities[symbol] = self.frequencies[symbol] / self.nsymbols

        ranges = {}
        accumulate = 0.0
        for symbol in self.alphabet:
            ranges[symbol] = (accumulate, accumulate + probabilities[symbol])
            accumulate += probabilities[symbol]

        return probabilities, ranges

    def show_table(self):
        print(f"\n{'=' * 80}")
        print(f"{'Symbol':<10} | {'Frequency':<10} | {'Probability':<12} | {'Low':<12} | {'High':<12}")
        print(f"{'=' * 80}")

        probabilities, accumulate_range = self.calculateProbabilities()

        for symbol in sorted(self.alphabet):
            f = self.frequencies[symbol]
            p = probabilities[symbol]
            low, high = accumulate_range[symbol]
            print(f"{str(symbol):<10} | {f:<10} | {p:<12.6f} | {low:<12.6f} | {high:<12.6f}")

        print(f"{'=' * 80}\n")


class ArithmeticCodding:
    def __init__(self, precision=32):
        self.precision = precision
        self.max_val = (1 << precision) - 1
        self.half = 1 << (precision - 1)
        self.quarter = self.half >> 1

    def encode(self, message, show_table=True):
        frequencytable = FrequencyTable()
        for c in message:
            frequencytable.add(c)

        if show_table:
            frequencytable.show_table()

        total = frequencytable.nsymbols
        symbols = list(frequencytable.frequencies.keys())
        freqs = list(frequencytable.frequencies.values())

        cumulative = {}
        acc = 0
        for s, f in zip(symbols, freqs):
            cumulative[s] = (acc, acc + f)
            acc += f

        low, high = 0, self.max_val
        bits = []
        pending = 0

        for symbol in message:
            r_low, r_high = cumulative[symbol]
            range_width = high - low + 1
            high = low + (range_width * r_high // total) - 1
            low = low + (range_width * r_low // total)

            while True:
                if high < self.half:
                    bits.append(0)
                    bits.extend([1] * pending)
                    pending = 0
                elif low >= self.half:
                    bits.append(1)
                    bits.extend([0] * pending)
                    pending = 0
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < 3 * self.quarter:
                    pending += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break

                low <<= 1
                high = (high << 1) | 1

        pending += 1
        if low < self.quarter:
            bits.append(0)
            bits.extend([1] * pending)
        else:
            bits.append(1)
            bits.extend([0] * pending)

        return bits, frequencytable

    def write(self, filename, bits, table):
        with open(filename, "wb") as f:
            # símbolos distintos
            f.write(len(table.frequencies).to_bytes(2, "big"))

            # tabla de frecuencias
            for s, freq in table.frequencies.items():
                f.write(s.to_bytes(2, "big"))
                f.write(freq.to_bytes(4, "big"))

            # bits comprimidos
            data = bits_to_bytes(bits)
            f.write(data)

    def read(self, filename):
        with open(filename, "rb") as f:
            n_symbols = int.from_bytes(f.read(2), "big")

            table = FrequencyTable()
            for _ in range(n_symbols):
                symbol = int.from_bytes(f.read(2), "big")
                freq = int.from_bytes(f.read(4), "big")
                table.frequencies[symbol] = freq
                table.alphabet.append(symbol)
                table.nsymbols += freq

            bits = bytes_to_bits(f.read())
        return bits, table

    def decode(self, bits, table):
        total = table.nsymbols
        cumulative = {}
        acc = 0
        for s in table.alphabet:
            cumulative[s] = (acc, acc + table.frequencies[s])
            acc += table.frequencies[s]

        low, high = 0, self.max_val
        code = 0
        bit_iter = iter(bits)

        for _ in range(self.precision):
            code = (code << 1) | next(bit_iter, 0)

        result = []

        while True:
            range_width = high - low + 1
            value = ((code - low + 1) * total - 1) // range_width

            for s, (r_low, r_high) in cumulative.items():
                if r_low <= value < r_high:
                    if s == EOF():
                        return result
                    result.append(s)
                    high = low + (range_width * r_high // total) - 1
                    low = low + (range_width * r_low // total)
                    break

            while True:
                if high < self.half:
                    pass
                elif low >= self.half:
                    low -= self.half
                    high -= self.half
                    code -= self.half
                elif low >= self.quarter and high < 3 * self.quarter:
                    low -= self.quarter
                    high -= self.quarter
                    code -= self.quarter
                else:
                    break

                low <<= 1
                high = (high << 1) | 1
                code = (code << 1) | next(bit_iter, 0)

        return result


if __name__ == "__main__":
    arith = ArithmeticCodding()
    message = "Esto es un texto de prueba"
    data = list(message.encode("utf-8"))
    data.append(EOF())

    print("Codificando: ...")
    print(message)

    bits, table = arith.encode(data)
    arith.write("compressed.bin", bits, table)

    bits2, table2 = arith.read("compressed.bin")
    print("Decodificado: ...")
    decoded = arith.decode(bits2, table2)

    decoded_text = bytes(decoded).decode("utf-8")

    print(decoded_text)
