from collections import OrderedDict


def EOF():
    return 256


# Build a default table with 256 bits for image
def build_default_FrequencyTable():
    freqs = FrequencyTable()

    for symbol in range(256):
        freqs.add(symbol)

    eof = EOF()
    freqs.add(eof)

    freqs.calculateProbabilities()
    return freqs


class FrequencyTable:
    def __init__(self):
        self.alphabet = []
        self.frequencies = OrderedDict()
        self.cumulative_ranges = {}
        self.nsymbols = 0

    def add(self, symbol):
        if symbol in self.frequencies:
            self.frequencies[symbol] += 1
        else:
            self.alphabet.append(symbol)
            self.frequencies[symbol] = 1
        self.nsymbols += 1

    def updateFreqs(self, symbol):
        self.frequencies[symbol] += 1
        self.nsymbols += 1
        self.calculateProbabilities()

    def calculateProbabilities(self):
        ranges = {}
        accumulate = 0

        for symbol in self.alphabet:
            low = accumulate
            high = accumulate + self.frequencies[symbol]
            ranges[symbol] = (low, high)
            accumulate = high

        self.cumulative_ranges = ranges


class ArithmeticCodding:
    def __init__(self, precision=32):
        self.precision = precision
        self.max_val = (1 << precision) - 1
        self.half = 1 << (precision - 1)
        self.quarter = self.half >> 1

        self.high = self.max_val
        self.low = 0
        self.pending = 0

    def finish(self, bits):
        self.pending += 1
        if self.low < self.quarter:
            bits.append(0)
            bits.extend([1] * self.pending)
        else:
            bits.append(1)
            bits.extend([0] * self.pending)

    def encode_symbol(self, symbol, table, bits):
        total = table.nsymbols

        r_low, r_high = table.cumulative_ranges[symbol]
        range_width = self.high - self.low + 1

        self.high = self.low + (range_width * r_high // total) - 1
        self.low = self.low + (range_width * r_low // total)

        while True:
            if self.high < self.half:
                bits.append(0)
                bits.extend([1] * self.pending)
                self.pending = 0
            elif self.low >= self.half:
                bits.append(1)
                bits.extend([0] * self.pending)
                self.pending = 0
                self.low -= self.half
                self.high -= self.half
            elif self.low >= self.quarter and self.high < 3 * self.quarter:
                self.pending += 1
                self.low -= self.quarter
                self.high -= self.quarter
            else:
                break

            self.low <<= 1
            self.high = (self.high << 1) | 1

    def decode(self, bits, table, result):
        code = 0
        total = table.nsymbols
        low, high = self.low, self.high
        bit_iter = iter(bits)

        for _ in range(self.precision):
            code = (code << 1) | next(bit_iter, 0)

        while True:
            range_width = high - low + 1
            value = ((code - low + 1) * total - 1) // range_width

            for s, (r_low, r_high) in table.cumulative_ranges.items():
                if r_low <= value < r_high:
                    if s == EOF():
                        return
                    result.append(s)
                    high = low + (range_width * r_high // total) - 1
                    low = low + (range_width * r_low // total)

                    table.updateFreqs(s)
                    total = table.nsymbols
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


# Helpers
def _encode(message):
    table_enc = build_default_FrequencyTable()
    coder = ArithmeticCodding(precision=16)

    bits = []

    for symbol in message:
        coder.encode_symbol(symbol, table_enc, bits)
        table_enc.updateFreqs(symbol)

    coder.finish(bits)

    print("Bits codificados:")
    print(bits)
    return bits


def _decode(bits):
    table_dec = build_default_FrequencyTable()
    decoder = ArithmeticCodding(precision=16)
    result = []
    decoder.decode(bits, table_dec, result)

    print("Símbolos decodificados:")

    decoded_text = bytes(result).decode("utf-8")
    print("result:")
    print(decoded_text)


if __name__ == "__main__":
    text = "This is an example of Arith Adaptive"
    message = list(text.encode("utf-8"))
    message.append(EOF())

    bits = _encode(message)
    _decode(bits)
