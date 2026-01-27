import argparse

from encoder_ipp import main as encode_main
from decoder_ipp import main as decode_main
from test_ipp import main as test_main

def main():
    parser = argparse.ArgumentParser(description="IPP temporal codec (GOP I+P)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_enc = subparsers.add_parser("encode", help="Codificar secuencia (frames -> compressed)")
    p_dec = subparsers.add_parser("decode", help="Decodificar secuencia (compressed -> results)")
    p_tst = subparsers.add_parser("test", help="Calcular PSNR y CR")

    args = parser.parse_args()

    if args.command == "encode":
        encode_main()
    elif args.command == "decode":
        decode_main()
    elif args.command == "test":
        test_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
