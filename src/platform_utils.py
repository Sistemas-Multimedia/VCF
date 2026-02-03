"""
Cross-platform compatibility module for VCF.

This module provides functions and constants to handle file paths
compatible with Windows, Linux, and macOS.
"""

import os
import sys
import tempfile


def get_temp_dir():
    """Get the system temporary directory in a cross-platform way."""
    return tempfile.gettempdir()


def get_vcf_temp_dir():
    """Get the VCF temporary directory, creating it if it doesn't exist."""
    if sys.platform == 'win32':
        temp_dir = "C:/tmp"
    else:
        temp_dir = "/tmp"

    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def get_temp_path(filename):
    """Build a full path to a file in the VCF temporary directory."""
    return os.path.join(get_vcf_temp_dir(), filename)


def ensure_description_file(description_text):
    """Create the description.txt file required by parser.py."""
    desc_path = get_temp_path("description.txt")
    with open(desc_path, 'w') as f:
        f.write(description_text)


def get_file_uri(path):
    """Convert a file path to file:// URI format."""
    path = os.path.abspath(path)
    if sys.platform == 'win32':
        path = path.replace('\\', '/')
        return f"file:///{path}"
    else:
        return f"file://{path}"


# Predefined path constants
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/videos/mobile_352x288x30x420x300.mp4"
ENCODE_OUTPUT_PREFIX = get_temp_path("encoded")
DECODE_INPUT_PREFIX = ENCODE_OUTPUT_PREFIX
DECODE_OUTPUT_PREFIX = get_temp_path("decoded")
DECODE_OUTPUT = get_temp_path("decoded.mp4")

ORIGINAL = get_temp_path("original.png")
ENCODED = get_temp_path("encoded")
DECODED = get_temp_path("decoded.png")

FRAME_PREFIX = get_temp_path("img_")
ORIGINAL_FRAME_PREFIX = get_temp_path("original_")
DECODED_FRAME_PREFIX = get_temp_path("decoded_")


def get_original_frame_path(index, digits=4):
    """Generate path for an original frame."""
    return f"{ORIGINAL_FRAME_PREFIX}{index:0{digits}d}.png"


def get_decoded_frame_path(index, digits=4):
    """Generate path for a decoded frame."""
    return f"{DECODED_FRAME_PREFIX}{index:0{digits}d}.png"


# Ensure temp directory exists on import
_temp_dir = get_vcf_temp_dir()
