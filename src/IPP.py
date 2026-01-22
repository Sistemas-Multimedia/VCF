'''IPP coding: block-based motion-compensated video coding with RDO.
Stores residual frames as PNG, motion vectors and frame types as side information.
Applies an external 2D transform codec for each frame residual.'''

with open("/tmp/description.txt", "w") as f:
    f.write(__doc__)

import os
import json
import logging
import numpy as np
from PIL import Image
import av
import importlib
import parser
import main
from III import *

# ------------------------------------------------------------
# Argumentos específicos IPP
# ------------------------------------------------------------

parser.parser_encode.add_argument("--input_prefix", required=True, help="Input video")
parser.parser_encode.add_argument("--output_prefix", default="/tmp/ipp", help="Output directory")
parser.parser_encode.add_argument("-G", "--gop_size", type=int, default=10)
parser.parser_encode.add_argument("-b", "--block_size", type=int, default=16, help="IPP block size")
parser.parser_encode.add_argument("-S", "--search_range", type=int, default=4)
parser.parser_encode.add_argument("--lambda_rdo", type=float, default=0.01, help="IPP Lambda for RDO")

parser.parser_decode.add_argument("-o", "--output_prefix", default="/tmp/ipp", help="Input directory")



# ------------------------------------------------------------
# CoDec
# ------------------------------------------------------------
class CoDec:
    def __init__(self, args):
        self.args = args
        self.B = getattr(args, "block_size", None)
        try:
            transform_module = importlib.import_module(args.transform)
        except ImportError as e:
            raise ImportError(f"Error: No se encontró el módulo {args.transform} ({e})")
        self.transform_codec = transform_module.CoDec(args)
        logging.info(f"Using transform codec: {args.transform}")

    def bye(self):
        pass

    # ----------------- ENCODE -----------------
    def encode(self):
        os.makedirs(self.args.output_prefix, exist_ok=True)
        container = av.open(self.args.input_prefix)
        B = self.args.block_size
        SEARCH = self.args.search_range
        GOP = self.args.gop_size
        LAMBDA = self.args.lambda_rdo
        
        ref_frame = None
        frame_idx = 0
        
        header = {
            "block_size": B,
            "search_range": SEARCH,
            "gop_size": GOP,
            "lambda": LAMBDA,
            "transform": self.args.transform,
            "frames": []
        }

        for packet in container.demux():
            for frame in packet.decode():
                curr = np.array(frame.to_image()).astype(np.int16)
                H, W = curr.shape[:2]
                
                # Imagen para guardar (residuos o pixels)
                to_save_img = np.zeros_like(curr)
                
                mv = np.zeros((H // B, W // B, 2), dtype=np.int16)
                # Mapa de modos: 0 = Intra (raw), 1 = Inter (residual)
                modes = np.zeros((H // B, W // B), dtype=np.uint8) 

                # ---------------- I-FRAME ----------------
                if ref_frame is None or frame_idx % GOP == 0:
                    frame_type = "I"
                    # En I-Frame todo es Intra (raw pixels)
                    to_save_img = curr.copy() 
                    recon = curr.copy()
                    # Nota: En I-frame modes se queda todo en 0 (Intra)
                
                # ---------------- P-FRAME ----------------
                else:
                    frame_type = "P"
                    recon = np.zeros_like(curr)
                    
                    for y in range(0, H, B):
                        for x in range(0, W, B):
                            # Coordenadas de bloque (grid)
                            by, bx = y // B, x // B
                            
                            block = curr[y:y+B, x:x+B]
                            
                            # --- Motion Estimation ---
                            best_sad = np.inf
                            best_pred = None
                            best_dy = best_dx = 0
                            
                            for dy in range(-SEARCH, SEARCH+1):
                                for dx in range(-SEARCH, SEARCH+1):
                                    ry, rx = y+dy, x+dx
                                    if ry<0 or rx<0 or ry+B>H or rx+B>W:
                                        continue
                                    
                                    cand = ref_frame[ry:ry+B, rx:rx+B]
                                    sad = np.sum(np.abs(block - cand))
                                    
                                    if sad < best_sad:
                                        best_sad = sad
                                        best_pred = cand
                                        best_dy, best_dx = dy, dx
                            
                            residual = block - best_pred
                            
                            # --- RDO ---
                            # Coste Intra (transmitir pixel tal cual)
                            D_I = np.sum((block - block)**2) # Distortion es 0 si no hay cuantización
                            # Aproximación simple: varianza o energía como proxy de bitrate
                            mse_intra = np.mean(block**2) 
                            J_I = mse_intra # Simplificado para el ejemplo
                            
                            # Coste Inter (transmitir residuo)
                            mse_inter = np.mean(residual**2)
                            J_P = mse_inter + LAMBDA # Penalización por vector de movimiento
                            
                            # NOTA: Tu lógica original de RDO comparaba sumas directas.
                            # Aquí mantenemos la lógica de selección:
                            # Si la energía del residuo es menor que la del bloque, usa Inter.
                            
                            if np.sum(np.abs(block)) < np.sum(np.abs(residual)) + 100: # Heurística simple
                                # MODO INTRA (dentro de P-frame)
                                to_save_img[y:y+B, x:x+B] = block
                                recon[y:y+B, x:x+B] = block
                                modes[by, bx] = 0 # 0 = Intra
                            else:
                                # MODO INTER
                                # CORRECCIÓN CRÍTICA: Offset +128 para guardar negativos
                                to_save_img[y:y+B, x:x+B] = residual + 128
                                recon[y:y+B, x:x+B] = best_pred + residual
                                mv[by, bx] = (best_dy, best_dx)
                                modes[by, bx] = 1 # 1 = Inter

                # ---------------- Guardado ----------------
                # Guardamos la imagen preparada (con offset si aplica)
                residual_png = f"{self.args.output_prefix}/residual_{frame_idx:04d}.png"
                residual_prefix = f"{self.args.output_prefix}/residual_{frame_idx:04d}"
                
                Image.fromarray(np.clip(to_save_img, 0, 255).astype(np.uint8)).save(residual_png)
                
                # Aplicar códec 2D externo
                self.transform_codec.encode_fn(residual_png, residual_prefix)
                
                # Guardar Side Information (MV + MODES)
                np.save(f"{self.args.output_prefix}/frame_{frame_idx:04d}_mv.npy", mv)
                np.save(f"{self.args.output_prefix}/frame_{frame_idx:04d}_modes.npy", modes)
                
                with open(f"{self.args.output_prefix}/frame_{frame_idx:04d}.type","w") as f:
                    f.write(frame_type)
                
                header["frames"].append({
                    "index": frame_idx,
                    "type": frame_type
                })
                
                ref_frame = recon.copy()
                frame_idx += 1
                if self.args.number_of_frames and frame_idx >= self.args.number_of_frames:
                    break
            
            if self.args.number_of_frames and frame_idx >= self.args.number_of_frames:
                break
                
        # Guardar header
        with open(f"{self.args.output_prefix}/header.json","w") as f:
            json.dump(header,f,indent=2)
        logging.info("IPP encoding finished")

    # ----------------- DECODE -----------------
    def decode(self):
        header_path = f"{self.args.output_prefix}/header.json"
        with open(header_path,"r") as f:
            header = json.load(f)
            
        B = header["block_size"]
        ref_frame = None
        
        for frame_info in header["frames"]:
            idx = frame_info["index"]
            ftype = frame_info["type"]
            
            residual_prefix = f"{self.args.output_prefix}/residual_{idx:04d}"
            residual_png = f"{self.args.output_prefix}/residual_decoded_{idx:04d}.png"
            mv_fn = f"{self.args.output_prefix}/frame_{idx:04d}_mv.npy"
            modes_fn = f"{self.args.output_prefix}/frame_{idx:04d}_modes.npy"
            
            mv = np.load(mv_fn)
            modes = np.load(modes_fn) # Cargamos el mapa de modos
            
            # Decodificar la imagen (residual o pixels)
            self.transform_codec.decode_fn(residual_prefix, residual_png)
            decoded_img = np.array(Image.open(residual_png)).astype(np.int16)
            
            H, W = decoded_img.shape[:2]
            recon = np.zeros_like(decoded_img)
            
            # ---------------- Reconstrucción ----------------
            if ftype == "I":
                # En I-Frame, la imagen decodificada son pixels directos
                recon = decoded_img
            else:
                # En P-Frame, depende del modo del bloque
                for y in range(0, H, B):
                    for x in range(0, W, B):
                        by, bx = y // B, x // B
                        mode = modes[by, bx]
                        
                        block_val = decoded_img[y:y+B, x:x+B]
                        
                        if mode == 0: 
                            # MODO INTRA: La imagen guardada son pixels crudos
                            recon[y:y+B, x:x+B] = block_val
                        else:
                            # MODO INTER: La imagen guardada es (residual + 128)
                            # 1. Recuperar el residuo real (restando 128)
                            residual = block_val - 128
                            
                            # 2. Obtener predicción
                            dy, dx = mv[by, bx]
                            pred_block = ref_frame[y+dy:y+dy+B, x+dx:x+dx+B]
                            
                            # 3. Reconstruir
                            recon[y:y+B, x:x+B] = pred_block + residual

            # Clip final para asegurar rango válido de imagen
            recon = np.clip(recon, 0, 255)
            
            out_fn = f"{self.args.output_prefix}/decoded_{idx:04d}.png"
            Image.fromarray(recon.astype(np.uint8)).save(out_fn)
            logging.info(f"Saved reconstructed frame: {out_fn}")
            
            ref_frame = recon.copy()
# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
