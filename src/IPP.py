import LloydMax as deadzone
import os
import main
import logging
import PNG as EC
import cv2 
import numpy as np
from motion_estimation import full_search, prediction
import logging

EC.parser.add_argument("-f", "--input_folder", type=str, help="Carpeta de entrada de imágenes", default="images")


class CoDec(deadzone.CoDec):

    def __init__(self, args):
        super().__init__(args)

    def encode(self):
        os.chdir(self.args.input_folder)
        input_folder = os.getcwd()
        encode_folder = f"{input_folder}/encoded"
        os.makedirs(encode_folder, exist_ok=True)
        current_img = None
        index = 0
        for input_file in os.listdir(input_folder):
            if input_file.endswith(('.png', '.jpg')):
                if index == 0:
                    self.args.output = f'encoded/{index}.png'
                    self.args.input = f"{input_folder}/{input_file}"
                    super().encode()
                    current_img = cv2.imread(input_file)
                else:
                    #Lee un archivo de entrada y crea una copia de la imagen para que los cambios
                    #realizados sobre la imagen no afecten el archivo original.
                    next_img = cv2.imread(input_file)
                    copy_image = next_img.copy()
                    subtract = np.clip(np.subtract(next_img.astype(np.int16), current_img.astype(np.int16)) + 128, 0, 255).astype(np.uint8)                    
                    cv2.imwrite(f'encoded/{index}.png', subtract)
                    self.args.input = f'encoded/{index}.png'
                    self.args.output = self.args.input
                    super().encode()
                    current_img = copy_image
                index += 1
        logging.info(f'{index} images encoded')
        return 

    def decode(self):
        os.chdir(self.args.input_folder)
        encoded_folder = os.path.join(os.getcwd(), 'encoded')
        decode_folder = os.path.join(os.getcwd(), 'decoded')
        os.makedirs(decode_folder, exist_ok=True)
        current_img = None
        index = 0
        for input_file in os.listdir(encoded_folder):
            if input_file.endswith(('.png', '.jpg')):
                if index == 0:
                    output_path = 'decoded/' + input_file
                    self.args.output = output_path
                    self.args.input = f"{encoded_folder}/{input_file}"
                    super().decode()
                    current_img = cv2.imread(f'decoded/{index}.png')
                else:
                    path = f'decoded\\{index}.png'
                    self.args.output = path
                    self.args.input = f'{encoded_folder}/{input_file}'
                    super().decode()
                    #Lee una imagen de una ruta específica y la almacena en la variable reminder.
                    reminder = cv2.imread(path)
                    #Genera la siguiente imagen a partir de la imagen actual convirtiendo los tipos de 
                    #datos de current_img y reminder a int16 para luego sumar los dos valores y restar 128 del resultado  
                    #obtenido de la suma. Esto se hace para ajustar el brillo de la imagen. El resultado se limita entre  
                    #0 y 255 usando np.clip para evitar valores negativos o mayores que 255.
                    next_img = np.clip(np.add(current_img.astype(np.int16), reminder.astype(np.int16)) - 128, 0, 255).astype(np.uint8)
                    #Guarda la imagen de la variable next_img en la ruta especificada en la variable path
                    cv2.imwrite(path, next_img) 
                    current_img = next_img
                index += 1
        return 


if __name__ == "__main__":
    main.main(EC.parser, logging, CoDec)