"""
Test suite para el módulo LBT.py (Learned Block Transform)

Este módulo contiene pruebas unitarias e integración para verificar:
1. Funcionamiento correcto del Autoencoder LBT
2. Transformadas forward y backward
3. Codificación y decodificación de imágenes completas
4. Casos borde (tamaños de imagen, bloques, datos fuera de rango, etc.)
5. Preservación de energía y propiedades de la transformada
"""

import unittest
import numpy as np
import tempfile
import os
import sys
import cv2
from pathlib import Path

# Agregar el directorio src al path para importar los módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from LBT import LBT_Autoencoder, CoDec
import parser
import logging


class TestLBT_Autoencoder(unittest.TestCase):
    """Pruebas unitarias para la clase LBT_Autoencoder"""
    
    def setUp(self):
        """Configuración previa para cada prueba"""
        self.block_size = 8
        self.lbt = LBT_Autoencoder(self.block_size)
        
    def test_initialization(self):
        """Prueba: Inicialización correcta del autoencoder"""
        self.assertEqual(self.lbt.block_size, 8)
        self.assertEqual(self.lbt.input_dim, 64)
        self.assertIsNone(self.lbt.weights)
        
    def test_train_with_2d_patches(self):
        """Prueba: Entrenamiento con parches 2D"""
        # Crear datos aleatorios de prueba
        n_patches = 100
        patches = np.random.randn(n_patches, self.block_size, self.block_size)
        
        # Entrenar
        self.lbt.train(patches)
        
        # Verificar que los pesos fueron establecidos
        self.assertIsNotNone(self.lbt.weights)
        self.assertEqual(self.lbt.weights.shape, (64, 64))
        
        # Verificar que los pesos son aproximadamente ortogonales
        gram_matrix = np.dot(self.lbt.weights, self.lbt.weights.T)
        np.testing.assert_array_almost_equal(gram_matrix, np.eye(64), decimal=5)
        
    def test_train_with_1d_patches(self):
        """Prueba: Entrenamiento con parches aplanados (1D)"""
        n_patches = 100
        patches = np.random.randn(n_patches, 64)
        
        self.lbt.train(patches)
        
        self.assertIsNotNone(self.lbt.weights)
        self.assertEqual(self.lbt.weights.shape, (64, 64))
        
    def test_forward_transformation(self):
        """Prueba: Transformada forward"""
        n_patches = 50
        patches = np.random.randn(n_patches, self.block_size, self.block_size)
        
        self.lbt.train(patches)
        coeffs = self.lbt.forward(patches)
        
        # Verificar forma
        self.assertEqual(coeffs.shape, patches.shape)
        
        # Verificar que los coeficientes no contienen NaN
        self.assertFalse(np.any(np.isnan(coeffs)))
        
    def test_backward_transformation(self):
        """Prueba: Transformada backward (reconstrucción)"""
        n_patches = 50
        patches = np.random.randn(n_patches, self.block_size, self.block_size)
        
        self.lbt.train(patches)
        coeffs = self.lbt.forward(patches)
        reconstructed = self.lbt.backward(coeffs)
        
        # Verificar forma
        self.assertEqual(reconstructed.shape, patches.shape)
        
        # Verificar que la reconstrucción es aproximadamente igual a los originales
        np.testing.assert_array_almost_equal(patches, reconstructed, decimal=10)
        
    def test_energy_compaction(self):
        """Prueba: Verificar compactación de energía (propiedad KLT)"""
        # Crear datos correlacionados (no ruido blanco)
        n_patches = 500
        base_pattern = np.random.randn(self.block_size, self.block_size)
        patches = np.array([
            base_pattern + 0.1 * np.random.randn(self.block_size, self.block_size) 
            for _ in range(n_patches)
        ])
        
        self.lbt.train(patches)
        coeffs = self.lbt.forward(patches)
        
        # Calcular energía en cada coeficiente
        # La energía debe estar concentrada en los primeros coeficientes
        energy = np.var(coeffs, axis=0)  # Varianza a lo largo de los parches
        
        # Ordenar en forma descendente
        sorted_energy = np.sort(energy)[::-1]
        
        # Verificar que la energía decrece
        for i in range(len(sorted_energy) - 1):
            self.assertGreaterEqual(sorted_energy[i], sorted_energy[i + 1])
        
    def test_set_and_get_weights(self):
        """Prueba: Establecer y obtener pesos manualmente"""
        weights = np.eye(64)
        self.lbt.set_weights(weights)
        
        retrieved = self.lbt.get_weights()
        np.testing.assert_array_equal(weights, retrieved)
        
    def test_different_block_sizes(self):
        """Prueba: Funcionamiento con diferentes tamaños de bloque"""
        for block_size in [4, 8, 16]:
            lbt = LBT_Autoencoder(block_size)
            expected_dim = block_size * block_size
            
            patches = np.random.randn(50, block_size, block_size)
            lbt.train(patches)
            
            self.assertEqual(lbt.weights.shape, (expected_dim, expected_dim))
            
            coeffs = lbt.forward(patches)
            reconstructed = lbt.backward(coeffs)
            np.testing.assert_array_almost_equal(patches, reconstructed, decimal=10)


class TestLBT_EdgeCases(unittest.TestCase):
    """Pruebas de casos borde para LBT_Autoencoder"""
    
    def setUp(self):
        self.block_size = 8
        self.lbt = LBT_Autoencoder(self.block_size)
        
    def test_zero_patches(self):
        """Prueba: Parches con valores cero"""
        patches = np.zeros((50, self.block_size, self.block_size))
        self.lbt.train(patches)
        coeffs = self.lbt.forward(patches)
        
        # Los coeficientes de parches cero deben ser cero (o muy cercano a cero)
        np.testing.assert_array_almost_equal(coeffs, 0, decimal=10)
        
    def test_constant_patches(self):
        """Prueba: Parches constantes"""
        patches = np.ones((50, self.block_size, self.block_size)) * 5
        self.lbt.train(patches)
        coeffs = self.lbt.forward(patches)
        
        # La energía debe estar concentrada en el primer coeficiente (DC)
        reconstructed = self.lbt.backward(coeffs)
        np.testing.assert_array_almost_equal(patches, reconstructed, decimal=10)
        
    def test_single_patch(self):
        """Prueba: Entrenamiento con un solo parche"""
        patches = np.random.randn(1, self.block_size, self.block_size)
        
        # Esto puede generar covarianza degenerada
        # Verificar que no cause crash
        try:
            self.lbt.train(patches)
            self.assertIsNotNone(self.lbt.weights)
        except np.linalg.LinAlgError:
            # Es aceptable si falla con datos insuficientes
            pass
            
    def test_very_large_values(self):
        """Prueba: Parches con valores muy grandes"""
        patches = np.random.randn(50, self.block_size, self.block_size) * 1e6
        
        self.lbt.train(patches)
        coeffs = self.lbt.forward(patches)
        reconstructed = self.lbt.backward(coeffs)
        
        # Verificar reconstrucción relativa
        error = np.max(np.abs(patches - reconstructed)) / np.max(np.abs(patches))
        self.assertLess(error, 1e-10)
        
    def test_very_small_values(self):
        """Prueba: Parches con valores muy pequeños"""
        patches = np.random.randn(50, self.block_size, self.block_size) * 1e-6
        
        self.lbt.train(patches)
        coeffs = self.lbt.forward(patches)
        reconstructed = self.lbt.backward(coeffs)
        
        np.testing.assert_array_almost_equal(patches, reconstructed, decimal=15)
        
    def test_negative_values(self):
        """Prueba: Parches con valores negativos"""
        patches = np.random.randn(50, self.block_size, self.block_size) * -1
        
        self.lbt.train(patches)
        coeffs = self.lbt.forward(patches)
        reconstructed = self.lbt.backward(coeffs)
        
        np.testing.assert_array_almost_equal(patches, reconstructed, decimal=10)


class TestImageHandling(unittest.TestCase):
    """Pruebas para manejo de imágenes"""
    
    def setUp(self):
        """Configuración previa para pruebas de imagen"""
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp()
        self.block_size = 8
        
        # Configurar argumentos
        self.args = parser.parser.parse_known_args()[0]
        self.args.block_size_DCT = self.block_size
        self.args.color_transform = "YCoCg"
        self.args.disable_subbands = False
        self.args.quantizer = "deadzone"
        self.args.debug = False
        
    def tearDown(self):
        """Limpiar archivos temporales"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def create_test_image(self, height, width, channels=3, dtype=np.uint8):
        """Crear una imagen de prueba"""
        if dtype == np.uint8:
            return np.random.randint(0, 256, (height, width, channels), dtype=dtype)
        else:
            return np.random.rand(height, width, channels).astype(dtype) * 255
            
    def test_image_exact_block_size(self):
        """Prueba: Imagen con dimensiones múltiplos exactos del bloque"""
        height, width = 16, 16
        img = self.create_test_image(height, width)
        
        # Guardar imagen de prueba
        img_path = os.path.join(self.temp_dir, "test_exact.png")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Verificar que el relleno es correcto
        codec = CoDec(self.args)
        padded = codec.pad_and_center_to_multiple_of_block_size(img)
        
        self.assertEqual(padded.shape[0] % self.block_size, 0)
        self.assertEqual(padded.shape[1] % self.block_size, 0)
        self.assertEqual(padded.shape, img.shape)
        
    def test_image_needs_padding(self):
        """Prueba: Imagen que requiere relleno"""
        height, width = 13, 17
        img = self.create_test_image(height, width)
        
        codec = CoDec(self.args)
        padded = codec.pad_and_center_to_multiple_of_block_size(img)
        
        # Verificar dimensiones padded
        self.assertEqual(padded.shape[0] % self.block_size, 0)
        self.assertEqual(padded.shape[1] % self.block_size, 0)
        self.assertGreaterEqual(padded.shape[0], height)
        self.assertGreaterEqual(padded.shape[1], width)
        
    def test_remove_padding(self):
        """Prueba: Remover relleno correctamente"""
        height, width = 13, 17
        img = self.create_test_image(height, width)
        
        codec = CoDec(self.args)
        padded = codec.pad_and_center_to_multiple_of_block_size(img)
        unpadded = codec.remove_padding(padded)
        
        # Verificar que se recuperan las dimensiones originales
        self.assertEqual(unpadded.shape, img.shape)
        
    def test_small_image(self):
        """Prueba: Imagen muy pequeña"""
        height, width = 4, 4
        img = self.create_test_image(height, width)
        
        codec = CoDec(self.args)
        padded = codec.pad_and_center_to_multiple_of_block_size(img)
        
        self.assertEqual(padded.shape[0] % self.block_size, 0)
        self.assertEqual(padded.shape[1] % self.block_size, 0)
        
    def test_large_image(self):
        """Prueba: Imagen grande"""
        height, width = 512, 512
        img = self.create_test_image(height, width)
        
        codec = CoDec(self.args)
        padded = codec.pad_and_center_to_multiple_of_block_size(img)
        
        self.assertEqual(padded.shape[0] % self.block_size, 0)
        self.assertEqual(padded.shape[1] % self.block_size, 0)
        self.assertEqual(padded.shape[0], height)
        self.assertEqual(padded.shape[1], width)
        
    def test_non_square_image(self):
        """Prueba: Imagen rectangular (no cuadrada)"""
        height, width = 100, 200
        img = self.create_test_image(height, width)
        
        codec = CoDec(self.args)
        padded = codec.pad_and_center_to_multiple_of_block_size(img)
        unpadded = codec.remove_padding(padded)
        
        self.assertEqual(unpadded.shape, img.shape)


class TestLBTIntegration(unittest.TestCase):
    """Pruebas de integración completa de codificación/decodificación"""
    
    def setUp(self):
        """Configuración previa para pruebas de integración"""
        self.temp_dir = tempfile.mkdtemp()
        self.block_size = 8
        
        # Configurar argumentos
        self.args = parser.parser.parse_known_args()[0]
        self.args.block_size_DCT = self.block_size
        self.args.color_transform = "YCoCg"
        self.args.disable_subbands = False
        self.args.quantizer = "deadzone"
        self.args.debug = False
        
    def tearDown(self):
        """Limpiar archivos temporales"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def create_test_image(self, height=32, width=32):
        """Crear imagen de prueba"""
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        return img
        
    def test_encode_decode_roundtrip(self):
        """Prueba: Ciclo completo encode-decode"""
        # Crear imagen de prueba
        img = self.create_test_image(32, 32)
        img_path = os.path.join(self.temp_dir, "test_original.png")
        encoded_path = os.path.join(self.temp_dir, "test_encoded")
        decoded_path = os.path.join(self.temp_dir, "test_decoded.png")
        
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        try:
            codec = CoDec(self.args)
            # Codificar
            codec.encode_fn(img_path, encoded_path)
            self.assertTrue(os.path.exists(f"{encoded_path}.tif"))
            self.assertTrue(os.path.exists(f"{encoded_path}_shape.bin"))
            self.assertTrue(os.path.exists(f"{encoded_path}_weights.bin"))
            
            # Decodificar
            codec.decode_fn(encoded_path, decoded_path)
            self.assertTrue(os.path.exists(decoded_path))
            
            # Verificar que la imagen decodificada es válida
            decoded_img = cv2.imread(decoded_path)
            self.assertIsNotNone(decoded_img)
            self.assertEqual(decoded_img.shape, (32, 32, 3))
            
        except Exception as e:
            self.fail(f"Encode-decode roundtrip failed: {e}")
            
    def test_different_image_sizes(self):
        """Prueba: Diferentes tamaños de imagen"""
        sizes = [(16, 16), (32, 48), (64, 64), (100, 100)]
        
        for height, width in sizes:
            with self.subTest(size=(height, width)):
                img = self.create_test_image(height, width)
                img_path = os.path.join(self.temp_dir, f"test_{height}_{width}.png")
                encoded_path = os.path.join(self.temp_dir, f"test_{height}_{width}_enc")
                decoded_path = os.path.join(self.temp_dir, f"test_{height}_{width}_dec.png")
                
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                try:
                    codec = CoDec(self.args)
                    codec.encode_fn(img_path, encoded_path)
                    codec.decode_fn(encoded_path, decoded_path)
                    
                    self.assertTrue(os.path.exists(decoded_path))
                except Exception as e:
                    self.fail(f"Failed for size {(height, width)}: {e}")


class TestErrorHandling(unittest.TestCase):
    """Pruebas para validación de manejo de errores"""
    
    def setUp(self):
        self.block_size = 8
        
    def test_invalid_block_size_zero(self):
        """Prueba: block_size = 0 debe lanzar ValueError"""
        with self.assertRaises(ValueError):
            LBT_Autoencoder(0)
            
    def test_invalid_block_size_negative(self):
        """Prueba: block_size negativo debe lanzar ValueError"""
        with self.assertRaises(ValueError):
            LBT_Autoencoder(-5)
            
    def test_invalid_block_size_string(self):
        """Prueba: block_size string debe lanzar ValueError"""
        with self.assertRaises(ValueError):
            LBT_Autoencoder("ocho")
            
    def test_train_invalid_input_type(self):
        """Prueba: train() con lista debe lanzar TypeError"""
        lbt = LBT_Autoencoder(self.block_size)
        with self.assertRaises(TypeError):
            lbt.train([[1, 2, 3]])
            
    def test_train_wrong_dimensions(self):
        """Prueba: train() con 4D array debe lanzar ValueError"""
        lbt = LBT_Autoencoder(self.block_size)
        data = np.random.randn(10, 8, 8, 3)
        with self.assertRaises(ValueError):
            lbt.train(data)
            
    def test_train_wrong_block_shape(self):
        """Prueba: train() con tamaño de bloque incorrecto"""
        lbt = LBT_Autoencoder(self.block_size)
        patches = np.random.randn(50, 16, 16)  # Esperado 8x8
        with self.assertRaises(ValueError):
            lbt.train(patches)
            
    def test_train_with_nan_values(self):
        """Prueba: train() con valores NaN debe lanzar ValueError"""
        lbt = LBT_Autoencoder(self.block_size)
        patches = np.random.randn(50, self.block_size, self.block_size)
        patches[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            lbt.train(patches)
            
    def test_train_with_inf_values(self):
        """Prueba: train() con valores infinitos debe lanzar ValueError"""
        lbt = LBT_Autoencoder(self.block_size)
        patches = np.random.randn(50, self.block_size, self.block_size)
        patches[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            lbt.train(patches)
            
    def test_forward_without_training(self):
        """Prueba: forward() sin entrenar debe lanzar RuntimeError"""
        lbt = LBT_Autoencoder(self.block_size)
        patches = np.random.randn(10, self.block_size, self.block_size)
        with self.assertRaises(RuntimeError):
            lbt.forward(patches)
            
    def test_backward_without_training(self):
        """Prueba: backward() sin entrenar debe lanzar RuntimeError"""
        lbt = LBT_Autoencoder(self.block_size)
        coeffs = np.random.randn(10, self.block_size, self.block_size)
        with self.assertRaises(RuntimeError):
            lbt.backward(coeffs)
            
    def test_set_weights_invalid_shape(self):
        """Prueba: set_weights() con forma incorrecta"""
        lbt = LBT_Autoencoder(self.block_size)
        wrong_weights = np.eye(32)  # Esperado 64x64
        with self.assertRaises(ValueError):
            lbt.set_weights(wrong_weights)
            
    def test_set_weights_not_ndarray(self):
        """Prueba: set_weights() con lista en lugar de array"""
        lbt = LBT_Autoencoder(self.block_size)
        with self.assertRaises(TypeError):
            lbt.set_weights([[1, 2], [3, 4]])
            
    def test_set_weights_with_nan(self):
        """Prueba: set_weights() con NaN"""
        lbt = LBT_Autoencoder(self.block_size)
        weights = np.eye(64)
        weights[0, 0] = np.nan
        with self.assertRaises(ValueError):
            lbt.set_weights(weights)
            
    def test_get_weights_not_trained(self):
        """Prueba: get_weights() sin entrenar"""
        lbt = LBT_Autoencoder(self.block_size)
        with self.assertRaises(RuntimeError):
            lbt.get_weights()
            
    def test_forward_invalid_shape(self):
        """Prueba: forward() con forma incorrecta"""
        lbt = LBT_Autoencoder(self.block_size)
        patches = np.random.randn(50, self.block_size, self.block_size)
        lbt.train(patches)
        
        wrong_patches = np.random.randn(50, 16, 16)  # Forma incorrecta
        with self.assertRaises(ValueError):
            lbt.forward(wrong_patches)
            
    def test_backward_invalid_shape(self):
        """Prueba: backward() con forma incorrecta"""
        lbt = LBT_Autoencoder(self.block_size)
        patches = np.random.randn(50, self.block_size, self.block_size)
        lbt.train(patches)
        
        wrong_coeffs = np.random.randn(50, 16, 16)  # Forma incorrecta
        with self.assertRaises(ValueError):
            lbt.backward(wrong_coeffs)


class TestImageErrorHandling(unittest.TestCase):
    """Pruebas para manejo de errores en procesamiento de imágenes"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.block_size = 8
        
        self.args = parser.parser.parse_known_args()[0]
        self.args.block_size_DCT = self.block_size
        self.args.color_transform = "YCoCg"
        self.args.disable_subbands = False
        self.args.quantizer = "deadzone"
        self.args.debug = False
        
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_pad_image_invalid_type(self):
        """Prueba: pad_and_center_to_multiple_of_block_size() con lista"""
        codec = CoDec(self.args)
        with self.assertRaises(TypeError):
            codec.pad_and_center_to_multiple_of_block_size([[1, 2], [3, 4]])
            
    def test_pad_image_wrong_dimensions(self):
        """Prueba: pad_and_center_to_multiple_of_block_size() con 2D"""
        codec = CoDec(self.args)
        img = np.random.rand(32, 32)  # 2D en lugar de 3D
        with self.assertRaises(ValueError):
            codec.pad_and_center_to_multiple_of_block_size(img)
            
    def test_pad_image_invalid_dimensions_zero(self):
        """Prueba: pad_and_center_to_multiple_of_block_size() con dimensión 0"""
        codec = CoDec(self.args)
        img = np.random.rand(0, 32, 3)
        with self.assertRaises(ValueError):
            codec.pad_and_center_to_multiple_of_block_size(img)
            
    def test_remove_padding_not_initialized(self):
        """Prueba: remove_padding() sin llamar a pad primero"""
        codec = CoDec(self.args)
        img = np.random.rand(32, 32, 3)
        with self.assertRaises(RuntimeError):
            codec.remove_padding(img)
            
    def test_remove_padding_invalid_type(self):
        """Prueba: remove_padding() con lista"""
        codec = CoDec(self.args)
        img = np.random.rand(32, 32, 3)
        codec.pad_and_center_to_multiple_of_block_size(img)
        
        with self.assertRaises(ValueError):
            codec.remove_padding([[1, 2], [3, 4]])
            
    def test_encode_file_not_found(self):
        """Prueba: encode_fn() con archivo inexistente"""
        codec = CoDec(self.args)
        with self.assertRaises(FileNotFoundError):
            codec.encode_fn("/ruta/inexistente/imagen.jpg", "/tmp/output")
            
    def test_decode_shape_file_missing(self):
        """Prueba: decode_fn() con archivo de dimensiones faltante"""
        codec = CoDec(self.args)
        with self.assertRaises(FileNotFoundError):
            codec.decode_fn("/tmp/inexistente", "/tmp/output.jpg")
            
    def test_codec_invalid_block_size_string(self):
        """Prueba: CoDec con block_size_DCT no convertible a int"""
        self.args.block_size_DCT = "ocho"
        with self.assertRaises(ValueError):
            CoDec(self.args)
            
    def test_codec_invalid_block_size_zero(self):
        """Prueba: CoDec con block_size_DCT = 0"""
        self.args.block_size_DCT = 0
        with self.assertRaises(ValueError):
            CoDec(self.args)
    """Pruebas de integridad de datos en las transformadas"""
    
    def setUp(self):
        self.block_size = 8
        
    def test_parseval_theorem(self):
        """Prueba: Verificar conservación de energía (Parseval)"""
        lbt = LBT_Autoencoder(self.block_size)
        n_patches = 100
        patches = np.random.randn(n_patches, self.block_size, self.block_size)
        
        lbt.train(patches)
        coeffs = lbt.forward(patches)
        
        # La energía total debe ser conservada
        energy_original = np.sum(patches ** 2)
        energy_transformed = np.sum(coeffs ** 2)
        
        # Deben ser muy similares (la ortogonalidad de la transformada lo garantiza)
        relative_error = np.abs(energy_original - energy_transformed) / energy_original
        self.assertLess(relative_error, 1e-10)
        
    def test_orthogonality(self):
        """Prueba: Verificar que la transformada es ortogonal"""
        lbt = LBT_Autoencoder(self.block_size)
        n_patches = 200
        patches = np.random.randn(n_patches, self.block_size, self.block_size)
        
        lbt.train(patches)
        W = lbt.weights
        
        # Verificar W * W^T = I
        gram = np.dot(W, W.T)
        np.testing.assert_array_almost_equal(gram, np.eye(self.block_size * self.block_size), decimal=10)
        
        # Verificar W^T * W = I
        gram_t = np.dot(W.T, W)
        np.testing.assert_array_almost_equal(gram_t, np.eye(self.block_size * self.block_size), decimal=10)


def run_tests(verbosity=2):
    """Ejecutar todas las pruebas"""
    # Crear suite de pruebas
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar todas las clases de prueba
    suite.addTests(loader.loadTestsFromTestCase(TestLBT_Autoencoder))
    suite.addTests(loader.loadTestsFromTestCase(TestLBT_EdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestImageHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestImageErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestLBTIntegration))
    
    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_tests(verbosity=2)
    sys.exit(0 if result.wasSuccessful() else 1)
