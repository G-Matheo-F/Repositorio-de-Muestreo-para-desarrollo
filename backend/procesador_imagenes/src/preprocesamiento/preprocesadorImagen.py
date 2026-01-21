"""
Clase para preprocesamiento de imágenes
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, Dict
from pathlib import Path


class PreprocesadorImagen:
    """Clase para el preprocesamiento de imágenes con diversos algoritmos"""
    
    def __init__(self, tamanoObjetivo: Tuple[int, int] = (224, 224)):
        """
        Inicializa el preprocesador
        
        Args:
            tamanoObjetivo: Tamaño objetivo (altura, ancho)
        """
        self.tamanoObjetivo = tamanoObjetivo
    
    def cargarImagen(self, rutaImagen: str) -> np.ndarray:
        """
        Carga una imagen desde archivo
        
        Args:
            rutaImagen: Ruta de la imagen
            
        Returns:
            Imagen en formato BGR (OpenCV)
        """
        
        ruta = Path(rutaImagen)
        if not ruta.exists():# Si no existe el archivo
            raise FileNotFoundError(f"No existe la imagen: {rutaImagen}")
        
        imagen = None
        
        # Intento 1: cv2.imread normal
        try:
            imagen = cv2.imread(str(ruta))
            if imagen is not None and imagen.size > 0:
                return imagen
        except Exception as e:
            print(f"Advertencia: Error con cv2.imread: {e}")
        
        # Intento 2: Lectura alternativa desde bytes (Unicode-safe)
        try:
            with open(ruta, 'rb') as f:
                data = f.read()
            imagen = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if imagen is not None and imagen.size > 0:
                return imagen
        except Exception as e:
            print(f"Advertencia: Error al cargar la imagen con cv2.imdecode: {rutaImagen} - {e}")
        
        # Si ambos métodos fallan
        raise ValueError(f"No se pudo cargar la imagen: {rutaImagen}")
    
    def redimensionar(self, imagen: np.ndarray, tamanoObjetivo: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Redimensiona la imagen manteniendo aspecto o ajustando
        
        Args:
            imagen: Imagen de entrada
            tamanoObjetivo: Tamaño objetivo (altura, ancho)
            
        Returns:
            Imagen redimensionada
        """
        if imagen is None or imagen.size == 0:
            raise ValueError("La imagen es None o está vacía")
        
        if tamanoObjetivo is None:
            tamanoObjetivo = self.tamanoObjetivo

        # Redimensionar con interpolación de calidad
        redimensionada = cv2.resize(imagen, (tamanoObjetivo[1], tamanoObjetivo[0]), 
                                    interpolation=cv2.INTER_LANCZOS4)
        return redimensionada
    
    def aEscalaGrises(self, imagen: np.ndarray) -> np.ndarray:
        """
        Convierte imagen a escala de grises
        
        Args:
            imagen: Imagen en BGR
            
        Returns:
            Imagen en escala de grises
        """
        if len(imagen.shape) == 2:
            return imagen
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    def normalizar(self, imagen: np.ndarray) -> np.ndarray:
        """
        Normaliza los valores de píxeles a [0, 1]
        
        Args:
            imagen: Imagen de entrada
            
        Returns:
            Imagen normalizada
        """
        # Convertir a float32
        imagen = imagen.astype(np.float32)
        
        # Revisar valores mínimo y máximo
        valorMaximo = np.max(imagen)
        
        # Si los valores máximos superan 1, asumimos rango 0-255
        if valorMaximo > 1.0:
            imagenNormalizada = imagen / 255.0
        else:
            imagenNormalizada = imagen
        
        # Asegurar que quede estrictamente entre 0 y 1
        imagenNormalizada = np.clip(imagenNormalizada, 0.0, 1.0)
        
        return imagenNormalizada
    
    def binarizar(self, imagen: np.ndarray, umbral: int = 127) -> np.ndarray:
        """
        Binariza la imagen usando umbral de Otsu o valor fijo
        
        Args:
            imagen: Imagen en escala de grises
            umbral: Valor de umbral (si None, usa Otsu)
            
        Returns:
            Imagen binarizada
        """
        if umbral is None:
            # Usar umbral de Otsu
            _, binaria = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return binaria
        else:
            _, binaria = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY_INV)
            return binaria
    
    def obtenerParametrosStretchingAutomaticos(self,imgGris):
        """
        Calcula los parámetros para el estiramiento del histograma de forma automática
        Args:
            imgGris: Imagen en escala de grises
        Returns:
            Diccionario con los parámetros a, b, alfa, beta, gamma
        """
        #Se divide en las 3 zonas del histograma
        a,b=85,170
        histograma, _ = np.histogram(imgGris, bins=256, range=(0,256))
        #Calcular las ganancias alfa, beta y gamma basadas en la distribución del histograma
        totalPixeles=imgGris.size
        #Calcular proporciones en cada zona
        propRangoSombras=np.sum(histograma[0:a+1])/totalPixeles
        propRangoMedios=np.sum(histograma[a+1:b+1])/totalPixeles
        propRangoAltos=np.sum(histograma[b+1:256])/totalPixeles
        #Factores adaptativos: más proporción en un rango -> más estiramiento, menos porporción -> mas compresión
        minFactor,maxFactor=0.1,3.0
        alfa=minFactor+(maxFactor-minFactor)*propRangoSombras
        beta=minFactor+(maxFactor-minFactor)*propRangoMedios
        gamma=minFactor+(maxFactor-minFactor)*propRangoAltos
        #Asegur que no se salgan a valores tan extremos
        alfa=max(minFactor, min(alfa,maxFactor))
        beta=max(minFactor, min(beta,maxFactor))
        gamma=max(minFactor,min(gamma,maxFactor))
        return {"a": a, "b": b,"alfa": alfa,"beta": beta,"gamma": gamma}
    
    def miStreching(self,P,a,b,alfa,beta,gamma):
        """
        Realiza el estiramiento del histograma de una imagen P usando los parámetros dados.
        Args:
            P: Imagen en escala de grises (matriz 2D)
            a: Valor mínimo del rango de entrada
            b: Valor máximo del rango de entrada
            alfa: Ganancia para valores menores que a
            beta: Ganancia para valores entre a y b
            gamma: Ganancia para valores mayores que b
        """
        P=np.array(P)
        if P.ndim != 2:
            print("Error: P debe ser una matriz 2D de números.")
            return None
        #Verificar si los valores de la imagen están normalizados entre 0 y 1
        if P.max() <= 1.0:
            P=(P * 255).astype(np.uint8) #Escalar a 0-255 y convertir a uint8, por el histograma

        #Validar los valores de a y b
        if a < 0 or a > 255 or b < 0 or b > 255 or a >= b:
            print("Error: a y b deben estar en el rango de 0 a 255 y a debe ser menor que b.")
            return None
        
        Va=alfa*a
        Vb=beta*(b - a) + Va
        #Crear la tabla LUT para el stretching del histograma
        tablaLUT=np.zeros(256,dtype=np.uint8)#Crea un arreglo de 0 a 255 para calcular la tabla LUT y luego mapear los valores de P
        for valorU in range(256):
            if valorU>=0 and valorU < a:#Si el valor es menor que a se asigna la ganancia alfa
                tablaLUT[valorU]=np.clip(round(alfa*valorU),0,255)#Se usa np.clip para asegurar que el valor esté entre 0 y 255
            elif valorU >= a and valorU <= b:#Si el valor está entre a y b se asigna la ganancia beta
                tablaLUT[valorU]=np.clip(round(beta*(valorU - a) + Va),0,255)
            else:#Si el valor es mayor que b se asigna la ganancia gamma
                tablaLUT[valorU]=np.clip(round(gamma*(valorU - b) + Vb),0,255)
        #Aplicar la tabla LUT a la imagen P para obtener la imagen ampliada X
        #Mapear los valores de P usando la tabla LUT, gracias a numpy se filtra automáticamente cada valor de P y lo reemplaza por su correspondiente en tablaLUT
        X=tablaLUT[P]
        return X  
    
    def ecualizarHistograma(self, imagen: np.ndarray) -> np.ndarray:
        """
        Ecualiza el histograma de la imagen
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Imagen con histograma ecualizado
        """
        return cv2.equalizeHist(imagen)
    
        #Función para aplicar ampliación del histograma
    def ampliacionHistograma(self,P: np.ndarray,a:int,b:int,L:int=255) -> np.ndarray:
        """
        Realiza la ampliación del histograma de una imagen P usando los parámetros dados.
        Args:
            P: Imagen en escala de grises (matriz 2D)
            a: Valor mínimo del rango de entrada
            b: Valor máximo del rango de entrada
            L: Nivel máximo de intensidad (normalmente 255)
        """
        P=np.array(P)
        if P.ndim != 2:
            print("Error: P debe ser una matriz 2D de números.")
            return None
        #Verificar si los valores de la imagen están normalizados entre 0 y 1
        if P.max() <= 1.0:
            P=(P * 255).astype(np.uint8) #Escalar a 0-255 y convertir a uint8, por el histograma

        #Validar los valores de a y b
        if a < 0 or a > 255 or b < 0 or b > 255 or a >= b:
            print("Error: a y b deben estar en el rango de 0 a 255 y a debe ser menor que b.")
            return None
        
        #Crear la tabla LUT para la ampliación del histograma

        tablaLUT=np.zeros(256,dtype=np.uint8)
        for valorU in range(256):
            if valorU < a:#Si el valor es menor que a se asigna 0
                tablaLUT[valorU]=0
            elif valorU > b:#Si el valor es mayor que b se asigna L
                tablaLUT[valorU]=L
            else:#Se aplca la fórmula de ampliación del histograma guardando en la tabla LUT
                operacion=(L*1.0*(valorU-a))/(b-a)
                tablaLUT[valorU]=round(operacion)

        #Aplicar la tabla LUT a la imagen P para obtener la imagen ampliada X
        X=tablaLUT[P]#Mapear los valores de P usando la tabla LUT
        return X     
    
    def micuadrada(self,P: np.ndarray,O:int,L) -> np.ndarray:
        """
        Realiza la función cuadrada o raíz cuadrada en una imagen P usando los parámetros dados.
        Args:
            P: Imagen en escala de grises (matriz 2D)
            O: Tipo de función (0 = cuadrada, 1 = raíz cuadrada)
            L: Nivel máximo de intensidad (normalmente 255)
        """
        P=np.array(P)
        if P.ndim != 2:
            print("Error: P debe ser una matriz 2D de números.")
            return None
        #Verificar si los valores de la imagen están normalizados entre 0 y 1
        if P.max() <= 1.0:
            P=(P * 255).astype(np.uint8) #Escalar a 0-255 y convertir a uint8, por el histograma

        if O not in [0,1]:
            print("Error: O debe ser 0 (función cuadrada) o 1 (raíz cuadrada).")
            return None
        
        #Crear la tabla LUT para la función cuadrada o raíz cuadrada
        tablaLUT=np.zeros(L+1,dtype=np.uint8)#Crea un arreglo de 0 a 255 para calcular la tabla LUT y luego mapear los valores de P
        for valorU in range(L+1):
            if O==0:#Función cuadrada
                tablaLUT[valorU]=np.clip(round((valorU**2)/L),0,255)
            else:#Raíz cuadrada
                tablaLUT[valorU]=np.clip(round(np.sqrt(L*valorU)),0,255)
        X=tablaLUT[P] #Mapear los valores de P usando la tabla LUT
        #Streching final para aprovechar todo el rango de 0-255
        parametrosStretching=self.obtenerParametrosStretchingAutomaticos(X)
        X=self.miStreching(X,a=parametrosStretching['a'],b=parametrosStretching['b'],
                            alfa=parametrosStretching['alfa'],beta=parametrosStretching['beta'],
                            gamma=parametrosStretching['gamma'])
        return X
    

    def calcularEstadisticas(self, imagen: np.ndarray) -> Dict[str, float]:
        """
        Calcula estadísticas de la imagen para análisis
        
        Args:
            imagen: Imagen de entrada
            
        Returns:
            Diccionario con estadísticas
        """
        # Convertir a escala de grises si es necesario
        if len(imagen.shape) == 3:
            imagenGray = self.aEscalaGrises(imagen)
        else:
            imagenGray = imagen
        
        estadisticas = {
            'media': float(np.mean(imagenGray)),
            'varianza': float(np.var(imagenGray)),
            'desviacionEstandar': float(np.std(imagenGray)),
            'minimo': float(np.min(imagenGray)),
            'maximo': float(np.max(imagenGray)),
            'rangoTonal': float(np.max(imagenGray) - np.min(imagenGray)),
            'mediana': float(np.median(imagenGray))
        }
        
        return estadisticas
    
    def analizarImagen(self, imagen: np.ndarray, fondoBlanco: bool = True) -> Dict[str, any]:
        """
        Analiza la imagen y determina si está subexpuesta, sobreexpuesta,
        tiene bajo contraste, saturación, o está equilibrada
        
        Args:
            imagen: Imagen de entrada
            fondoBlanco: Si la imagen tiene fondo blanco (default True)
            
        Returns:
            Diccionario con el diagnóstico y estadísticas
        """
        # Calcular estadísticas
        stats = self.calcularEstadisticas(imagen)
        
        # Convertir a escala de grises para análisis de histograma
        if len(imagen.shape) == 3:
            imagenGray = self.aEscalaGrises(imagen)
        else:
            imagenGray = imagen

        # Calcular histograma
        histograma = cv2.calcHist([imagenGray], [0], None, [256], [0, 256])
        histograma = histograma.flatten()
        histogramaNorm = histograma / histograma.sum()
        
        # Análisis de distribución del histograma
        a, b = 85, 170
        
        propRangoSombras = histogramaNorm[0:a+1].sum()# Rango 1: 0-85 (tonos oscuros)
        propRangoMedios = histogramaNorm[a+1:b+1].sum()# Rango 2: 86-170 (tonos medios)
        propRangoAltos = histogramaNorm[b+1:256].sum()# Rango 3: 171-255 (tonos claros)

        if fondoBlanco:
            # Considerar solo píxeles con intensidad < 240 para análisis
            mascara = imagenGray < 240
            if mascara.sum() > 0:
                pixelesObjeto = imagenGray[mascara]
                mediaObjeto = float(np.mean(pixelesObjeto))
                stdObjeto = float(np.std(pixelesObjeto))
            else:
                mediaObjeto = stats['media']
                stdObjeto = stats['desviacionEstandar']
        else:
            mediaObjeto = stats['media']
            stdObjeto = stats['desviacionEstandar']
        
        # Determinar condición
        condicion = 'equilibrada'
        razon = 'La imagen presenta una distribución tonal adecuada'
        transformacionRecomendada = 'ampliacionHistograma'
        
        UMBRAL_BAJO_CONTRASTE = 30
        UMBRAL_CONTRASTE_MEJORABLE = 50
        UMBRAL_MEDIA_BAJA = 85
        UMBRAL_MEDIA_ALTA = 170
        UMBRAL_DOMINANCIA = 0.6

        # Bajo contraste: desviación estándar baja
        if stdObjeto < UMBRAL_BAJO_CONTRASTE:
            condicion = 'bajoContraste'
            razon = f'Desviación estándar baja ({stdObjeto:.2f}) → bajo contraste'
            transformacionRecomendada = 'ecualizarHistograma'

        #Subexpuesta: media baja
        elif mediaObjeto < UMBRAL_MEDIA_BAJA:
            condicion = 'subexpuesta'
            razon = f'Media baja ({mediaObjeto:.2f}) → imagen oscura'
            transformacionRecomendada = 'funcionRaizCuadrada'

        # Subexpuesta: predominio de tonos oscuros
        elif propRangoSombras > UMBRAL_DOMINANCIA:
            condicion = 'subexpuesta'
            razon = f'Predominio de tonos oscuros ({propRangoSombras:.2f})'
            transformacionRecomendada = 'funcionRaizCuadrada'

        # Sobreexpuesta: media alta
        elif mediaObjeto > UMBRAL_MEDIA_ALTA:
            condicion = 'sobreexpuesta'
            razon = f'Media alta ({mediaObjeto:.2f}) → imagen clara'
            transformacionRecomendada = 'funcionCuadrada'

        # Sobreexpuesta: predominio de tonos claros
        elif propRangoAltos > UMBRAL_DOMINANCIA:
            condicion = 'sobreexpuesta'
            razon = f'Predominio de tonos claros ({propRangoAltos:.2f})'
            transformacionRecomendada = 'funcionCuadrada'

        # Contraste aceptable pero mejorable
        elif stdObjeto < UMBRAL_CONTRASTE_MEJORABLE:
            condicion = 'contrasteMejorable'
            razon = f'Contraste moderado ({stdObjeto:.2f}), puede mejorarse'
            transformacionRecomendada = 'streching'
        else:
            condicion = 'equilibrada'
            razon = 'La imagen presenta una distribución tonal adecuada'
            transformacionRecomendada = 'ampliacionHistograma'
        
        return {
            'condicion': condicion,
            'razon': razon,
            'transformacionRecomendada': transformacionRecomendada,
            'estadisticas': stats,
            'mediaObjeto': mediaObjeto if fondoBlanco else stats['media'],
            'stdObjeto': stdObjeto if fondoBlanco else stats['desviacionEstandar']
        }
    
    def aplicarCorreccionAutomatica(self, imagen: np.ndarray, fondoBlanco: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Analiza la imagen y aplica automáticamente la corrección apropiada
        
        Args:
            imagen: Imagen de entrada
            fondoBlanco: Si la imagen tiene fondo blanco (default True)
            
        Returns:
            Tupla con (imagen corregida, información del análisis)
        """
        # Analizar la imagen
        analisis = self.analizarImagen(imagen, fondoBlanco)
        
        # Convertir a escala de grises si es necesario para procesamiento
        if len(imagen.shape) == 3:
            imagenGray = self.aEscalaGrises(imagen)
        else:
            imagenGray = imagen.copy()
        
        # Aplicar transformación recomendada
        transformacion = analisis['transformacionRecomendada']
        
        if transformacion is None:
            # Imagen equilibrada, no se requiere corrección
            imagenCorregida = imagenGray
        elif transformacion == 'streching':
            parametros = self.obtenerParametrosStretchingAutomaticos(imagenGray)
            imagenCorregida = self.miStreching(imagenGray, a=parametros['a'], b=parametros['b'],
                                                alfa=parametros['alfa'], beta=parametros['beta'],
                                                gamma=parametros['gamma'])
            
        elif transformacion == 'ampliacionHistograma':
            imagenCorregida = self.ampliacionHistograma(imagenGray, a=imagenGray.min(), b=imagenGray.max())
        elif transformacion == 'funcionRaizCuadrada':
            imagenCorregida = self.micuadrada(imagenGray,O=1,L=255)
        elif transformacion == 'funcionCuadrada':
            imagenCorregida = self.micuadrada(imagenGray,O=0,L=255)
        elif transformacion == 'ecualizarHistograma':
            imagenCorregida = self.ecualizarHistograma(imagenGray)
        else:
            imagenCorregida = imagenGray
        
        # Agregar información sobre la transformación aplicada
        analisis['transformacionAplicada'] = transformacion
        
        return imagenCorregida, analisis
    
    def guardarImagen(self, imagen: np.ndarray, rutaSalida: str, crearDirectorio: bool = True) -> bool:
        """
        Guarda la imagen procesada en disco
        
        Args:
            imagen: Imagen a guardar
            rutaSalida: Ruta donde guardar la imagen
            crearDirectorio: Si crear el directorio si no existe
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            # Asegurar formato uint8 para escritura. Si viene normalizada [0,1], escalar.
            if imagen.dtype in (np.float32, np.float64):
                factor = 255.0 if imagen.max() <= 1.0 else 1.0
                imagen = np.clip(imagen * factor, 0, 255).astype(np.uint8)
            elif imagen.dtype != np.uint8:
                imagen = np.clip(imagen, 0, 255).astype(np.uint8)

            # Crear directorio si no existe
            if crearDirectorio:
                directorio = os.path.dirname(rutaSalida)
                if directorio and not os.path.exists(directorio):
                    os.makedirs(directorio, exist_ok=True)
            
            # Guardar imagen
            exito = cv2.imwrite(rutaSalida, imagen)
            
            """if exito:
                print(f"Imagen guardada exitosamente en: {rutaSalida}")
            else:
                print(f"Error al guardar la imagen en: {rutaSalida}")"""
            
            return exito
        except Exception as e:
            print(f"Error al guardar imagen: {str(e)}")
            return False
    
    def preprocesar(self, rutaImagen: str, 
                    espacioColor: str = 'grayscale',
                    normalizar: bool = False,
                    binarizar: bool = False,
                    correccionAutomatica: bool = False,
                    fondoBlanco: bool = True,
                    tamanoObjetivo: Optional[Tuple[int, int]] = None,
                    guardar: bool = False,
                    rutaSalida: Optional[str] = None) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Realiza el preprocesamiento completo de una imagen
        
        Args:
            rutaImagen: Ruta de la imagen
            espacioColor: 'rgb', 'bgr' o 'grayscale'
            normalizar: Si normalizar a [0, 1]
            binarizar: Si binarizar la imagen
            correccionAutomatica: Si aplicar corrección automática según análisis
            fondoBlanco: Si la imagen tiene fondo blanco (para análisis automático)
            tamanoObjetivo: Tamaño objetivo
            guardar: Si guardar la imagen procesada
            rutaSalida: Ruta donde guardar la imagen (si guardar=True)
            
        Returns:
            Tupla con (imagen preprocesada, análisis de la imagen si correccionAutomatica=True)
        """
        if tamanoObjetivo is None:
            tamanoObjetivo = self.tamanoObjetivo
        
        # Cargar y redimensionar
        imagen = self.cargarImagen(rutaImagen)
        imagen = self.redimensionar(imagen, tamanoObjetivo)
        
        # Conversión de espacio de color
        if espacioColor == 'grayscale':
            imagen = self.aEscalaGrises(imagen)
        elif espacioColor == 'rgb':
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        analisis = None
        
        # Aplicar corrección automática si se requiere (sin stretching)
        if correccionAutomatica:
            imagen, analisis = self.aplicarCorreccionAutomatica(imagen, fondoBlanco)
        
        # Normalizar si se requiere
        if normalizar:
            imagen = self.normalizar(imagen)
        
        # Binarizar si se requiere
        if binarizar and len(imagen.shape) == 2:
            if normalizar:
                # Convertir de vuelta a uint8 para binarización
                imagen = (imagen * 255).astype(np.uint8)
            imagen = self.binarizar(imagen)
            if normalizar:
                imagen = self.normalizar(imagen)
        
        # Guardar si se requiere
        if guardar and rutaSalida:
            self.guardarImagen(imagen, rutaSalida)
        
        return imagen, analisis

    def preprocesarEnMemoria(self, imagen: np.ndarray,
                              espacioColor: str = 'grayscale',
                              normalizar: bool = False,
                              binarizar: bool = False,
                              estirar: bool = False,
                              correccionAutomatica: bool = False,
                              fondoBlanco: bool = True,
                              returnBinaria: bool = False) -> Tuple[np.ndarray, Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
        """Preprocesa una imagen ya cargada en memoria (una sola vez)."""

        if imagen is None or imagen.size == 0:
            raise ValueError("La imagen es None o está vacía")

        img = imagen.copy()

        # Conversión de espacio de color
        if espacioColor == 'grayscale':
            img = self.aEscalaGrises(img)
        elif espacioColor == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        analisis = None

        # Corrección automática
        if correccionAutomatica:
            img, analisis = self.aplicarCorreccionAutomatica(img, fondoBlanco)

        # Normalización
        if normalizar:
            img = self.normalizar(img)

        preNoBin = None
        binaria = None
        if binarizar and len(img.shape) == 2:
            preNoBin = img.copy()
            imgUint8 = (img * 255).astype(np.uint8) if normalizar else img
            binaria = self.binarizar(imgUint8)
            if normalizar:
                binaria = self.normalizar(binaria)

        if returnBinaria:
            return (binaria if binaria is not None else img, analisis, binaria, preNoBin)
        return (img, analisis, None, None)
