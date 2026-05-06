# Localización de Fuentes de PM10/PM2.5 con Adaptive Inverse PINNs y Agentes MLOps

Este repositorio contiene el código fuente para un proyecto de investigación enfocado en resolver el problema inverso de dispersión de material particulado (PM2.5 y PM10) en el **Valle de Aburrá, Colombia**, utilizando **Redes Neuronales Informadas por la Física (PINNs)** apoyadas por una arquitectura **Multi-Agente**.

## 📌 Contexto del Problema

El Valle de Aburrá presenta una topografía compleja (un cañón estrecho) y condiciones meteorológicas variables que dificultan la identificación precisa de los "hotspots" o fuentes de emisión de contaminación del aire. Al tratar de identificar estas fuentes utilizando exclusivamente los datos de los sensores, nos enfrentamos a un problema matemático "mal planteado" (*ill-posed*).

Para solucionar esto, integramos:
1. **Datos de Alta Densidad**: Red de "Ciudadanos Científicos" del SIATA.
2. **Restricción Física**: Ecuación de Advección-Difusión-Reacción (ADR).
3. **Optimización Agéntica**: Modelos de Lenguaje (LLMs) orquestando el entrenamiento y validando la termodinámica.

---

## 🏗️ Arquitectura y Estado del Proyecto (Curriculum Learning)

El proyecto se está desarrollando de forma iterativa y "hueso a hueso" para garantizar estabilidad matemática.

### ✅ Fase 1: Fundamentos Geoespaciales y Datos (Python)
**Estado:** `Completado`
- **Ingeniería de Datos**: Cliente automatizado para la Red de Ciudadanos Científicos de SIATA.
- **Filtrado de Ruido**: Algoritmo `IsolationForest` para detectar y descartar lecturas anómalas por descalibración de sensores *low-cost*.
- **Restricción de Dominio ($\Omega$)**: Bounding box topológico del Valle de Aburrá para evitar distorsión de la matriz Jacobiana con coordenadas irreales.
- **Preprocesamiento Adimensional**: Mapeo estricto del espacio a $[-1, 1]$ y tiempo a $[0, 1]$.

### ✅ Fase 2: Motor Físico PINN (Julia)
**Estado:** `Completado (Script Base)`
- **Framework**: `NeuralPDE.jl` y `ModelingToolkit.jl` por su alto rendimiento.
- **Modelo de Viento Topológico**: Se sustituyó el viento constante por un perfil parabólico de canalización ($v_y \propto 1 - x^2$) para simular adecuadamente la dinámica del valle.
- **Fase Interpolativa**: Entrenamiento inicial asumiendo fuentes estáticas ($S=0$) para que la PINN pre-acondicione sus pesos aprendiendo la topología de la concentración antes del descubrimiento de parámetros.

### ⏳ Fase 3: Arquitectura Agéntica (CrewAI)
**Estado:** `En Planificación`
Desarrollo de un ecosistema de agentes LLM que disocian el razonamiento semántico del cómputo numérico:
- **Physics Architect:** Configura los límites del dominio y asimila los datos satelitales/terrestres.
- **Reaction Validator:** Verifica que las tasas de decaimiento termodinámicas sean científicamente plausibles.
- **Source Identification (Forense):** Cruza las coordenadas descubiertas por la PINN con bases de datos (OpenStreetMap) para atribuir responsabilidad industrial o de tráfico.

---

## ⚙️ Instalación y Uso

El proyecto opera bajo un ecosistema dual (Python para orquestación de datos/agentes, y Julia para cálculo de ecuaciones diferenciales).

### 1. Entorno de Python (Datos y Agentes)
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Entorno de Julia (PINN)
Asegúrese de tener Julia `1.10+` instalado.
```bash
julia init_julia.jl
```
*Nota: La primera instalación descargará y compilará el stack científico de SciML (NeuralPDE, Optimization), lo que puede tomar entre 5 y 10 minutos.*

### 3. Pruebas y Validación
Para verificar la integridad matemática y espacial de los datos:
```bash
python -m pytest tests/test_data_integrity.py -v
python src/geo/map_generator.py # Generará un mapa interactivo (mapa_validacion.html)
```

Para probar el pre-acondicionamiento interpolativo de la red neuronal:
```bash
julia src/pinn/train_interpolative.jl
```

---
*Desarrollado como proyecto de Aprendizaje Automático.*
