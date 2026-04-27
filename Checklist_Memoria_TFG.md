# Checklist exhaustivo — Memoria del TFG

**Título:** *Development and evaluation of a multimodal deep learning system with explainability techniques for skin cancer stratification*
**Alumna:** Maialen Blanco Ibarra
**Modelo final:** E09 — Color Constancy + age + localization (seed=42) + TTA + umbral 0.31 + flag incertidumbre 0.70

---

## Cómo usar este checklist

- Organización **híbrida**: la **columna izquierda** sigue la estructura oficial de memoria PFG de Deusto (sección 4.1 de la especificación); la **columna derecha** mapea cada bloque a la fase CRISP-DM correspondiente y a los notebooks/archivos de trazabilidad técnica.
- Cada apartado tiene **casillas `[ ]`** para marcar lo que ya has cubierto.
- Las notas en *cursiva* son recordatorios de decisiones concretas que no debes olvidar justificar.
- El símbolo 🔥 marca puntos **diferenciadores** del TFG (aportaciones propias que elevan nota).
- El símbolo ⚠️ marca **limitaciones o errores corregidos** que conviene mencionar honestamente (demuestra madurez crítica).

---

# 📑 VISTA 1 — Estructura oficial de la memoria (PFG Deusto)

## 0. Páginas preliminares

- [ ] **Portada** con título, grado, tutora, fecha, curso académico.
- [ ] **Resumen** (abstract) — 150–250 palabras. Incluir problema, dataset, arquitectura, XAI, resultados clave (Melanoma Recall 0.91, ratio imagen/metadatos 140–200×), hallazgo del shortcut learning.
- [ ] **Resumen en inglés** (si lo requiere Deusto).
- [ ] **Descriptores / Keywords** — 3 a 5. Ejemplo: *multimodal deep learning, skin cancer classification, HAM10000, explainable AI, late fusion, Grad-CAM, SHAP*.
- [ ] **Índice general, índice de figuras, índice de tablas.**
- [ ] **Agradecimientos** (opcional).

---

## 1. Introducción

- [ ] Contextualización breve del problema: cáncer de piel como problema de salud pública.
- [ ] Motivación personal del proyecto.
- [ ] Qué se va a encontrar el lector en la memoria (párrafo-guía de la estructura).

---

## 2. Antecedentes y Justificación ⚠️ **GRAN HUECO actual**

> *Esta sección no existe aún en tus bullets. Es obligatoria y hay que construirla.*

### 2.1 Contexto clínico y epidemiológico

- [ ] Incidencia y mortalidad del cáncer de piel (global, Europa, España).
- [ ] Diferencia entre melanoma y otras lesiones (tasa de mortalidad del melanoma vs BCC).
- [ ] Importancia de la **detección temprana** (tasa de supervivencia a 5 años: >99% en melanoma in situ, <30% metastásico).
- [ ] La dermatoscopía como técnica estándar. Breve descripción.
- [ ] Criterios clínicos establecidos: **regla ABCDE**, **7-point checklist**, **patrón de Menzies**. *Importante porque Grad-CAM se evalúa implícitamente contra estos criterios.*
- [ ] Rol de los metadatos clínicos en el diagnóstico dermatológico (edad, localización como factores de riesgo).

### 2.2 Estado del arte técnico

#### 2.2.1 Deep learning en clasificación de lesiones cutáneas
- [ ] **Esteva et al. 2017** (Nature) — primer trabajo que iguala a dermatólogos en clasificación binaria.
- [ ] **Tschandl et al. 2018** — paper original de HAM10000.
- [ ] ISIC Challenges 2018/2019/2020 — resultados top.
- [ ] Arquitecturas habituales: ResNet, EfficientNet, ViT.

#### 2.2.2 Modelos multimodales clínicos
- [ ] **Kawahara et al.** — multimodal con metadatos sobre 7-point.
- [ ] **Pacheco & Krohling** — MetaBlock, integración de metadatos en CNN.
- [ ] **Yap et al.** — fusión de imagen dermatoscópica + clínica + metadatos.
- [ ] Justificación técnica de **late fusion** vs early fusion vs attention-based fusion.

#### 2.2.3 XAI en dermatología
- [ ] **Selvaraju et al. 2017** — Grad-CAM original.
- [ ] **Chattopadhyay et al. 2018** — Grad-CAM++.
- [ ] **Lundberg & Lee 2017** — SHAP original.
- [ ] **Smilkov et al. 2017** — SmoothGrad.
- [ ] Trabajos que aplican Grad-CAM a dermatoscopía (discusión de fiabilidad).

### 2.3 Justificación del proyecto

- [ ] **Gap identificado**: la mayoría de trabajos en HAM10000 son unimodales; los multimodales rara vez incluyen XAI sobre la rama de metadatos; los que lo hacen no diagnostican el *shortcut learning* cuantitativamente.
- [ ] **Aportaciones propias** (🔥 críticas para la nota):
  - Análisis cuantitativo de la contribución imagen vs metadatos (ratio 140–200×).
  - Diagnóstico del *shortcut learning por frecuencia* (ρ=0.96 vs n_total, ρ=0.37 vs prev_mel).
  - Integración TTA + ajuste de umbral con validación metodológicamente limpia.
  - Análisis multi-seed como prueba de estabilidad.
  - Análisis de concordancia entre métodos Grad-CAM / Grad-CAM++ / Vanilla / SmoothGrad.

---

## 3. Objetivos y Alcance ⚠️ **Falta en tus bullets**

### 3.1 Objetivo general
- [ ] Formulación explícita: *Desarrollar y evaluar un sistema de deep learning multimodal con técnicas XAI para la clasificación y estratificación de lesiones cutáneas sobre HAM10000.*

### 3.2 Objetivos específicos (medibles)
- [ ] OE1 — Seleccionar y justificar un pipeline de preprocesado óptimo entre 4 candidatos.
- [ ] OE2 — Determinar qué combinación de metadatos clínicos aporta mayor valor diagnóstico.
- [ ] OE3 — Alcanzar un Melanoma Recall ≥ 0.85 con ROC-AUC ≥ 0.95 manteniendo Macro F1 ≥ 0.75.
- [ ] OE4 — Implementar y comparar 4 métodos de explicabilidad visual.
- [ ] OE5 — Cuantificar la contribución diferencial de imagen y metadatos.
- [ ] OE6 — Diagnosticar posibles sesgos sistemáticos del modelo.
- [ ] OE7 — Desplegar el sistema en una interfaz interactiva (Streamlit) con mecanismo de rechazo por incertidumbre.

### 3.3 Alcance
- [ ] Lo que SÍ cubre: dataset HAM10000, 7 clases, imagen dermatoscópica + 3 metadatos clínicos, XAI con 5 técnicas, despliegue prototipo.
- [ ] Lo que NO cubre (delimitación): validación con dermatólogos reales, fototipos oscuros, validación externa en ISIC 2019/2020, segmentación previa, certificación sanitaria (marcado CE), integración con HIS hospitalario.
- [ ] Requisitos funcionales y no funcionales (apartado 5.1.1 de la especificación PFG).

---

## 4. Planificación ⚠️ **Falta en tus bullets**

- [ ] **Diagrama de Gantt** con fases: definición (Sept 2025) → data understanding → preparation → modelado → evaluación → XAI → despliegue → redacción → defensa (prevista Mayo/Jun 2026).
- [ ] Estimación de horas por fase.
- [ ] Hitos intermedios alcanzados:
  - Hito 1: EDA completado — Notebook 01.
  - Hito 2: Pipeline de preprocesado seleccionado (Fase 1 experimentos).
  - Hito 3: Modelo final elegido (E09 multi-seed).
  - Hito 4: XAI completo.
  - Hito 5: Streamlit desplegado.
- [ ] **Replanificaciones** (honestidad): menciona si algún hito se desplazó.
- [ ] Plan de recursos humanos: dedicación estimada del estudiante y tutora.

---

## 5. Presupuesto ⚠️ **Falta en tus bullets**

> *Obligatorio según apartado 7 de la especificación PFG.*

### 5.1 Costes laborales
- [ ] Horas estudiante × tarifa junior CDIA (referencia mercado ~25–40 €/h).
- [ ] Distribución por roles: data scientist, ML engineer, technical writer.
- [ ] Horas de tutorización.

### 5.2 Material y equipo
- [ ] Servidor con GPU — coste por amortización u hora de cómputo equivalente en cloud (estimación AWS g4dn.xlarge ≈ 0.52 $/h × horas de entrenamiento).
- [ ] Portátil del estudiante.
- [ ] Software: todo open source → coste 0.

### 5.3 Otros costes
- [ ] Licencias (Office, si aplica).
- [ ] Costes indirectos / generales.

### 5.4 Total estimado
- [ ] Tabla resumen con total.

---

## 6. Metodología

### 6.1 CRISP-DM — justificación de elección
- [ ] Breve descripción de CRISP-DM y sus 6 fases.
- [ ] Por qué CRISP-DM y no TDSP / SEMMA / DDME.
- [ ] **Mapeo explícito de cada notebook a fases CRISP-DM** (ver VISTA 2 abajo).

### 6.2 Metodologías complementarias
- [ ] Control de versiones con Git (si aplica — confirmar).
- [ ] Reproducibilidad: fijación de semilla (42), mixed precision, splits inmutables.
- [ ] Enfoque iterativo: primero imagen-only, luego multimodal, luego XAI, luego despliegue.

---

## 7. Desarrollo ← **Núcleo de la memoria, cubre ~60% de páginas**

> *Aquí es donde entran todos los notebooks 01–10. Estructura sugerida abajo.*

### 7.1 Entendimiento del negocio y de los datos

#### 7.1.1 Infraestructura y entorno ⚠️ **Falta en tus bullets**
- [ ] Servidor remoto `acroba-System-Product-Name`, IP 10.48.0.22.
- [ ] GPU CUDA, Python 3.8.10, PyTorch 2.4.1, entorno virtual `pfg-venv`.
- [ ] VS Code Remote SSH.
- [ ] Estructura de carpetas del proyecto (árbol completo: `data/`, `src/`, `notebooks/`, `experiments/`, `outputs/`, `app/`).
- [ ] Lista de dependencias principales con versiones (ya está en Contexto_después_xai).

#### 7.1.2 Dataset HAM10000 — Análisis Exploratorio (Notebook 01) ✅ **Ya en bullets**
- [ ] Origen Kaggle (kmader/skin-cancer-mnist-ham10000), 10015 imágenes.
- [ ] Las 7 clases: akiec, bcc, bkl, df, mel, nv, vasc.
- [ ] **Desbalanceo** (ratio nv:df = 58.3×) — figura donut + barras.
- [ ] Variable edad: media 51.9 ± 17.0, rango 0–85, 57 nulos (0.6%).
- [ ] Variable sexo: 5406 male / 4552 female / 57 unknown.
- [ ] Variable localización: 15 categorías, top back (2192), lower extremity (2077), trunk (1404).
- [ ] Boxplots por clase para edad y localización.
- [ ] Grid 3×7 de imágenes muestra por clase.

#### 7.1.3 ⚠️ Limitaciones y sesgos del dataset (**añadir a Notebook 01 o aquí**) 🔥
- [ ] **Sesgo demográfico**: HAM10000 proviene mayoritariamente de población europea (Austria, Australia) — fototipos I–II dominantes. Implicación: el modelo no generalizará a fototipos oscuros (IV–VI).
- [ ] **Desbalanceo por localización**: nv mayoritario en todas las zonas de alta frecuencia. Esto es lo que genera el *shortcut learning* detectado luego en el Notebook 09.
- [ ] **Split por imagen, no por paciente**: HAM10000 contiene `lesion_id`; el mismo paciente puede aparecer múltiples veces. *Hay que comprobar si `make_splits.py` usa stratify por clase pero ignora `lesion_id`.* Si es así, reconocer el posible leakage entre train/test.
- [ ] **Resolución variable** de imagen original; se redimensiona a 224×224.
- [ ] **Sesgo de "ground truth"**: no todas las etiquetas son histopatología confirmada (ver paper Tschandl — aprox. 53% son histopato).

### 7.2 Preparación de los datos

#### 7.2.1 Decisiones de preprocesado ⚠️ **Cubrir en detalle**
- [ ] **Splits** 70/15/15 (7009/1503/1503), estratificados por clase, seed=42 — fijos e inmutables.
- [ ] **Normalización de age**: `age / 90.0`. Justificar por qué 90 y no 100 ni MinMaxScaler.
- [ ] **Imputación de edad nula**: mediana 45.0 sobre 57 casos. Discutir alternativas (MICE, dejar nulos como dimensión extra).
- [ ] **One-hot encoding**:
  - `sex`: 3 dimensiones [male, female, unknown]. **Justificar inclusión de unknown como clase propia** en lugar de imputar.
  - `localization`: 15 dimensiones en orden alfabético exacto (listar el orden — crítico para cargar el modelo).
- [ ] **Redimensionado** a 224×224 (impuesto por EfficientNet-B0 preentrenado).
- [ ] **Normalización ImageNet** [mean=0.485,0.456,0.406 / std=0.229,0.224,0.225].

#### 7.2.2 Estudio comparativo de preprocesado de imagen (Notebook 02) ✅
- [ ] DullRazor: filtro blackhat + inpainting TELEA. Limitación: falla con pelo rubio.
- [ ] Color Constancy Shades-of-Gray (power=6.0).
- [ ] Modo "both".
- [ ] Comparativa visual 8×3 y 4×4.
- [ ] Benchmark de tiempos: CC 12.2 ms, DullRazor 64.7 ms, ambos 77.0 ms → ~7.5 min/época extra.
- [ ] Imágenes de CC pre-guardadas en `data/processed/colorconstancy/`.

#### 7.2.3 Data Augmentation
- [ ] Train: HFlip, VFlip, Rotation(20°), **ColorJitter(0.2, 0.2, 0.2)**, ToTensor, Normalize.
- [ ] Val/Test: solo Resize + Normalize.
- [ ] ⚠️ **Reflexión crítica**: *ColorJitter + Color Constancy aplicados simultáneamente* — ¿se solapan? ¿ColorJitter deshace parcialmente la normalización de CC? *Mencionar como limitación o decisión consciente.*

#### 7.2.4 Gestión del desbalanceo
- [ ] Weighted Cross Entropy con pesos inversos a frecuencia — cálculo explícito: `w_c = N_total / (num_classes * N_c)`.
- [ ] Justificar por qué **no oversampling** (overfitting) ni **focal loss**.

### 7.3 Modelado

#### 7.3.1 Arquitectura del modelo ⚠️ **Cubrir en profundidad**
- [ ] Backbone **EfficientNet-B0** (timm, num_classes=0, preentrenado ImageNet). Justificar elección: eficiencia en FLOPs, tamaño adecuado, rendimiento SOTA en medicina.
  - Alternativas consideradas y descartadas: ResNet50, ViT, DenseNet (justificar brevemente).
- [ ] **ImageBranch**: 1280 → Linear(256) + ReLU + Dropout(0.3) → 256 dims.
- [ ] **MetadataBranch**: 16 → Linear(64) + ReLU + Dropout(0.2) → Linear(64) + ReLU → 64 dims.
- [ ] **Fusion**: concat [256, 64] = 320 dims → Linear(128) + ReLU + Dropout(0.3) → Linear(7) = logits.
- [ ] Diagrama de bloques de la arquitectura (figura obligatoria).
- [ ] Vector de metadatos E09 — 16 dimensiones en orden exacto (listar).
- [ ] Justificar dimensiones elegidas (256, 64, 128): ¿hubo búsqueda de hiperparámetros o es elección inicial razonada?

#### 7.3.2 Configuración de entrenamiento
- [ ] Loss: CrossEntropyLoss con class_weights.
- [ ] Optimizador: Adam, lr=1e-4.
- [ ] Scheduler: ReduceLROnPlateau(mode='min', patience=3, factor=0.5).
- [ ] Early stopping: patience=7 sobre val_loss, guardar mejor epoch.
- [ ] Batch size 32, max epochs 30.
- [ ] **Mixed precision** (torch.amp autocast + GradScaler) — justificar (velocidad con GPU).
- [ ] Semilla fija 42.

#### 7.3.3 Diseño experimental — Fase 1 (Notebook 03) ✅
- [ ] Objetivo Fase 1: seleccionar preprocesado.
- [ ] E01–E04: baseline / DullRazor / ColorConstancy / ambos.
- [ ] Tabla de resultados con Macro F1, Macro Recall, ROC-AUC, Melanoma Recall.
- [ ] Decisión: **E03 (Color Constancy)** por mejor Macro F1 (0.770) y Macro Recall (0.830).
- [ ] Justificación clínica: CC reduce variabilidad de iluminación entre cámaras.
- [ ] ⚠️ **Reconocer trade-off**: E01 (baseline) tiene mejor Melanoma Recall (0.695) que E03 (0.665) — explicar por qué se prioriza rendimiento global y se aborda el Melanoma Recall con metadatos.

#### 7.3.4 Diseño experimental — Fase 2 (Notebook 03) ✅
- [ ] Objetivo Fase 2: ablation de metadatos con CC fijo.
- [ ] E05–E11: 7 combinaciones de age / sex / localization.
- [ ] Tabla completa de resultados con mAP y Brier Score añadidos.
- [ ] Candidatos finalistas: E06 (mayor Macro F1 0.816) y E09 (mayor Melanoma Recall 0.844).

#### 7.3.5 Análisis multi-seed — prueba de estabilidad 🔥
- [ ] Reentreno de E06 y E09 con seeds 42, 123, 7.
- [ ] Tabla multi-seed con media ± std.
- [ ] Hallazgo crítico: **E06 con seed 123 cae a F1=0.720** (peor que E03 baseline unimodal).
- [ ] E09 es **4× más estable** (std F1 0.013 vs 0.049).
- [ ] **Decisión final: E09.** Justificación clínica (edad y localización como factores de riesgo establecidos) + técnica (estabilidad).
- [ ] ⚠️ Justificar por qué multi-seed solo con E06 y E09 y no con todos (coste computacional).
- [ ] ⚠️ Justificar por qué multi-seed con 3 semillas y no 5 o 10 (balance estadística/coste).

### 7.4 Evaluación y discusión de resultados

#### 7.4.1 Selección y justificación de métricas ⚠️ **Cubrir como sección propia**
- [ ] **Macro F1** — por desbalanceo (average='macro' da peso igual a cada clase).
- [ ] **Macro Recall** — clínicamente prioritario.
- [ ] **Melanoma Recall** 🔥 métrica clínica crítica. Justificar: coste asimétrico FN vs FP en oncología.
- [ ] **ROC-AUC** — independiente del umbral, útil para ranking.
- [ ] **mAP** — más informativa que AUC en datasets desbalanceados.
- [ ] **Brier Score** — calibración probabilística, relevante para el umbral de rechazo.
- [ ] **Por clase**: Precision, Recall, F1, AUC para las 7 clases.
- [ ] Discusión: por qué **NO accuracy** (engañoso en desbalanceados).

#### 7.4.2 Análisis de resultados (Notebook 04) ✅
- [ ] Tabla ranking completa E01–E11.
- [ ] Gráfico comparativo Fase 1 (3 métricas).
- [ ] Ablation horizontal Fase 2.
- [ ] Scatter F1 vs Mel Recall.
- [ ] Radar chart E06 vs E09.
- [ ] Tabla multi-seed + gráfico de líneas con banda.
- [ ] Métricas por clase E09 — tabla y 2×2 de barras.
- [ ] Matriz de confusión E09 + matrices Fase 1.
- [ ] Hallazgos registrados:
  - 141/167 melanomas correctos.
  - 13 mel→nv (7.8%) — error clínico más grave.
  - 138 nv→mel (comportamiento conservador, Precision mel 0.453).
  - Error nv→mel sistemático en todos los preprocesados.
- [ ] ⚠️ Corrección del notebook 04: rutas de archivos con sufijo `__seed_42`, orden de seeds en gráfico multi-seed (error del `sort_values` alfabético).

#### 7.4.3 Evaluación extendida con curvas (Notebook 05) ✅
- [ ] Curvas ROC multiclase E09 vs E06.
- [ ] Curvas Precision-Recall con AP.
- [ ] Curvas de calibración (4 clases principales, 7 bins spline).
- [ ] Tabla resumen 6 métricas.
- [ ] **Bootstrap IC95%** con 1000 iteraciones — 🔥 aportación de rigor estadístico.
- [ ] Delta E09−E06 con solapamiento de intervalos.
- [ ] Comparativa contra baseline unimodal E03: **+0.180 en Mel Recall** (+27% relativo).
- [ ] Conclusión: mel es la única clase donde E09 supera a E06 en ROC-AUC y AP.

#### 7.4.4 Test Time Augmentation (Notebook 06) ✅
- [ ] Justificación conceptual de TTA.
- [ ] 6 transformaciones geométricas deterministas (listar).
- [ ] Decisión: **excluir transformaciones de color** porque CC ya normaliza iluminación.
- [ ] Tabla E09 baseline vs E09+TTA con todos los deltas.
- [ ] Mejora universal de métricas, Mel Recall +0.0180.
- [ ] Trade-off en F1 de mel (−0.009) justificado clínicamente.
- [ ] ~3 melanomas adicionales detectados por cada 167.

#### 7.4.5 Ajuste de umbral de melanoma (Notebook 07) ✅ ⚠️ **con corrección importante**
- [ ] Motivación: asimetría de coste FN/FP en oncología.
- [ ] ⚠️ **Error metodológico original y corrección** 🔥 *(excelente anécdota de rigor crítico)*:
  - Versión inicial: barrido de umbral directamente sobre test → **data leakage**.
  - Corrección: sweep sobre validación → umbral 0.31 → evaluación única sobre test.
  - Resultado corregido: Mel Recall 0.9102 (ligeramente superior al 0.9042 del test-sweep), confirmando que el problema no inflaba mucho pero era metodológicamente incorrecto.
- [ ] Gráfico de curva threshold vs métricas con líneas de referencia.
- [ ] Configuración de despliegue final (tabla): E09 + CC + TTA(6) + thr=0.31 + flag <0.70.

#### 7.4.6 Explicabilidad visual (Notebook 08) ✅
- [ ] Implementación de 4 métodos: Grad-CAM, Grad-CAM++, Vanilla Saliency, SmoothGrad.
- [ ] ⚠️ **Decisión metodológica**: XAI usa un único forward pass (sin TTA) porque los gradientes no son compatibles con promediado.
- [ ] Selección de casos usando probs TTA (config de producción).
- [ ] 12 casos en 4 categorías (correct_melanoma, critical_error, high_uncertainty, correct_non_mel).
- [ ] Figuras 1×5 por caso.
- [ ] **Análisis de concordancia entre métodos** 🔥 — matriz Pearson:
  - Grad-CAM vs Grad-CAM++ r=0.997.
  - SmoothGrad vs Grad-CAM r≈−0.19 — hallazgo contraintuitivo, interpretado como "Grad-CAM mira el núcleo, SmoothGrad mira el borde".
  - Correlación por tipo de caso (high_uncertainty máxima: 0.327).

#### 7.4.7 Explicabilidad SHAP — metadatos (Notebook 09) ✅ **INCOMPLETO en tus bullets**
- [ ] Justificación del SHAP KernelExplainer sobre MetadataBranch con image features fijas.
- [ ] Background de 100 muestras aleatorias del train.
- [ ] SHAP caso por caso (12 casos) — tabla de hallazgos.
- [ ] Escala SHAP diferencial: casos fáciles 1e-6 vs inciertos 1e-3 (3 órdenes de magnitud).
- [ ] Figuras combinadas (imagen + Grad-CAM + SHAP) — las principales de la memoria.
- [ ] Análisis agregado sobre 167 melanomas del test.
  - Importancia media |SHAP| por feature.
  - Ranking: loc:back > loc:upper_extremity ≈ loc:trunk > age.
  - Direcciones con signo (violin plots top 6).
  - Hallazgo: `loc:back` tiene importancia 2.07× mayor en melanomas mal clasificados.
- [ ] ⚠️ **AÑADIR — Sección 4.1 del Additions_17_04 (falta en tus bullets)** 🔥:
  - **Contribución imagen vs metadatos cuantificada**.
  - Método A (ablación marginal) + Método B (Shapley por bloques).
  - Ratio imagen/metadatos 140–200×.
  - Interpretación: el sistema es fundamentalmente visual (99.3–99.5%).
- [ ] ⚠️ **AÑADIR — Sección 4.3 del Additions_17_04 (falta en tus bullets)** 🔥 **"dato bomba"**:
  - **Shortcut learning por frecuencia**.
  - Correlaciones Spearman: ρ=+0.96 vs n_total, ρ=+0.91 vs n_melanomas, ρ=+0.37 vs prev_mel.
  - Conclusión: el modelo aprende "cuántas veces vio cada localización", no "cómo de informativa es".
  - **Caso del `loc:ear`**: 31.7% prevalencia real (la más alta), SHAP casi cero porque solo 41 casos en train.
- [ ] Sección 7 — Hipótesis del desempate: Pearson r=−0.268 (p=4.54e-4) entre p(mel) y magnitud SHAP.
- [ ] Sección 8 — Calibración vs magnitud SHAP: Brier empeora 35% cuando metadatos intervienen.
- [ ] Sección 9 — Replicación del análisis para bcc y akiec: patrón **sistémico** (no específico de melanoma).

#### 7.4.8 Síntesis XAI — correctos vs incorrectos (Notebook 10) ✅
- [ ] Pregunta central: *¿cuándo falla, el modelo no ve la lesión o la malinterpreta por priors de metadatos?*
- [ ] Decisión metodológica: usar 12 fallos post-TTA (no los 23 baseline) porque TTA es la config de producción.
- [ ] 4 correctos vs 4 incorrectos — layout 2×4 por grupo.
- [ ] **Hallazgo integrador**: Grad-CAM bien centrado en TODOS los casos → el fallo no es visual.
- [ ] **Escala SHAP como discriminador**:
  - Correctos: 0 – 8.66e-5.
  - Incorrectos: 2.48e-5 – 7.37e-4.
  - Ratio 10–100×.
- [ ] **MetadataBranch como doble filo**:
  - Refuerzo correcto (ISIC_0029212: `loc:lower_extremity=1` + `age=75` empuja hacia melanoma correctamente).
  - Fuente de fallos (ISIC_0033444: `loc:abdomen=1` empuja contra melanoma).
- [ ] Caso `ISIC_0029271`: única inversión del patrón (metadatos apoyan mel, imagen branch anula).
- [ ] Puente narrativo entre notebooks 08 / 09 / 10.
- [ ] **Implicación clínica**: el sistema es más peligroso cuando coinciden imagen borderline + prior de benignidad fuerte → el flag de incertidumbre puede no activarse.

### 7.5 Despliegue — Streamlit ⚠️ **COMPLETO HUECO en tus bullets** 🔥

- [ ] **Arquitectura de la aplicación**:
  - Carga de E09 weights.
  - Pipeline de inferencia: CC preprocesado on-the-fly → ImageBranch + MetadataBranch → TTA(6) → umbral 0.31 → clasificación → flag <0.70.
- [ ] **Diseño de UI**:
  - Upload de imagen dermatoscópica.
  - Inputs de age (slider 0–85) y localización (dropdown 15 opciones).
  - Output: gráfico de barras con probabilidades por clase.
  - Overlay Grad-CAM sobre imagen.
  - Barras horizontales SHAP de contribución de metadatos.
  - Aviso de incertidumbre (confianza <70% → "derivar a especialista").
- [ ] **Aviso especial para localizaciones raras de alto riesgo** (propuesta del Additions_17_04): lookup contra tabla `prev_mel`; si `prev_mel>15%` y `n_total<100`, flag visual rojo "atención: localización de alto riesgo epidemiológico poco representada en el entrenamiento".
- [ ] Decisiones de UX (paleta de colores, jerarquía, tipografía, accesibilidad).
- [ ] Capturas de pantalla con ejemplos:
  - Caso correcto con alta confianza (correct_melanoma).
  - Caso de alta incertidumbre.
  - Caso de localización rara.
- [ ] Limitaciones del prototipo: monousuario, sin base de datos, sin autenticación, sin integración HIS.

---

## 8. Valoración Ética del Proyecto ⚠️ **OBLIGATORIO, falta completo**

### 8.1 Bias audit — análisis de sesgos del modelo 🔥 **Aprovechar shortcut learning**
- [ ] **Sesgo de localización** — ya documentado en Notebook 09 (shortcut).
- [ ] **Sesgo por grupo demográfico** — *pendiente de analizar si no se ha hecho*:
  - Rendimiento por sexo (male vs female vs unknown).
  - Rendimiento por rango de edad (<40, 40–60, >60).
  - Ejecutar metrics por grupo sobre el test set y reportar deltas.
- [ ] **Sesgo de fototipo**: HAM10000 mayoritariamente piel clara europea. **Limitación crítica para despliegue en poblaciones con fototipos IV–VI**.
- [ ] Propuesta de mitigación: entrenamiento futuro con ISIC 2020 + Fitzpatrick 17k.

### 8.2 Implicaciones éticas del despliegue
- [ ] **Responsabilidad médica**: el sistema es *apoyo a la decisión*, no diagnóstico autónomo. Quién asume la responsabilidad de un FN.
- [ ] **RGPD**: manejo de imágenes y metadatos de pacientes (anonimización, consentimiento informado, derecho al olvido).
- [ ] **Marcado CE sanitario** (Regulation EU 2017/745, MDR) — el sistema sería dispositivo médico clase IIa/IIb; discutir implicación para despliegue real.
- [ ] **AI Act** de la UE (vigor 2024) — los sistemas médicos son de "alto riesgo", requisitos de transparencia, supervisión humana, documentación técnica.
- [ ] **Riesgo de automatización**: riesgo de que el clínico pase a confiar ciegamente en la IA ("automation bias").
- [ ] **Explicabilidad como requisito ético**, no solo técnico (justificación extra de Grad-CAM y SHAP).

### 8.3 Principios de IA responsable
- [ ] Mapeo del proyecto contra los 7 principios éticos de la CE para IA confiable: transparencia, supervisión humana, robustez, privacidad, no discriminación, bienestar social, accountability.

---

## 9. Incidencias ⚠️ **Falta en tus bullets, muy buen apartado para honestidad**

> *Lista honesta de problemas encontrados y cómo se resolvieron — demuestra madurez.*

- [ ] **Bug del SmoothGrad** en `src/xai.py` — device mismatch entre CPU y CUDA (resuelto con `.detach().cpu()`).
- [ ] **Bug del overlay_heatmap** — valores float fuera de rango rompiendo OpenCV (resuelto con `np.clip(0, 1)`).
- [ ] **Bug de shape en SmoothGrad** — faltaba `max(dim=0)` para colapsar canales.
- [ ] **Error de data leakage en umbral** — notebook 07 original sweepeaba en test; corregido a sweep en val.
- [ ] **Error de rutas con/sin `__seed_42`** en notebook 04 — matriz de confusión y pesos.
- [ ] **Error de orden de seeds** en gráfico multi-seed — `sort_values` alfabético ordenaba ['seed 123','seed 42','seed 7'].
- [ ] **IProgress warning** de SHAP — inofensivo, documentado.
- [ ] **JSON serialization error** — numpy.int64 no serializable, resuelto con función `to_serializable()`.
- [ ] **ModuleNotFoundError src/** — añadida celda 0 de setup a todos los notebooks.
- [ ] **Decisión arrastrada**: inicialmente se eligió E11 como modelo final; tras multi-seed se cambió a E09.

---

## 10. Conclusiones y Líneas Futuras

### 10.1 Conclusiones generales
- [ ] Objetivo general alcanzado (revisar OE1–OE7 del apartado 3.2).
- [ ] Cumplimiento de métricas objetivo.
- [ ] **Cuatro hallazgos clave** (síntesis narrativa del TFG — sacados del Additions_17_04):
  1. El sistema es fundamentalmente visual (ratio 140–200× imagen vs metadatos).
  2. Los metadatos actúan como desempate en casos borderline (r=−0.268 p<0.001).
  3. El desempate está sesgado hacia la clase mayoritaria por localización.
  4. **Causa raíz: shortcut learning por frecuencia** (ρ=0.96 vs n_total vs ρ=0.37 vs prev_mel).
- [ ] Valor añadido respecto al estado del arte.

### 10.2 Limitaciones del sistema
- [ ] Dataset único HAM10000 — no validación externa.
- [ ] Split posiblemente por imagen y no por paciente — verificar.
- [ ] Sin validación con dermatólogos reales.
- [ ] Solo 7 clases.
- [ ] Shortcut learning documentado pero no mitigado en el modelo final.
- [ ] Sin segmentación previa de la lesión (modelo puede atender a píxeles de piel sana).
- [ ] Sin K-Fold Cross Validation (multi-seed es alternativa parcial).
- [ ] Solo fototipos claros.
- [ ] No se testeó en imágenes dermatoscópicas reales de distintos dispositivos.

### 10.3 Líneas futuras (priorizadas por coste/beneficio del Additions_17_04)
- [ ] **Corto plazo / bajo coste**:
  - Post-hoc calibration por localización usando `prev_mel` como prior bayesiano (sin reentrenamiento).
  - Flag de alto riesgo en UI para localizaciones `prev_mel>15%` y `n_total<100`.
- [ ] **Medio plazo / coste medio**:
  - Augmentation dirigida a localizaciones raras de alto riesgo (ear, face).
  - Segmentación con U-Net previa al clasificador.
- [ ] **Largo plazo / alto coste**:
  - Location-stratified class balancing durante entrenamiento.
  - Validación externa en ISIC 2019/2020 y PH2.
  - K-Fold Cross Validation completa.
  - Ampliación con Fitzpatrick 17k para cubrir fototipos oscuros.
  - Evaluación comparativa contra panel de dermatólogos.
  - Arquitecturas alternativas: ViT, Swin Transformer, attention-based fusion.
  - Integración clínica real (pruebas piloto hospitalarias).

---

## 11. Bibliografía

- [ ] Formato consistente (APA o IEEE — Deusto suele pedir IEEE).
- [ ] **Papers obligatorios**:
  - Tschandl et al. 2018 — HAM10000 dataset paper.
  - Esteva et al. 2017 — Nature paper clasificación dermatológica.
  - Tan & Le 2019 — EfficientNet.
  - Selvaraju et al. 2017 — Grad-CAM.
  - Chattopadhyay et al. 2018 — Grad-CAM++.
  - Smilkov et al. 2017 — SmoothGrad.
  - Lundberg & Lee 2017 — SHAP (NeurIPS).
  - Kawahara et al. 2019 — multimodal 7-point.
  - Pacheco & Krohling 2021 — MetaBlock.
  - Finlayson et al. — Shades-of-Gray Color Constancy.
  - Lee & Chen 1997 — DullRazor original.
- [ ] **Documentación técnica**: PyTorch, timm, pytorch-grad-cam, SHAP, Streamlit.
- [ ] Referencias clínicas: guías AEDV/EADV, incidencia AECC/IARC.
- [ ] AI Act UE 2024, MDR 2017/745.

---

## 12. Definiciones, Acrónimos y Abreviaturas

- [ ] HAM10000, CNN, DL, ML, IA/AI.
- [ ] TTA, XAI, SHAP, Grad-CAM, CAM.
- [ ] ROC, AUC, AP, mAP, F1, Brier Score.
- [ ] nv, mel, bkl, bcc, akiec, vasc, df (con nombre completo).
- [ ] CRISP-DM, MLP, ReLU, Dropout, Softmax, Cross-Entropy.
- [ ] FN, FP, TN, TP.
- [ ] ABCDE (regla dermatológica).
- [ ] GDPR/RGPD, MDR, CE, AI Act.

---

## 13. Apéndices

- [ ] **Apéndice A — Código fuente** (extractos de src/ más relevantes).
- [ ] **Apéndice B — Tablas extendidas** (métricas por clase, multi-seed, bootstrap completo).
- [ ] **Apéndice C — Figuras suplementarias** (matrices confusión de todos los experimentos, todos los heatmaps XAI).
- [ ] **Apéndice D — Manual de usuario Streamlit**.
- [ ] **Apéndice E — Guía de reproducibilidad** (cómo montar el entorno, lanzar experimentos, ejecutar notebooks).

---

# 🔄 VISTA 2 — Mapeo CRISP-DM ↔ Notebooks ↔ Secciones de memoria

| Fase CRISP-DM | Notebooks | Sección memoria | Estado |
|---|---|---|---|
| **Business understanding** | — | §1 Intro, §2 Antecedentes, §3 Objetivos | ⚠️ Falta |
| **Data understanding** | NB01 EDA | §7.1 | ✅ Ok |
| **Data preparation** | NB02 Preprocessing | §7.2 | ✅ Ok |
| **Modeling** | NB03 Experiments | §7.3 | ✅ Ok |
| **Evaluation** | NB04 Results, NB05 Extended, NB06 TTA, NB07 Threshold, NB08 XAI Visual, NB09 XAI SHAP, NB10 XAI Synthesis | §7.4 | ✅ Ok (⚠️ añadir Additions_17_04 al NB09) |
| **Deployment** | Streamlit app | §7.5 | ⚠️ Falta |

### Observaciones sobre el mapeo
- [ ] Explicar por qué los notebooks 05–10 están todos bajo "Evaluation" y no se han dividido en sub-fases.
- [ ] Reconocer que CRISP-DM es iterativo: se volvió a data preparation (Color Constancy fijado) tras la Fase 1, y luego a modeling para Fase 2.
- [ ] Iteraciones del proyecto: E01–E04 (iter 1) → E05–E11 (iter 2) → multi-seed (iter 3) → TTA + umbral (iter 4) → XAI (iter 5) → Streamlit (iter 6).

---

# 📌 Resumen de huecos críticos vs tus bullets originales

| Hueco | Dónde iba | Prioridad |
|---|---|---|
| Antecedentes clínicos + estado del arte | §2 | 🔴 Alta (obligatorio) |
| Objetivos medibles y alcance | §3 | 🔴 Alta (obligatorio) |
| Planificación + Gantt | §4 | 🔴 Alta (obligatorio) |
| Presupuesto | §5 | 🔴 Alta (obligatorio) |
| Infraestructura y entorno | §7.1.1 | 🟡 Media |
| Justificación arquitectura y dimensiones | §7.3.1 | 🟡 Media |
| Limitaciones dataset HAM10000 | §7.1.3 | 🟡 Media |
| ColorJitter + CC reflexión | §7.2.3 | 🟢 Baja (matiz) |
| **Sección 4.1 Additions_17_04 (imagen vs metadatos 140–200×)** | §7.4.7 | 🔴 Alta 🔥 |
| **Sección 4.3 Additions_17_04 (shortcut ρ=0.96)** | §7.4.7 | 🔴 Alta 🔥 |
| Streamlit completo | §7.5 | 🔴 Alta (pendiente por implementar) |
| Valoración ética + bias audit | §8 | 🔴 Alta (obligatorio) |
| Incidencias | §9 | 🟡 Media |
| Líneas futuras priorizadas | §10.3 | 🟡 Media |
| Bibliografía formal | §11 | 🔴 Alta (obligatorio) |
| Acrónimos/Glosario | §12 | 🟢 Baja |

---

# ✅ Siguiente paso recomendado

1. [ ] Marcar en este checklist lo que **ya tienes escrito** vs lo **pendiente**.
2. [ ] Priorizar los bloques 🔴 que aún no tienes.
3. [ ] Decidir si el Streamlit se implementa antes o en paralelo a la redacción de los huecos.
4. [ ] Reservar al menos una semana para bibliografía + glosario + apéndices al final (se subestiman sistemáticamente).

---

*Documento generado como acompañante al documento de bullets por notebook. Úsalos en paralelo: este marca qué cubrir en la memoria; el otro detalla qué se ha hecho en el código.*
