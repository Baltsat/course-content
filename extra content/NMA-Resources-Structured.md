# Neuromatch Academy - Структурированный обзор ресурсов

> Составлено на основе исследований сессий #S225 и #S226 (Январь 2026)

---

## 1. Официальная экосистема Neuromatch Academy

### Основные курсы

#### NeuromatchAcademy/course-content
- **Stars:** 2,948 | **Forks:** 1,100+
- **Обновлено:** Июль 2025 (активно поддерживается)
- **Содержимое:**
  - Полный курс вычислительной нейронауки (3 недели интенсива)
  - Jupyter notebooks по темам: линейная алгебра, статистика, модели нейронов, машинное обучение, динамические системы
  - Готовые датасеты для практики
  - Видео-лекции интегрированы
- **Актуальность:** ⭐⭐⭐⭐⭐ Основной ресурс, постоянно обновляется

#### NeuromatchAcademy/course-content-dl
- **Stars:** 804 | **Forks:** 283
- **Обновлено:** Июль 2025
- **Содержимое:**
  - Отдельный курс глубокого обучения для нейронауки
  - CNN для нейровизуализации
  - RNN для временных рядов нейронных данных
  - Reinforcement Learning
  - Современные архитектуры (трансформеры, foundation models)
- **Актуальность:** ⭐⭐⭐⭐⭐ Активный параллельный трек

#### NeuromatchAcademy/precourse
- **Stars:** 628
- **Обновлено:** Октябрь 2025
- **Содержимое:**
  - Подготовительные материалы перед курсом
  - Основы Python, NumPy, matplotlib
  - Базовая статистика
  - Введение в нейронауку
- **Актуальность:** ⭐⭐⭐⭐⭐ Критично для подготовки к курсу

### Инфраструктура и инструменты

| Репозиторий | Назначение | Обновлено |
|-------------|------------|-----------|
| `NeuromatchAcademy/nmaci` | CI/CD для автоматической проверки notebooks | Активен |
| `NeuromatchAcademy/widgets` | Интерактивные компоненты для notebooks | Январь 2026 |
| `NeuromatchAcademy/nma-bot` | Discord бот для сообщества | Активен |
| `NeuromatchAcademy/content-creation` | Инструменты для создания туториалов | - |
| `NeuromatchAcademy/TA-applications` | Набор TA (21 star) | Ноябрь 2025 |

### Материалы для преподавателей

| Репозиторий | Статус |
|-------------|--------|
| `instructor-course-content` | Публичный, jupyter-book формат |
| `instructor-course-content-dl` | Публичный |

> **Примечание:** Вопреки ожиданиям, инструкторские материалы с решениями публично доступны

---

## 2. Студенческие проекты по тематикам

### 2.1 Hidden Markov Models (HMM) — Анализ скрытых состояний

#### ivanorsolic/NeuromatchAcademy ⭐ TOP PICK
- **Stars:** 23 (второй по популярности среди студенческих)
- **Содержимое:**
  ```
  ├── Certificate.pdf           # Сертификат о прохождении
  ├── Detecting latent state sequences using HMMs.ipynb  # Основной проект
  ├── course-prep/              # Подготовительные материалы
  ├── notebooks/                # Туториалы с решениями
  └── images/                   # Визуализации
  ```
- **Темы notebooks:**
  - Modeling Practice: Framing the question
  - Optimal Control using LQG controller
  - Real Neurons: Synaptic transmission
- **Почему ценен:** Полный путь от подготовки до сертификата, продвинутые темы
- **Актуальность:** ⭐⭐⭐⭐ Классика, но датасеты могут устареть

#### NMA2022Chaosfan/HMM_SteinmetzDataset
- **Stars:** 2
- **Содержимое:**
  ```
  ├── NMA_hidden_states_PPT.pdf          # Презентация результатов
  ├── HMM_explore.ipynb                  # Исследовательский анализ
  ├── decoding_using_HMM.ipynb           # Декодирование
  ├── posterior.ipynb                    # Апостериорные вероятности
  ├── Spiking_decoder_and_cross_validation_in_Poisson_HMM.ipynb  # Продвинутая валидация
  ├── Areas_trials_analysis.ipynb        # Анализ по зонам мозга
  ├── compare_vis_motor.ipynb            # Визуальные vs моторные области
  ├── statistical_analysis.ipynb         # Статистика
  └── steinmetz_part[0-2].npz           # Партиционированный датасет
  ```
- **Почему ценен:** Профессиональный research workflow, Poisson HMM с кросс-валидацией
- **Актуальность:** ⭐⭐⭐⭐ 2022, Steinmetz датасет актуален

#### "Seals Evading Hidden States" (NMA 2020)
- **Stars:** 3
- **Содержимое:**
  ```
  ├── NMA_Project_Seals_Evading_Hidden_States.ipynb  # Основной notebook
  ├── loading_in_npy.ipynb / loading_in_tar.ipynb   # Загрузка данных
  ├── visual_task_data v5.ipynb                      # Визуальные задачи
  ├── spikes_for_region v2.ipynb                     # Спайки по регионам
  ├── ipsilateral_for_pca.txt / matrix_for_pca.txt  # PCA данные
  └── npy/                                           # Предобработанные данные
  ```
- **Почему ценен:** Показывает итеративный процесс (v2, v5), полный pipeline от raw до PCA
- **Актуальность:** ⭐⭐⭐ 2020, может требовать обновления зависимостей

---

### 2.2 Allen Institute Visual Behavior 2P — Кальциевый имейджинг

#### hosseindamavandi/NMA-Allen-Visual-Behavior-2p ⭐ TOP PICK
- **Stars:** 5
- **Обновлено:** Август 2025
- **Датасет:** Allen Visual Behavior 2-Photon (calcium imaging, visual cortex, change detection)
- **Почему ценен:** Самый свежий (2025 когорта), актуальный код
- **Актуальность:** ⭐⭐⭐⭐⭐ Свежий, Allen датасеты стабильны

#### farhadpanthaki/NMA_Allen
- **Год:** 2023
- **Датасет:** Allen Visual Behaviour 2p
- **Актуальность:** ⭐⭐⭐⭐ Относительно свежий

#### spicyHands/nma-allen
- **Год:** 2021
- **Актуальность:** ⭐⭐⭐ Может требовать обновления API

---

### 2.3 Human Connectome Project (HCP) — Мультимодальная нейровизуализация

#### seannoah/nma2020-HCP-project
- **Stars:** 4
- **Содержимое:**
  ```
  ├── GroupPresentationFinal.pdf    # Финальная презентация
  └── JupyterNotebooks/             # Анализ
  ```
- **Датасет:** HCP (structural MRI, fMRI, diffusion MRI, behavioral data)
- **Темы:** Функциональная связность, brain-behavior correlations
- **Актуальность:** ⭐⭐⭐ 2020, HCP API может измениться

#### vijpandaturtle/empathy-hcp-neuromatch
- **Тема:** Исследование эмпатии через HCP данные
- **Актуальность:** ⭐⭐⭐ Нишевый проект

---

### 2.4 Brain Networks & fMRI — Сетевой анализ мозга

#### tipsyfermion/NMA_Mapping_Brain_Networks_2020
- **Stars:** 5
- **Содержимое:**
  ```
  ├── Documentation.pdf     # Полное описание методологии
  ├── Presentation.pptx     # Презентация
  ├── Code/                 # Реализация
  └── Plots/                # Визуализации
  ```
- **Темы:** Graph theory, community detection, hub identification, network topology
- **Почему ценен:** Профессиональная документация, шаблон для курсовых проектов
- **Актуальность:** ⭐⭐⭐⭐ Методы graph theory вечнозеленые

#### Hazperera/Gender_Prediction_fMRI_data
- **Stars:** 5
- **Содержимое:**
  ```
  └── Gender_Prediction_from_BOLD_signals.py  # Единственный файл
  ```
- **Подход:** ML классификация (SVM/LogReg/NN) для предсказания пола по BOLD
- **Почему ценен:** Минималистичный, всё в одном скрипте — легко понять
- **Актуальность:** ⭐⭐⭐⭐ Простой и воспроизводимый

---

### 2.5 Steinmetz Dataset / Neuropixels — Электрофизиология

#### seungjaeryanlee/nma-cn-project (AwareWolfs team)
- **Stars:** 3
- **Год:** 2021
- **Содержимое:**
  ```
  ├── NMA-2021-AwareWolfs.pdf           # Финальная презентация
  ├── exploratory_data_analysis.ipynb   # EDA
  ├── single_cell_analysis.ipynb        # Анализ отдельных нейронов
  ├── population_analysis.ipynb         # Популяционная динамика
  ├── outdated_notebooks/               # История разработки
  ├── references/                       # Литература
  └── requirements.txt                  # Воспроизводимость
  ```
- **Почему ценен:** Системный подход (EDA → single cell → population), сохранена история
- **Актуальность:** ⭐⭐⭐⭐ Steinmetz — золотой стандарт

#### kayteeflick/nma_cn_flow_state
- **Stars:** 2 | **Год:** 2021
- **Тема:** Flow state analysis на Steinmetz 2019 данных
- **Актуальность:** ⭐⭐⭐ 2021

---

### 2.6 Attractor Networks & RNNs — Теоретическая нейронаука

#### wl17443/ring-attractor ⭐ TOP PICK
- **Stars:** 9
- **Язык:** Julia (высокая производительность)
- **Содержимое:**
  ```
  ├── src/                  # Исходный код
  ├── notebooks/            # Jupyter анализ
  ├── _research/            # Исследовательские заметки
  ├── data/                 # Обработанные данные
  ├── docs/                 # Документация
  ├── plots/                # Визуализации
  ├── scripts/              # Скрипты
  ├── Project.toml          # Julia зависимости
  └── Manifest.toml         # Lock файл
  ```
- **Тема:** Ring attractor networks для head direction systems, spatial navigation
- **Почему ценен:** Вырос из курсового в полноценный research tool, Julia для performance
- **Актуальность:** ⭐⭐⭐⭐⭐ Теоретические модели не устаревают

#### doomgutt/NMA-CN-2022-Project--Group-A---Attractor-Networks
- **Год:** 2022
- **Тема:** Attractor network models
- **Актуальность:** ⭐⭐⭐ 2022

#### martonaronvarga/nma_23_rnn
- **Год:** 2023
- **Тема:** Recurrent neural networks
- **Актуальность:** ⭐⭐⭐⭐ 2023, RNN концепции актуальны

---

### 2.7 Deep Learning проекты

#### sazio/NMAs
- **Stars:** 3
- **Тема:** Neural Manifold Animals — анализ нейронных многообразий с DL
- **Актуальность:** ⭐⭐⭐⭐ Современная тема

#### kushaldev75/NMA-Deep-Learning
- **Stars:** 2
- **Тема:** Transfer learning для классификации опухолей мозга по fMRI
- **Актуальность:** ⭐⭐⭐⭐ Медицинское применение

#### sarinaheshmati (проект)
- **Тема:** Foundation models для сравнения embeddings из EEG, изображений и объектов
- **Почему важен:** Cutting-edge тема с foundation models
- **Актуальность:** ⭐⭐⭐⭐⭐ Передовая тема

#### MartynasArba/NMA_DL_project
- **Год:** 2025
- **Актуальность:** ⭐⭐⭐⭐⭐ Самый свежий

---

### 2.8 Другие примечательные проекты

#### maryamdaniali/NeuromatchAcademy
- **Год:** 2020
- **Содержимое:**
  ```
  ├── Tutorials/            # Решённые туториалы (W0D1, W0D2, W1D1...)
  └── Project/              # Классификация поведения мышей по нейронной активности
  ```
- **Почему ценен:** Чёткое разделение учёбы и проекта
- **Актуальность:** ⭐⭐⭐ 2020

#### RR5555/SuperWM (Virtual Pugs team)
- **Stars:** 1
- **Тема:** Super working memory — исследование рабочей памяти
- **Актуальность:** ⭐⭐⭐ Нишевый

---

## 3. Curated Lists — Агрегаторы ресурсов

### Tier 1: Основные точки входа

#### analyticalmonk/awesome-neuroscience ⭐ MUST HAVE
- **Stars:** 1,535
- **Содержимое:** Курируемый список нейронаучного софта и библиотек
- **Для кого:** Всем, кто начинает в computational neuroscience
- **Актуальность:** ⭐⭐⭐⭐⭐ Постоянно обновляется

#### eselkin/awesome-computational-neuroscience ⭐ MUST HAVE
- **Stars:** 900
- **Содержимое:** Списки школ, исследователей, программ, лабораторий
- **Для кого:** Ищущим PhD/postdoc/программы обучения
- **Актуальность:** ⭐⭐⭐⭐⭐ Критичен для career planning

### Tier 2: Специализированные

#### CYHSM/awesome-neuro-ai-papers
- **Stars:** 428
- **Содержимое:** Статьи на пересечении deep learning и neuroscience
- **Для кого:** Исследователям neuro-AI
- **Актуальность:** ⭐⭐⭐⭐⭐ Горячая тема

#### open-neuromorphic/awesome-neuromorphic-hw
- **Stars:** 201
- **Содержимое:** Нейроморфное железо (ASICs, FPGAs, SNNs)
- **Для кого:** Hardware researchers, neuromorphic computing
- **Актуальность:** ⭐⭐⭐⭐⭐ Растущая область

#### Julie-Fabre/awesome_neuropixels
- **Stars:** 123
- **Содержимое:** Всё про Neuropixels — анализ, софт, pipelines
- **Для кого:** Экспериментаторам с Neuropixels
- **Актуальность:** ⭐⭐⭐⭐⭐ Neuropixels — стандарт индустрии

### Tier 3: Emerging

#### mazabou/awesome-neurofm
- **Stars:** 70
- **Содержимое:** Neuro-foundation models
- **Для кого:** Cutting-edge AI in neuroscience
- **Актуальность:** ⭐⭐⭐⭐⭐ Новейшая тема

#### sakimarquis/awesome-computational-neuroscience
- **Stars:** 46
- **Содержимое:**
  ```
  ├── README.md        # Курируемый список
  ├── Background/      # Фоновые материалы
  └── Blog/            # Образовательный контент
  ```
- **Почему ценен:** Не просто список, а образовательный портал
- **Актуальность:** ⭐⭐⭐⭐

---

## 4. Репозитории топ-контрибьюторов NMA

> Ключевые авторы курса имеют собственные образовательные репозитории

### ebatty (248 коммитов — #1 контрибьютор)

#### ebatty/MathToolsforNeuroscience ⭐ MUST HAVE
- **Stars:** 462
- **Содержимое:** Материалы курса Harvard Neurobio 212 — математические основы нейронауки
- **Темы:** Линейная алгебра, дифференциальные уравнения, вероятность, статистика для нейронауки
- **Почему ценен:** Более глубокая математика, чем в NMA — отличное дополнение
- **Актуальность:** ⭐⭐⭐⭐⭐

| Репозиторий | Описание |
|-------------|----------|
| `ebatty/IntroCompNeuro` | Вводный курс computational neuroscience |
| `ebatty/QuantitativeMethodsforBiologists` | Количественные методы для биологов |
| `ebatty/CodingBootcamp` | Буткемп по программированию |
| `ebatty/EncodingDecodingNotes` | Заметки по neural encoding/decoding |
| `ebatty/WorkingWithResearchCode` | Работа с исследовательским кодом |

### patrickmineault (107 коммитов)

#### patrickmineault/codebook ⭐ MUST HAVE
- **Stars:** 624
- **Содержимое:** "The Good Research Code Handbook" — руководство по написанию качественного исследовательского кода
- **Почему ценен:** Лучшие практики кодирования для учёных
- **Актуальность:** ⭐⭐⭐⭐⭐

| Репозиторий | Stars | Описание |
|-------------|-------|----------|
| `patrickmineault/research_code` | 117 | Дополнение к codebook |
| `patrickmineault/xcorr-notebooks` | 78 | Коллекция нейронаучных notebooks |
| `patrickmineault/big-neuro-ideas` | 34 | Журнальный клуб по нейронауке |
| `patrickmineault/fmn-tutorial` | 6 | Foundation models для нейронауки |
| `patrickmineault/neuromatch-nlp` | - | NLP материалы для Neuromatch |

### marius10p / Marius Pachitariu (87 коммитов)

#### marius10p/NeuralDataScienceCSHL2019
- **Stars:** 30
- **Содержимое:** Курс Neural Data Science в Cold Spring Harbor Laboratory
- **Темы:** Практический анализ нейронных данных, imaging pipelines
- **Почему ценен:** Автор Suite2p — знает calcium imaging изнутри
- **Актуальность:** ⭐⭐⭐⭐

| Репозиторий | Описание |
|-------------|----------|
| `marius10p/NeuralDynamics` | Нейронная динамика |
| `marius10p/donuts` | Детекция donut-shaped клеток в imaging |
| `marius10p/gui2P` | GUI для сортировки ROI |

### spirosChv (211 коммитов — #2 контрибьютор)

#### spirosChv/neuro208
- **Содержимое:** Собственный курс computational neuroscience
- **Почему ценен:** Альтернативный взгляд от второго по вкладу автора NMA

---

## 5. Альтернативные университетские курсы

> Другие публичные курсы computational neuroscience

| Репозиторий | Stars | Университет | Описание |
|-------------|-------|-------------|----------|
| `computational-neuroscience/Computational-Neuroscience-UW` | 148 | U. Washington | Дополнение к Coursera курсу |
| `conorhoughton/COMS30127` | 94 | Bristol | CS курс computational neuroscience |
| `kuz/Computational-Neuroscience-Course` | 25 | Tartu | Европейская перспектива |
| `arabs-in-neuro/intro_to_comp_neuro` | 24 | - | Введение для арабоязычного сообщества |
| `melizalab/comp-neurosci` | 16 | - | Problem sets по systems neuroscience |
| `CNS-OIST/ocnc` | 6 | OIST (Япония) | Азиатская перспектива |

---

## 6. Анализ форков — самые ценные варианты

### Форки с наибольшим переиспользованием

| Форк | Stars | Вторичные форки | Описание |
|------|-------|-----------------|----------|
| `erlichlab/course-content` | 21 | 14 | Самый популярный, snapshot лета 2020 |
| `slinderman/course-content` | 3 | 0 | Scott Linderman (Stanford), 2023 версия |
| `KordingLab/ENGR344` | 2 | 2 | Адаптирован в инженерный курс Data Science |
| `annkennedy/course-content` | 1 | 2 | Активное образовательное использование |

### Специальные версии

#### NeuromatchAcademy/chatify-course-content
- **Stars:** 1
- **Обновлено:** Июль 2025
- **Описание:** "Chatified" версия — интеграция с AI/чат-ботами
- **Статус:** Экспериментальный, пилотный проект
- **Актуальность:** ⭐⭐⭐ Интересен для AI-интеграции в образование

### Последние обновлённые форки (после июля 2025)

| Форк | Обновлено | Размер | Примечание |
|------|-----------|--------|------------|
| `ssushant0011/course-content` | Январь 2026 | 440 MB | Самый свежий |
| `ConnorSkye/course-content` | Декабрь 2025 | 447 MB | Возможно активная работа |
| `rinivarg/course-content` | Октябрь 2025 | 449 MB | - |

> **Примечание:** Эти форки меньше основного (5.6 GB), что может означать селективный контент или WIP

---

## 7. Дополнительная инфраструктура и ресурсы

### Orthogonal-Research-Lab/Neuromatch-Academy ⭐ TOP PICK
- **Stars:** 42 (самый популярный сторонний ресурс)
- **Содержимое:**
  ```
  ├── Complexity of Feedback Loops/    # Теория сложности
  ├── Epistemological Directory/       # Эпистемология
  ├── ML-DL Meta-science/              # Мета-наука ML/DL
  ├── Precourse/                       # Подготовительные материалы
  ├── Slow NMA Pod/                    # "Медленный" режим обучения
  └── Students/                        # Материалы студентов
  ```
- **Почему ценен:** Альтернативный темп обучения, теоретическая глубина, мета-научный подход
- **Актуальность:** ⭐⭐⭐⭐⭐ Уникальный ресурс

### Инструменты и инфраструктура

| Репозиторий | Назначение |
|-------------|------------|
| `XaqLab/neuromatch-academy-matching` | Алгоритм матчинга студентов в поды |
| `john-s-butler-dit/NMA-Concept-Map` | Визуализация структуры курса |

---

## 8. Дополнительные студенческие проекты

### Steinmetz Dataset — расширенный список

| Репозиторий | Описание | Примечание |
|-------------|----------|------------|
| `shyamven/SteinmetzNeuralNetworks` | Complex-valued neural networks | **AISTATS '25 paper!** |
| `bahramani/steinmetz-neural-prediction` | RNN для предсказания нейронной активности | NMA DL 2022 |
| `terse-o/Steinmetz-Neural-Dataset` | Влияние внимания на нейронную модуляцию | - |
| `shir-openu/steinmetz-neural-decision-signals` | Temporal dissociation в принятии решений | - |

### Проекты 2024-2025

| Репозиторий | Год | Тема |
|-------------|-----|------|
| `trumpeter122/NMA2025-fMRI-Spatial-Resolution-and-Neural-Decoding` | 2025 | fMRI spatial resolution |
| `ZohA-9/NMA-2024-PROJECT` | 2024 | Групповой проект (обновлён ноябрь 2024) |
| `cagrioguz/NMA-Computational-Neuroscience-2023-Group-Project` | 2023 | Trial novelty effect на face-selective areas |

### Другие примечательные проекты

| Репозиторий | Stars | Тема |
|-------------|-------|------|
| `zenkavi/neuromatch_project` | 5 | - |
| `jordanlei/neuromatch-project` | 5 | - |
| `saamehsanaaee/WMAD-Montbretia_Cabinet-ISP` | 2 | Working Memory Architecture and Demand |
| `rmgeddert/NMA_2020_SocialDecisionmaking_BayesianApproximationfMRI` | - | Social decision-making, Bayesian methods |

---

## 9. Утилиты и инструменты

| Репозиторий | Назначение | Для кого |
|-------------|------------|----------|
| `yakupcatalkaya/NMA_notebook_downloader` | Скачать все NMA notebooks локально | Офлайн-доступ |
| `mwaskom/nbci` | CI для тестирования notebooks | Контрибьюторы NMA |

---

## 10. Итоговый рейтинг: TOP-20 для изучения

### Tier S: Обязательные ресурсы

| # | Ресурс | Stars | Почему |
|---|--------|-------|--------|
| 1 | `NeuromatchAcademy/course-content` | 2,948 | Основной курс — база всего |
| 2 | `ebatty/MathToolsforNeuroscience` | 462 | Математика глубже, чем в NMA |
| 3 | `patrickmineault/codebook` | 624 | Как писать хороший research код |
| 4 | `analyticalmonk/awesome-neuroscience` | 1,535 | Главный агрегатор софта |

### Tier A: Для серьёзного изучения

| # | Ресурс | Stars | Почему |
|---|--------|-------|--------|
| 5 | `NeuromatchAcademy/precourse` | 628 | Подготовка к курсу |
| 6 | `NeuromatchAcademy/course-content-dl` | 804 | Deep learning трек |
| 7 | `eselkin/awesome-computational-neuroscience` | 900 | Карьера и программы |
| 8 | `Orthogonal-Research-Lab/Neuromatch-Academy` | 42 | "Slow pod" + теория |
| 9 | `computational-neuroscience/Computational-Neuroscience-UW` | 148 | Альтернативный курс |

### Tier B: Лучшие проекты-шаблоны

| # | Ресурс | Stars | Почему |
|---|--------|-------|--------|
| 10 | `ivanorsolic/NeuromatchAcademy` | 23 | Полный путь от prep до сертификата |
| 11 | `seungjaeryanlee/nma-cn-project` | 3 | Идеальный research workflow |
| 12 | `wl17443/ring-attractor` | 9 | Вырос в research tool (Julia) |
| 13 | `tipsyfermion/NMA_Mapping_Brain_Networks_2020` | 5 | Шаблон документации |

### Tier C: Cutting-edge и специализированные

| # | Ресурс | Stars | Почему |
|---|--------|-------|--------|
| 14 | `CYHSM/awesome-neuro-ai-papers` | 428 | Neuro-AI papers |
| 15 | `mazabou/awesome-neurofm` | 70 | Foundation models |
| 16 | `Julie-Fabre/awesome_neuropixels` | 123 | Neuropixels ecosystem |
| 17 | `shyamven/SteinmetzNeuralNetworks` | - | AISTATS '25 публикация |
| 18 | `hosseindamavandi/NMA-Allen-Visual-Behavior-2p` | 5 | Свежий 2025 проект |

### Tier D: Для преподавателей и исследователей

| # | Ресурс | Stars | Почему |
|---|--------|-------|--------|
| 19 | `marius10p/NeuralDataScienceCSHL2019` | 30 | CSHL курс от автора Suite2p |
| 20 | `erlichlab/course-content` | 21 | Самый переиспользуемый форк |

---

## 11. Ключевые датасеты (используются в проектах)

| Датасет | Тип данных | Проекты | Доступность |
|---------|------------|---------|-------------|
| **Steinmetz** | Neuropixels, поведение мышей | HMM, flow state, population analysis | Публичный |
| **Allen Visual Behavior 2P** | Calcium imaging, visual cortex | 3+ проекта (2021-2025) | Allen Brain Observatory |
| **Human Connectome Project** | MRI, fMRI, dMRI, поведение | HCP projects, empathy | Требует регистрации |

---

## 12. Рекомендации по использованию

### Если ты только начинаешь:
1. Пройди `precourse` материалы
2. Параллельно изучай `ebatty/MathToolsforNeuroscience` для математической базы
3. Изучи `course-content` (хотя бы первые 2 недели)
4. Читай `patrickmineault/codebook` для навыков кодирования
5. Посмотри `ivanorsolic/NeuromatchAcademy` как пример прохождения

### Если хочешь альтернативный темп:
1. Используй `Orthogonal-Research-Lab/Neuromatch-Academy` — "Slow NMA Pod"
2. Дополняй из `computational-neuroscience/Computational-Neuroscience-UW`

### Если ищешь проект для вдохновения:
1. Выбери датасет:
   - **Steinmetz** (электрофизиология) → `NMA2022Chaosfan/HMM_SteinmetzDataset`
   - **Allen** (calcium imaging) → `hosseindamavandi/NMA-Allen-Visual-Behavior-2p`
   - **HCP** (MRI) → `seannoah/nma2020-HCP-project`
2. Используй структуру `seungjaeryanlee/nma-cn-project` как шаблон
3. Для публикации смотри `shyamven/SteinmetzNeuralNetworks` (AISTATS '25)

### Если интересует AI/ML в нейронауке:
1. Начни с `course-content-dl`
2. Изучи `CYHSM/awesome-neuro-ai-papers`
3. Следи за `mazabou/awesome-neurofm` для foundation models
4. Пройди `patrickmineault/fmn-tutorial` для практики

### Если ищешь программу/PhD:
1. Изучи `eselkin/awesome-computational-neuroscience` — списки программ
2. Посмотри топ-контрибьюторов NMA (ebatty, spirosChv, mwaskom, patrickmineault) — потенциальные менторы
3. Изучи их личные репозитории для понимания их исследований

### Если хочешь преподавать:
1. Используй `erlichlab/course-content` как проверенную базу (14 форков)
2. Адаптируй как `KordingLab/ENGR344` для своего курса
3. Изучи `NeuromatchAcademy/instructor-course-content` для решений

---

## 13. Статистика исследования

| Метрика | Значение |
|---------|----------|
| Всего репозиториев проанализировано | 80+ |
| Официальных NMA репозиториев | 12 |
| Студенческих проектов | 30+ |
| Curated lists | 8 |
| Альтернативных курсов | 7 |
| Форков с вторичными форками | 15 |
| Топ-контрибьюторов изучено | 6 |

---

*Документ создан: Январь 2026*
*Источник: Сессии исследования #S225, #S226*
*Observations: #482-527 (46 записей)*
