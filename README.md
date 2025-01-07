# Sage Realtime Health Surveillance

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![React](https://img.shields.io/badge/React-18-61DAFB) ![License](https://img.shields.io/badge/License-MIT-yellow)

Production-grade real-time health anomaly detection engine processing 50K events/sec with 200ms latency budget across 194 countries -- the healthcare analog of financial fraud detection

## Architecture

```
SAGE Data Lake (1.78B rows)     Processing Layer              Application Layer
+-------------------------+    +----------------------+    +-------------------+
| ERA5 Climate (362M)     |    | Feature Engineering  |    | REST API          |
| WHO Health (290M)       |--->| Model Training/Infer |--->| Real-time Stream  |
| IHME GBD (47M)          |    | Knowledge Graph      |    | Dashboard UI      |
| 268M Vector Embeddings  |    | Agent Orchestration  |    | Alert System      |
| 33M Causal KG Triples   |    | Evaluation Pipeline  |    | Report Generator  |
+-------------------------+    +----------------------+    +-------------------+
```

## Data Sources (SAGE Lake)

| Source | Records | Domain | Usage |
|--------|---------|--------|-------|
| WHO GHO | 190M+ | Health | Disease indicators across 194 countries |
| ERA5 CCD | 362M+ | Climate | Daily climate reanalysis (temperature, precipitation) |
| IHME GBD | 47M+ | Health | Global burden of disease estimates |
| World Bank WDI | 24M+ | Economics | Socioeconomic covariates |
| UNICEF | 8.8M+ | Child Health | Immunization, nutrition, WASH indicators |
| OpenDengue | 5.7M+ | Epidemiology | Dengue case counts across 129 countries |
| OECD Pharma | 19.6M+ | Pharmaceutical | Drug consumption, pricing, expenditure |
| CARD AMR | 271K | Genomics | Antimicrobial resistance genomes |
| WorldPop COGs | 200K+ | Geospatial | Population density rasters at 100m/1km |

## Quick Start

```bash
# Backend
pip install -r requirements.txt
python -m src.main

# Frontend (if applicable)
cd frontend && npm install && npm run dev
```

## License

MIT License -- Kaushik Sarkar 2025
