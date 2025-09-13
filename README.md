# SigmaRx7

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SigmaRx7** is a comprehensive Python FHIR pipeline for synthetic healthcare data processing. It provides end-to-end ETL/ELT capabilities for medication management, including overlap detection, generic recommendations, payer rule alignment, and ML-based denial prediction.

## Features

- 🏥 **FHIR Data Ingestion**: Support for Synthea, ForgeRx, and standard FHIR bundles
- 💊 **RxNorm Normalization**: Automated medication normalization using RxNorm API
- 🔍 **Overlap Detection**: Identify therapeutic duplications and drug interactions
- 💰 **Generic Recommendations**: Cost-saving generic medication suggestions
- 📋 **Payer Rules Alignment**: Prior authorization, step therapy, and formulary compliance
- 🤖 **ML Denial Prediction**: Machine learning models to predict claim denials
- 📊 **Analytics Export**: FHIR bundles and feature tables for downstream analysis
- 🗄️ **SQL-Centric Architecture**: PostgreSQL-based data staging and processing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/pchj/sigma-rx7.git
cd sigma-rx7

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Configuration

```bash
# Initialize configuration file
sigma-rx7 init-config

# Edit config/sigma_rx7.yaml with your settings
```

### Database Setup

```bash
# Initialize database schema
sigma-rx7 init-db
```

### Run Pipeline

```bash
# Run complete pipeline with Synthea data
sigma-rx7 run -i data/sample_synthea_data.json -f synthea -o output/

# Run specific analysis
sigma-rx7 analyze overlaps
sigma-rx7 analyze generics
sigma-rx7 analyze payer-rules
sigma-rx7 analyze predict-denials

# Export results
sigma-rx7 export fhir -o output/fhir/
sigma-rx7 export features -o output/features/
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│   SigmaRx7       │───▶│    Outputs      │
│                 │    │   Pipeline       │    │                 │
│ • Synthea       │    │                  │    │ • FHIR Bundles  │
│ • ForgeRx       │    │ ┌──────────────┐ │    │ • Feature Tables│
│ • FHIR Bundles  │    │ │ Ingestion    │ │    │ • Analytics     │
└─────────────────┘    │ └──────────────┘ │    └─────────────────┘
                       │ ┌──────────────┐ │
                       │ │ Normalization│ │
                       │ └──────────────┘ │
                       │ ┌──────────────┐ │
                       │ │ Rules Engine │ │
                       │ └──────────────┘ │
                       │ ┌──────────────┐ │
                       │ │ ML Prediction│ │
                       │ └──────────────┘ │
                       └──────────────────┘
```

## Core Components

### 1. FHIR Ingestion Engine
- **Purpose**: Ingest and parse FHIR data from multiple sources
- **Formats**: Synthea, ForgeRx, standard FHIR bundles
- **Features**: Batch processing, validation, error handling

### 2. RxNorm Normalization
- **Purpose**: Standardize medication names using RxNorm
- **Features**: 
  - Exact and approximate matching
  - Spelling suggestions
  - Ingredient and strength extraction
  - Caching for performance

### 3. Overlap Detection
- **Purpose**: Identify medication overlaps and interactions
- **Features**:
  - Therapeutic duplication detection
  - Drug-drug interaction checking
  - Contraindication identification
  - Severity scoring

### 4. Generic Recommendations
- **Purpose**: Suggest cost-effective generic alternatives
- **Features**:
  - Therapeutic equivalence validation
  - Cost analysis and savings calculation
  - Formulary alignment
  - Patient copay optimization

### 5. Payer Rules Engine
- **Purpose**: Ensure compliance with payer requirements
- **Features**:
  - Prior authorization checking
  - Step therapy validation
  - Quantity limits enforcement
  - Age and diagnosis restrictions

### 6. ML Denial Predictor
- **Purpose**: Predict claim denial probability
- **Features**:
  - Random Forest and Gradient Boosting models
  - Feature engineering from clinical and administrative data
  - Risk categorization and recommendations
  - Model retraining capabilities

### 7. Export Engines
- **Purpose**: Export processed data for downstream use
- **Formats**:
  - FHIR bundles (JSON)
  - Feature tables (CSV, Parquet, JSON)
  - Metadata and data dictionaries

## Configuration

The pipeline is configured via YAML files. Key sections include:

```yaml
database:
  host: localhost
  port: 5432
  database: sigma_rx7
  username: postgres
  password: password

fhir:
  version: R4
  batch_size: 100

rxnorm:
  api_url: https://rxnav.nlm.nih.gov/REST
  rate_limit: 0.5

ml:
  enable_ml: true
  model_path: models/denial_predictor.joblib

pipeline:
  enable_overlap_detection: true
  enable_generic_recommendations: true
  parallel_workers: 4
```

## CLI Commands

### Core Pipeline
- `sigma-rx7 run` - Run complete pipeline
- `sigma-rx7 status` - Show pipeline status
- `sigma-rx7 init-db` - Initialize database

### Data Loading
- `sigma-rx7 load-synthea` - Load Synthea data
- `sigma-rx7 load-forgerx` - Load ForgeRx data

### Analysis
- `sigma-rx7 analyze overlaps` - Detect medication overlaps
- `sigma-rx7 analyze generics` - Generate generic recommendations
- `sigma-rx7 analyze payer-rules` - Check payer compliance
- `sigma-rx7 analyze predict-denials` - Predict claim denials

### Export
- `sigma-rx7 export fhir` - Export FHIR bundles
- `sigma-rx7 export features` - Export feature tables

## Data Sources

### Synthea
[Synthea](https://synthetichealth.github.io/synthea/) is an open-source synthetic patient generator that models the medical history of synthetic patients.

### ForgeRx
ForgeRx provides synthetic prescription data for testing and development purposes.

### Custom FHIR
Standard FHIR R4 bundles are supported for custom data sources.

## Output Formats

### FHIR Bundles
- Patient bundles with medications and observations
- Medication analysis bundles
- Analysis result bundles (overlaps, recommendations, etc.)

### Feature Tables
- **patient_features.csv**: Patient-level analytics
- **medication_features.csv**: Medication-level features
- **overlap_features.csv**: Overlap analysis results
- **recommendation_features.csv**: Generic recommendations
- **payer_rules_features.csv**: Payer compliance analysis
- **denial_prediction_features.csv**: ML predictions
- **cost_analysis_features.csv**: Cost analysis results

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Code formatting
black src/
isort src/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sigma_rx7

# Run specific test modules
pytest tests/test_pipeline.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For support and questions:
- Create an [issue](https://github.com/pchj/sigma-rx7/issues)
- Join our [discussions](https://github.com/pchj/sigma-rx7/discussions)

## Roadmap

- [ ] Real-time streaming data processing
- [ ] Advanced ML models for clinical outcomes
- [ ] Integration with EHR systems
- [ ] Cloud deployment templates
- [ ] Dashboard and visualization tools
- [ ] API endpoints for real-time queries
