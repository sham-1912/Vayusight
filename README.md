# AQI-Fusion: Multi-Source Explainable AQI Forecasting & Spatial Estimation System

AQI-Fusion combines ground station pollutant measurements (CPCB, OpenAQ, WAQI), meteorological drivers (OpenWeatherMap), satellite Aerosol Optical Depth (MODIS & Sentinel-5P via Google Earth Engine), SHAP explainability, and a Plotly Dash dashboard with a hyperlocal health-risk estimation layer (~1 km OpenStreetMap grid).

## Project Structure

```text
aqi_fusion/
├── config/
│   ├── config.yaml               # API keys, city bounding boxes, pollutants, collection parameters
│   └── schemas.py                # Pydantic v2 data models for spatio-temporal alignment
├── src/
│   ├── data_ingestion/
│   │   ├── openaq_client.py       # OpenAQ API v2 connector 
│   │   ├── waqi_client.py         # WAQI API connector 
│   │   ├── cpcb_client.py         # CPCB / data.gov.in connector 
│   │   ├── weather_client.py      # OpenWeatherMap connector
│   │   ├── kaggle_loader.py       # Historical benchmark CSV loader
│   │   └── gee_aod.py             # NASA Earthdata & GEE satellite AOD connector 
│   ├── data_preprocessing/
│   │   ├── ground_cleaning.py     # Outlier filtering, hourly resampling & KNN imputation 
│   │   └── satellite_preprocessing.py # Cloud masking, spatial alignment & IDW interpolation 
│   ├── pipeline/
│   │   └── run_etl.py             # Master ETL orchestrator 
│   └── utils/
│       └── geo_utils.py           # Bounding box filter, grid generator, Haversine dist 
├── tests/
│   ├── test_schemas.py            # Schema validation test suite
│   ├── test_geo_utils.py          # Geospatial utilities test suite
│   ├── test_ingestion.py          # Multi-source data ingestion test suite
│   └── test_preprocessing.py      # Preprocessing & Master ETL integration test suite
├── data/
│   ├── raw/                       # Staging directory for raw API JSON & CSVs
│   └── processed/                 # Fused Parquet and CSV datasets
├── requirements.txt               # Environment dependencies
└── README.md
```

## Setup & Testing Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run complete test suite (Weeks 1 through 5):
   ```bash
   pytest tests/
   ```
3. Execute master end-to-end ETL pipeline:
   ```bash
   python -m src.pipeline.run_etl
   ```

## Completed Phases & Deliverables (Weeks 1 to 5)
- **Phase 1 (Weeks 1–2)**: Data schemas (`config/schemas.py`), multi-source ingestion connectors (`src/data_ingestion/`), and geospatial utilities (`src/utils/geo_utils.py`).
- **Phase 2 (Weeks 3–5)**:
  - **Ground Data Cleaning (`ground_cleaning.py`)**: IQR outlier detection, hourly resampling, and KNN/MICE missing value imputation.
  - **Satellite Preprocessing (`satellite_preprocessing.py`)**: Quality flag filtering, cloud masking, nearest-neighbour spatial alignment, and spatial Inverse Distance Weighting (IDW) interpolation.
  - **Master ETL Orchestrator (`run_etl.py`)**: Unified extraction, cleaning, alignment, target AQI computation, and Parquet dataset export (`data/processed/fused_delhi_aqi.parquet`).

