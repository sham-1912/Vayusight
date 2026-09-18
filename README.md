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
│   ├── eda/
│   │   └── exploratory_analysis.py # Statistical correlation, seasonal decomposition & spatial summary 
│   ├── features/
│   │   ├── time_features.py       # Temporal lags (t-1h..72h), rolling stats (6h/24h/7d), sin/cos time encoding 
│   │   ├── spatial_features.py    # Wind vector U/V components, satellite AOD calibration ratio & OSM land-use 
│   │   └── feature_pipeline.py    # Master feature transformer pipeline 
│   ├── pipeline/
│   │   └── run_etl.py             # Master ETL orchestrator 
│   └── utils/
│       └── geo_utils.py           # Bounding box filter, grid generator, Haversine dist 
├── tests/
│   ├── test_schemas.py            # Schema validation test suite
│   ├── test_geo_utils.py          # Geospatial utilities test suite
│   ├── test_ingestion.py          # Multi-source data ingestion test suite
│   ├── test_preprocessing.py      # Preprocessing & Master ETL integration test suite
│   └── test_features.py           # Feature engineering & EDA test suite
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
2. Run complete test suite:
   ```bash
   pytest tests/
   ```
3. Execute master end-to-end ETL & Feature Engineering pipeline:
   ```bash
   python -m src.pipeline.run_etl
   ```

## Completed Phases & Deliverables
- **Phase 1**: Data schemas (`config/schemas.py`), multi-source ingestion connectors (`src/data_ingestion/`), and geospatial utilities (`src/utils/geo_utils.py`).
- **Phase 2**: Ground data cleaning (`ground_cleaning.py`), satellite preprocessing (`satellite_preprocessing.py`), and Master ETL orchestrator (`run_etl.py`).
- **Phase 3**: Exploratory data analysis (`src/eda/`), temporal lag & rolling feature generation (`time_features.py`), wind vector decomposition & satellite calibration ratios (`spatial_features.py`), and master feature pipeline (`feature_pipeline.py`).


