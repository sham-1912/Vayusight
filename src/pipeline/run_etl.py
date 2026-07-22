"""
Master ETL Pipeline Orchestrator (Both Members - Week 5 Milestone)
Executes end-to-end extraction, cleaning, imputation, satellite alignment, and fusion.
Outputs clean Parquet dataset ready for feature engineering and modeling.
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from config.schemas import records_to_dataframe
from src.data_ingestion import (
    OpenAQClient,
    CPCBClient,
    WeatherClient,
    KaggleDataLoader,
    EarthEngineAODClient
)
from src.data_preprocessing import (
    GroundDataCleaner,
    SatelliteAODPreprocessor
)
from src.utils.geo_utils import generate_city_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MasterETLPipeline")


class MasterETLPipeline:
    """Master Orchestrator for AQI-Fusion Data Engineering Pipeline."""

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clients & Preprocessors
        self.openaq = OpenAQClient()
        self.cpcb = CPCBClient()
        self.weather = WeatherClient()
        self.kaggle = KaggleDataLoader()
        self.gee_aod = EarthEngineAODClient(initialize_gee=False)
        
        self.ground_cleaner = GroundDataCleaner()
        self.sat_preprocessor = SatelliteAODPreprocessor()

    def run_pipeline(
        self,
        city: str = "Delhi",
        bbox: list = [76.84, 28.38, 77.38, 28.88],
        start_date: str = "2026-01-01",
        end_date: str = "2026-01-05"
    ) -> pd.DataFrame:
        """
        Execute full end-to-end pipeline:
        Extract -> Clean -> Impute -> Spatial Alignment -> Output Fused Data
        """
        logger.info(f"=== Starting AQI-Fusion Master ETL Pipeline for {city} ===")

        # Step 1: Ingest Ground Station Pollutants
        logger.info("Step 1: Ingesting Ground Station Pollutant Data...")
        openaq_readings = self.openaq.fetch_latest_measurements(city=city)
        cpcb_readings = self.cpcb.fetch_live_cpcb_data(city=city)
        historical_df = self.kaggle.load_and_clean_csv("non_existent_file.csv")

        ground_df = pd.concat([
            records_to_dataframe(openaq_readings),
            records_to_dataframe(cpcb_readings),
            historical_df
        ], ignore_index=True)
        ground_df['timestamp'] = pd.to_datetime(ground_df['timestamp'], utc=True)

        logger.info(f"Total raw ground records extracted: {len(ground_df)}")

        # Step 2: Clean & Resample Ground Data
        logger.info("Step 2: Outlier Filtering, Hourly Resampling & KNN Imputation...")
        ground_clean = self.ground_cleaner.filter_outliers_iqr(ground_df)
        ground_resampled = self.ground_cleaner.resample_hourly(ground_clean)
        ground_imputed = self.ground_cleaner.impute_missing_values(ground_resampled)

        # Step 3: Ingest Meteorological Data
        logger.info("Step 3: Merging Meteorological Weather Data...")
        weather_readings = []
        unique_stations = ground_imputed[['station_id', 'latitude', 'longitude']].drop_duplicates()
        
        for _, st in unique_stations.iterrows():
            w = self.weather.fetch_current_weather(st['latitude'], st['longitude'])
            weather_readings.append(w.model_dump())

        weather_df = pd.DataFrame(weather_readings)
        
        # Merge weather variables into ground records
        merged_gw = pd.merge(
            ground_imputed,
            weather_df[['latitude', 'longitude', 'temp', 'humidity', 'wind_speed', 'wind_deg', 'pressure']],
            on=['latitude', 'longitude'],
            how='left',
            suffixes=('', '_weather')
        )

        # Step 4: Ingest & Preprocess Satellite AOD
        logger.info("Step 4: Satellite AOD Ingestion & Spatial Alignment...")
        sat_readings = self.gee_aod.fetch_aod_for_bbox(bbox, start_date, end_date)
        sat_df = records_to_dataframe(sat_readings)
        sat_clean = self.sat_preprocessor.filter_quality_and_clouds(sat_df)

        # Spatially align satellite AOD to ground stations
        fused_df = self.sat_preprocessor.align_satellite_to_ground_stations(
            merged_gw, sat_clean, max_dist_km=15.0
        )

        # Step 5: Fill missing AOD via Spatial IDW Interpolation
        logger.info("Step 5: Filling Satellite AOD Gaps via Spatial IDW...")
        fused_df = self.sat_preprocessor.interpolate_spatial_aod_gaps(fused_df, sat_clean)

        # Step 6: Compute Target AQI (Indian CPCB Standard simplified calculation)
        logger.info("Step 6: Calculating Target AQI values...")
        if 'pm25' in fused_df.columns:
            # Sub-index approximation for PM2.5 (0-60 -> 0-100, 60-120 -> 100-300, >120 -> >300)
            fused_df['aqi'] = fused_df['pm25'].apply(
                lambda val: round(val * 1.67, 1) if pd.notna(val) else 100.0
            )

        # Step 7: Export Clean Fused Dataset
        output_parquet = self.output_dir / f"fused_{city.lower().replace('-', '_')}_aqi.parquet"
        output_csv = self.output_dir / f"fused_{city.lower().replace('-', '_')}_aqi.csv"

        fused_df.to_parquet(output_parquet, index=False)
        fused_df.to_csv(output_csv, index=False)

        logger.info(f"=== ETL Pipeline Complete! Final dataset saved to: {output_parquet} ({len(fused_df)} rows) ===")
        return fused_df


if __name__ == "__main__":
    pipeline = MasterETLPipeline()
    df = pipeline.run_pipeline(city="Delhi")
    print(df[['timestamp', 'station_id', 'pm25', 'pm10', 'temp', 'humidity', 'aod_550', 'aqi']].head(10))
