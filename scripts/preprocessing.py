import sys

# Ensure project root is on sys.path so we can import the top-level `data` package
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import DataProcessor

if __name__ == "__main__":
    processor = DataProcessor(data_dir="processed_data/raw")
    
    df = processor.prepare_modeling_dataset(
        start_date="2022-05-01",
        end_date="2025-10-31",
        include_lags=True,
        output_filename="dataset_v1_pogoh_weather.csv"
    )