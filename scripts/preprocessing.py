from data.preprocess import DataProcessor

if __name__ == "__main__":
    processor = DataProcessor(data_dir="processed_data/raw")
    
    df = processor.prepare_modeling_dataset(
        start_date="2020-01-01",
        end_date="2024-10-01",
        include_lags=True,
        output_filename="dataset_v1_pogoh_weather.csv"
    )