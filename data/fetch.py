import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import io
import time
from typing import Optional


BASE_POGOH_URL = "https:://data.wprdc.org/api/3/action/package_show?id=pogoh-trip-data"


class POGOHDataFetcher:
    """
     A class to fetch and process POGOH data from WPRDC API.

     Example Usage
     -------------
     >>> fetcher = POGOHDataFetcher()
     >>> df = fetcher.fetch_data(
     ...     start_date="2025-07",
     ...     end_date="2025-09",
     ...     output_file_path=None,
     ...     delay=1.0  # Wait 1 second between downloads
     ... )
    >>> if not df.empty:
     ...     print("DATA SUMMARY")
     ...     print("=" * 60)
     ...     print(f"Total trips: {len(df):,}")
     ...     print(f"Date range: {df['source_month'].min()} to {df['source_month'].max()}")
     ...     print(f"\nColumns: {', '.join(df.columns.tolist())}")
     ...     print("\nFirst few rows:")
     ...     print(df.head())
    """

    def __init__(self):
        self.base_url = "https://data.wprdc.org"
        self.api_url = f"{self.base_url}/api/3/action"
        self.package_id = "pogoh-trip-data"

    def _list_pogoh_resources(self) -> list[dict]:
        """
        Fetch all monthly files for POGOH trip data.

        Returns
        -------
        list[dict]
            A list of dictionaries containing metadata for each file
        """
        url = f"{self.api_url}/package_show"
        params = {"id": self.package_id}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                return data["result"]["resources"]
            else:
                raise Exception(f"API returned error: {data.get('error')}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching resources: {e}")

    def _get_resource_date(self, resource: dict) -> Optional[datetime]:
        """
        Extracts the date from the resource name.

        Parameters
        ----------
        resource : dict
            A dictionary containing metadata for a file.

        Returns
        -------
        Optional[datetime]
            The extracted date or None if not found.
        """
        name = resource.get("name", "").lower()
        description = resource.get("description", "").lower()

        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        for month_name, month_num in months.items():
            if month_name in name or month_name in description:
                # Try to find year
                import re

                year_match = re.search(r"20\d{2}", name + description)
                if year_match:
                    year = int(year_match.group())
                    return datetime(year, month_num, 1)

        return None

    def _filter_resources_by_date(
        self, resources: list[dict], start_date: datetime, end_date: datetime
    ) -> list[dict]:
        """
        Filters resources based on a date range.

        Parameters
        ----------
        resources : list[dict]
            A list of dictionaries containing metadata for each file.
        start_date : datetime
            The start date for filtering.
        end_date : datetime
            The end date for filtering.
        Returns
        -------
        list[dict]
            A list of dictionaries containing metadata for files within the date range.
        """
        filtered = []
        for resource in resources:
            resource_date = self._get_resource_date(resource)

            if resource_date and start_date <= resource_date <= end_date:
                filtered.append({"resource": resource, "date": resource_date})

        filtered.sort(key=lambda x: x["date"])

        return filtered

    def _download_resource(self, resource: dict) -> pd.DataFrame:
        """
        Downloads a resource and loads it into a DataFrame.

        Parameters
        ----------
        resource : dict
            A dictionary containing metadata for a file.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the data from the resource.
        """
        url = resource.get("url")

        if not url:
            raise Exception(f"No URL found for resource: {resource.get('name')}")

        print(f"Downloading: {resource.get('name')} from {url}")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Determine file type from URL or content type
            if (
                url.endswith(".xlsx")
                or "excel" in response.headers.get("content-type", "").lower()
            ):
                df = pd.read_excel(io.BytesIO(response.content))
            elif (
                url.endswith(".csv")
                or "csv" in response.headers.get("content-type", "").lower()
            ):
                df = pd.read_csv(io.StringIO(response.text))
            else:
                try:
                    df = pd.read_excel(io.BytesIO(response.content))
                except:
                    df = pd.read_csv(io.StringIO(response.text))

            print(f"Successfully downloaded {len(df)} rows")
            return df

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading resource: {e}")

    def fetch_data(
        self,
        start_date: str,
        end_date: str,
        output_file_path: Optional[str] = None,
        delay: float = 1.0,
    ) -> pd.DataFrame:
        """
        Fetch POGOH data within a date range.

        Parameters
        ----------
        start_date : str
            The start date in 'YYYY-MM-DD' format.
        end_date : str
            The end date in 'YYYY-MM-DD' format.
        output_file_path : Optional[str], optional
            Path to save the combined data as CSV, by default None.
        delay : float, optional
            Delay between downloads in seconds, by default 1.0.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the combined data.
        """
        try:
            if len(start_date) == 7:  # YYYY-MM format
                start_dt = datetime.strptime(start_date, "%Y-%m")
            else:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")

            if len(end_date) == 7:  # YYYY-MM format
                end_dt = datetime.strptime(end_date, "%Y-%m")
            else:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD' or 'YYYY-MM': {e}")

        print(f"Fetching POGOH data from {start_date} to {end_date}")
        print("-" * 60)

        # Get all available resources
        print("Fetching available datasets...")
        all_resources = self._list_pogoh_resources()
        print(f"Found {len(all_resources)} total resources")

        # Filter by date range
        filtered = self._filter_resources_by_date(all_resources, start_dt, end_dt)
        print(f"Found {len(filtered)} resources within date range")
        print("-" * 60)

        if not filtered:
            print("No data found for the specified date range.")
            return pd.DataFrame()

        # Download and combine all resources
        all_data = []

        for item in filtered:
            resource = item["resource"]
            date = item["date"]

            try:
                df = self._download_resource(resource)

                # Add a column to track the source month
                df["source_month"] = date.strftime("%Y-%m")

                all_data.append(df)

                # Be respectful to the server
                time.sleep(delay)

            except Exception as e:
                print(f"Warning: Failed to download {resource.get('name')}: {e}")
                continue

        if not all_data:
            print("No data was successfully downloaded.")
            return pd.DataFrame()

        # Combine all dataframes
        print("-" * 60)
        print("Combining data...")
        combined_df = pd.concat(all_data, ignore_index=True)

        # Save to CSV
        if output_file_path is None:
            output_file_dir = Path.cwd() / f"processed_data/raw"
            output_file_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = (
                output_file_dir
                / f"pogoh_data_{start_dt.strftime('%Y%m')}_to_{end_dt.strftime('%Y%m')}.csv"
            )
        else:
            output_file_path = Path(output_file_path)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_file_path, index=False)
        print(f"Saved {len(combined_df):,} rows to {output_file_path}")

        return combined_df


fetcher = POGOHDataFetcher()
df = fetcher.fetch_data(
    start_date="2022-05",
    end_date="2025-09",
    output_file_path=None,
    delay=1.0,  # Wait 1 second between downloads
)

if not df.empty:
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total trips: {len(df):,}")
    print(f"Date range: {df['source_month'].min()} to {df['source_month'].max()}")
    print(f"\nColumns: {', '.join(df.columns.tolist())}")
    print("\nFirst few rows:")
    print(df.head())
