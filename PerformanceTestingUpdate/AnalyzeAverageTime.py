import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
import os

def select_csv_files():
    """Open file dialog to select multiple CSV files."""
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_paths = filedialog.askopenfilenames(
        title="Select CSV files to analyze",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_paths


def load_and_combine_csv_files(file_paths):
    """Load multiple CSV files and combine them into a single DataFrame."""
    if not file_paths:
        print("No files selected!")
        return None
    
    dataframes = []
    for file_path in file_paths:
        print(f"Loading: {file_path}")
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Remove any completely empty rows
    combined_df = combined_df.dropna(how='all')
    
    # Remove rows where critical columns are NaN
    initial_rows = len(combined_df)
    combined_df = combined_df.dropna(subset=['method_name', 'batch_size', 'time_seconds'])
    rows_removed = initial_rows - len(combined_df)
    
    if rows_removed > 0:
        print(f"\nRemoved {rows_removed} incomplete/empty rows")
    
    print(f"Total valid rows: {len(combined_df)}")
    print(f"Methods found: {sorted(combined_df['method_name'].unique())}")
    print(f"Batch sizes found: {sorted(combined_df['batch_size'].unique())}")
    
    return combined_df


def calculate_average_time_per_sample(df):
    """
    Calculate average time per sample for each method and batch size.
    
    The time_seconds column represents the time to generate batch_size samples.
    So, time per sample = time_seconds / batch_size
    """
    # Calculate time per sample for each row
    df['time_per_sample'] = df['time_seconds'] / df['batch_size']
    
    # Group by method and batch size, then calculate average
    results = df.groupby(['method_name', 'batch_size']).agg({
        'time_per_sample': ['mean', 'std', 'count'],
        'time_seconds': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    results.columns = [
        'method_name', 
        'batch_size', 
        'avg_time_per_sample_seconds',
        'std_time_per_sample_seconds',
        'num_measurements',
        'avg_total_time_seconds',
        'std_total_time_seconds'
    ]
    
    # Sort by method name and batch size
    results = results.sort_values(['method_name', 'batch_size'])
    
    return results


def save_results(results, output_filename='timing_analysis.csv'):
    """Save the analysis results to a CSV file."""
    output_path = os.path.join(os.getcwd(), output_filename)
    results.to_csv(output_path, index=False, float_format='%.6f')
    print(f"\nResults saved to: {output_path}")
    return output_path


def print_summary(results):
    """Print a formatted summary of the results."""
    print("\n" + "="*80)
    print("TIMING ANALYSIS SUMMARY")
    print("="*80)
    print("\nAverage Time Per Sample (seconds):")
    print("-"*80)
    
    # Set pandas display options for better formatting
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.6f}'.format)
    
    # Create a summary table
    summary = results[['method_name', 'batch_size', 'avg_time_per_sample_seconds', 
                       'std_time_per_sample_seconds', 'num_measurements']].copy()
    
    print(summary.to_string(index=False))
    
    print("\n" + "="*80)
    print("\nDetailed Statistics (Total Time per Batch):")
    print("-"*80)
    
    detailed = results[['method_name', 'batch_size', 'avg_total_time_seconds', 
                       'std_total_time_seconds']].copy()
    
    print(detailed.to_string(index=False))
    print("="*80)


def main():
    print("="*80)
    print("TIMING DATA ANALYSIS")
    print("="*80)
    print("\nPlease select the CSV files to analyze...")
    
    # Select files
    file_paths = select_csv_files()
    
    if not file_paths:
        print("No files selected. Exiting.")
        return
    
    # Load and combine data
    print(f"\nSelected {len(file_paths)} file(s)")
    combined_df = load_and_combine_csv_files(file_paths)
    
    if combined_df is None or combined_df.empty:
        print("No data to analyze. Exiting.")
        return
    
    # Calculate statistics
    print("\nCalculating average time per sample...")
    results = calculate_average_time_per_sample(combined_df)
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()