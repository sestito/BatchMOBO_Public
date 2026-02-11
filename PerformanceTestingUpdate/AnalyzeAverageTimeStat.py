import pandas as pd
import numpy as np
from scipy import stats
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
    
    return combined_df


def perform_anova_tests(df):
    """
    Perform one-way ANOVA for each method to test if batch size affects time per sample.
    Also performs pairwise t-tests with Bonferroni correction and trend analysis.
    """
    # Calculate time per sample
    df['time_per_sample'] = df['time_seconds'] / df['batch_size']
    
    results = []
    
    methods = sorted(df['method_name'].unique())
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: Effect of Batch Size on Time Per Sample")
    print("="*80)
    
    for method in methods:
        method_data = df[df['method_name'] == method]
        batch_sizes = sorted(method_data['batch_size'].unique())
        
        print(f"\n{'='*80}")
        print(f"Method: {method}")
        print(f"{'='*80}")
        
        # Prepare data for each batch size
        groups = []
        group_stats = []
        
        for bs in batch_sizes:
            bs_data = method_data[method_data['batch_size'] == bs]['time_per_sample'].values
            groups.append(bs_data)
            group_stats.append({
                'batch_size': bs,
                'n': len(bs_data),
                'mean': np.mean(bs_data),
                'std': np.std(bs_data, ddof=1)
            })
        
        # Calculate trend using linear regression
        batch_size_values = np.array([stat['batch_size'] for stat in group_stats])
        mean_times = np.array([stat['mean'] for stat in group_stats])
        
        if len(batch_size_values) > 1:
            # Pearson correlation
            correlation, corr_p_value = stats.pearsonr(batch_size_values, mean_times)
            
            # Linear regression slope
            slope, intercept, r_value, p_value_slope, std_err = stats.linregress(batch_size_values, mean_times)
            
            # Determine trend
            if slope > 0:
                trend_direction = "Increasing"
                trend_description = f"Time per sample INCREASES with batch size (slope={slope:.6f})"
            elif slope < 0:
                trend_direction = "Decreasing"
                trend_description = f"Time per sample DECREASES with batch size (slope={slope:.6f})"
            else:
                trend_direction = "Flat"
                trend_description = "No clear trend"
            
            # Check if trend is statistically significant
            if p_value_slope < 0.05:
                trend_significance = "Significant"
            else:
                trend_significance = "Not significant"
        else:
            correlation = None
            slope = None
            p_value_slope = None
            r_value = None
            trend_direction = "N/A"
            trend_description = "Not enough data"
            trend_significance = "N/A"
        
        # Print descriptive statistics
        print("\nDescriptive Statistics:")
        print("-" * 60)
        for stat in group_stats:
            print(f"  Batch Size {stat['batch_size']}: "
                  f"n={stat['n']}, "
                  f"mean={stat['mean']:.6f}s, "
                  f"std={stat['std']:.6f}s")
        
        # Print trend analysis
        if len(batch_size_values) > 1:
            print(f"\nTrend Analysis:")
            print("-" * 60)
            print(f"  {trend_description}")
            print(f"  Correlation coefficient: {correlation:.4f}")
            print(f"  Trend p-value: {p_value_slope:.6f} ({trend_significance})")
            print(f"  R²: {r_value**2:.4f}")
        
        # Perform one-way ANOVA
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            
            print(f"\nOne-Way ANOVA Results:")
            print("-" * 60)
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  p-value: {p_value:.6f}")
            
            if p_value < 0.05:
                print(f"  Result: SIGNIFICANT difference (p < 0.05)")
                significance = "Yes"
            else:
                print(f"  Result: NO significant difference (p >= 0.05)")
                significance = "No"
            
            # Perform pairwise t-tests if ANOVA is significant
            if p_value < 0.05 and len(groups) > 2:
                print(f"\nPost-hoc Pairwise T-tests (with Bonferroni correction):")
                print("-" * 60)
                
                # Number of comparisons for Bonferroni correction
                n_comparisons = len(batch_sizes) * (len(batch_sizes) - 1) / 2
                bonferroni_alpha = 0.05 / n_comparisons
                
                print(f"  Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")
                print()
                
                for i in range(len(batch_sizes)):
                    for j in range(i + 1, len(batch_sizes)):
                        t_stat, t_p_value = stats.ttest_ind(groups[i], groups[j])
                        
                        diff = group_stats[i]['mean'] - group_stats[j]['mean']
                        sig_marker = "***" if t_p_value < bonferroni_alpha else ""
                        
                        print(f"  Batch {batch_sizes[i]} vs {batch_sizes[j]}: "
                              f"t={t_stat:.4f}, p={t_p_value:.6f}, "
                              f"diff={diff:.6f}s {sig_marker}")
            
            results.append({
                'method': method,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': significance,
                'trend_direction': trend_direction,
                'trend_slope': slope,
                'trend_significance': trend_significance,
                'batch_sizes_tested': batch_sizes
            })
        else:
            print("\nNot enough batch sizes to perform ANOVA")
            results.append({
                'method': method,
                'f_statistic': None,
                'p_value': None,
                'significant': 'N/A',
                'trend_direction': trend_direction,
                'trend_slope': slope,
                'trend_significance': trend_significance,
                'batch_sizes_tested': batch_sizes
            })
    
    return pd.DataFrame(results)


def save_results(results, output_filename='statistical_tests.csv'):
    """Save the test results to a CSV file."""
    output_path = os.path.join(os.getcwd(), output_filename)
    results.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")
    return output_path


def print_summary(results):
    """Print a summary table of all tests."""
    print("\n" + "="*80)
    print("SUMMARY OF STATISTICAL TESTS")
    print("="*80)
    print("\nOne-Way ANOVA Results for Each Method:")
    print("-"*80)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    summary = results[['method', 'f_statistic', 'p_value', 'significant', 
                       'trend_direction', 'trend_slope', 'trend_significance']].copy()
    print(summary.to_string(index=False))
    
    print("\n" + "="*80)
    print("\nInterpretation:")
    print("  - 'Significant' = Batch size has a statistically significant effect (p < 0.05)")
    print("  - 'Increasing' = Time per sample increases as batch size increases")
    print("  - 'Decreasing' = Time per sample decreases as batch size increases")
    print("  - Slope = Change in time per sample (seconds) per unit increase in batch size")
    print("="*80)


def main():
    print("="*80)
    print("STATISTICAL TESTING: Batch Size Effect on Time Per Sample")
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
    
    # Perform statistical tests
    results = perform_anova_tests(combined_df)
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()