#!/usr/bin/env python3
"""
Phase 0: Data Formatting & Cleaning Agent

This script cleans raw survey export data and transforms it into a standardized CSV format
for use in the qualitative analysis pipeline (Phase 1: Initial Coding and Phase 2: Thematic Analysis).

Author: Qualitative Coding Agent
"""

import pandas as pd
import csv
import os
import sys
from pathlib import Path


def load_raw_data(file_path):
    """
    Load raw data from various formats (CSV, TXT) and handle different delimiters.
    
    Args:
        file_path (str): Path to the raw data file
        
    Returns:
        pd.DataFrame: Loaded raw data
    """
    print(f"Loading raw data from: {file_path}")
    
    try:
        # First, try to read as standard CSV
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Standard CSV loading failed: {e}")
        
        try:
            # If standard CSV fails, try space-delimited format
            print("Attempting to read as space-delimited format...")
            df = pd.read_csv(file_path, sep=r'\s\s+', header=None, engine='python')
            print(f"Successfully loaded space-delimited data with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e2:
            print(f"Space-delimited loading failed: {e2}")
            
            try:
                # Try tab-delimited as another fallback
                print("Attempting to read as tab-delimited format...")
                df = pd.read_csv(file_path, sep='\t')
                print(f"Successfully loaded tab-delimited data with {len(df)} rows and {len(df.columns)} columns")
                return df
            except Exception as e3:
                print(f"All loading attempts failed. Last error: {e3}")
                raise Exception(f"Could not load data file: {file_path}")


def assign_headers_if_needed(df):
    """
    Assign proper headers if the data doesn't have them or has incorrect ones.
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Dataframe with proper headers
    """
    # Check if we have the expected columns
    expected_columns = ['crs_number', 'SectionNumber_ASU', 'question', 'response']
    
    if all(col in df.columns for col in expected_columns):
        print("Data already has proper headers")
        return df
    
    # If we don't have headers or they're wrong, we need to define them
    # This is a fallback for cases where the data comes without headers
    if df.columns[0] == 0 or 'Unnamed' in str(df.columns[0]):
        print("No headers detected, assigning standard headers...")
        
        # Expected header structure based on the sample data
        headers = [
            'period', 'COURSETYPE', 'crs_number', 'SectionNumber_CE', 'SectionNumber_ASU',
            'dept', 'question', 'response', 'SubmitDate', 'eval_username', 'survey', 'DummyID'
        ]
        
        if len(df.columns) == len(headers):
            df.columns = headers
            print("Headers assigned successfully")
        else:
            print(f"Warning: Expected {len(headers)} columns but found {len(df.columns)}")
            # Create generic headers if count doesn't match
            df.columns = [f'col_{i}' for i in range(len(df.columns))]
    
    return df


def select_essential_columns(df):
    """
    Select only the essential columns needed for qualitative analysis.
    
    Args:
        df (pd.DataFrame): Full dataframe
        
    Returns:
        pd.DataFrame: Dataframe with only essential columns
    """
    essential_columns = ['crs_number', 'SectionNumber_ASU', 'question', 'response']
    
    print(f"Selecting essential columns: {essential_columns}")
    
    # Check if all essential columns exist
    missing_columns = [col for col in essential_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing essential columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        
        # Try to map similar column names
        column_mapping = {}
        for missing_col in missing_columns:
            for existing_col in df.columns:
                if missing_col.lower() in existing_col.lower() or existing_col.lower() in missing_col.lower():
                    column_mapping[existing_col] = missing_col
                    print(f"Mapping '{existing_col}' to '{missing_col}'")
                    break
        
        # Rename mapped columns
        if column_mapping:
            df = df.rename(columns=column_mapping)
    
    # Select available essential columns
    available_essential = [col for col in essential_columns if col in df.columns]
    
    if not available_essential:
        raise Exception("No essential columns found in the data")
    
    selected_df = df[available_essential].copy()
    print(f"Selected {len(available_essential)} essential columns: {available_essential}")
    
    return selected_df


def clean_data_cells(df):
    """
    Clean data by trimming whitespace, removing problematic characters, and handling missing values.
    This aggressive cleaning prevents CSV parsing issues in downstream analysis.
    
    Args:
        df (pd.DataFrame): Dataframe to clean
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("Cleaning data cells...")
    
    # Create a copy to avoid modifying the original
    clean_df = df.copy()
    
    # Clean text columns aggressively to prevent CSV parsing issues
    for col in clean_df.columns:
        if clean_df[col].dtype == 'object':
            # Convert to string and handle basic cleaning
            clean_df[col] = clean_df[col].astype(str)
            
            # Replace 'nan' strings with actual NaN first
            clean_df[col] = clean_df[col].replace(['nan', 'NaN', 'NULL', ''], pd.NA)
            
            # Only clean non-null values
            mask = clean_df[col].notna()
            if mask.any():
                # Aggressive text cleaning for CSV compatibility
                text_series = clean_df.loc[mask, col]
                
                # Remove or replace problematic characters
                text_series = text_series.str.replace('\n', ' ', regex=False)  # Replace newlines with spaces
                text_series = text_series.str.replace('\r', ' ', regex=False)  # Replace carriage returns
                text_series = text_series.str.replace('\t', ' ', regex=False)  # Replace tabs with spaces
                
                # Clean up multiple spaces and trim
                text_series = text_series.str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
                text_series = text_series.str.strip()  # Trim leading/trailing whitespace
                
                # Handle problematic quote characters that can break CSV
                text_series = text_series.str.replace('"', "'", regex=False)  # Replace double quotes with single
                text_series = text_series.str.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', regex=True)  # Remove control characters
                
                # Limit extremely long text that might cause issues (optional safeguard)
                max_length = 10000  # Reasonable limit for survey responses
                text_series = text_series.apply(lambda x: x[:max_length] if len(x) > max_length else x)
                
                # Update the cleaned values back to the dataframe
                clean_df.loc[mask, col] = text_series
    
    print("Data cleaning completed")
    return clean_df


def deduplicate_data(df):
    """
    Remove duplicate rows based on question and response columns.
    
    Args:
        df (pd.DataFrame): Dataframe to deduplicate
        
    Returns:
        pd.DataFrame: Deduplicated dataframe
    """
    print("Removing duplicate entries...")
    
    initial_count = len(df)
    
    # Check if both question and response columns exist
    dedup_columns = []
    if 'question' in df.columns:
        dedup_columns.append('question')
    if 'response' in df.columns:
        dedup_columns.append('response')
    
    if not dedup_columns:
        print("Warning: No question or response columns found for deduplication")
        return df
    
    # Remove duplicates based on available columns
    deduplicated_df = df.drop_duplicates(subset=dedup_columns, keep='first')
    
    final_count = len(deduplicated_df)
    removed_count = initial_count - final_count
    
    print(f"Removed {removed_count} duplicate entries ({initial_count} -> {final_count} rows)")
    
    return deduplicated_df


def save_cleaned_data(df, output_path):
    """
    Save the cleaned dataframe to a CSV file with robust formatting to prevent parsing issues.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        output_path (str): Path for the output file
    """
    print(f"Saving cleaned data to: {output_path}")
    
    # Save with robust CSV formatting to prevent downstream parsing issues
    df.to_csv(
        output_path,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,  # Quote all non-numeric fields
        doublequote=True,              # Handle quotes within quotes
        encoding='utf-8',
        lineterminator='\n'            # Use consistent line endings
    )
    
    print(f"Successfully saved {len(df)} rows to {output_path}")


def format_data(input_file, output_file=None):
    """
    Main function to format and clean raw survey data.
    
    Args:
        input_file (str): Path to the raw data file
        output_file (str, optional): Path for the cleaned output file (for backward compatibility)
        
    Returns:
        pd.DataFrame: Cleaned DataFrame ready for analysis
    """
    print("=" * 60)
    print("Phase 0: Data Formatting & Cleaning Agent")
    print("=" * 60)
    
    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    try:
        # Step 1: Load raw data
        df = load_raw_data(input_file)
        
        # Step 2: Assign headers if needed
        df = assign_headers_if_needed(df)
        
        # Step 3: Select essential columns
        df = select_essential_columns(df)
        
        # Step 4: Clean data cells
        df = clean_data_cells(df)
        
        # Step 5: Deduplicate data
        df = deduplicate_data(df)
        
        # Step 6: Save cleaned data if output_file is specified (for backward compatibility)
        if output_file is not None:
            save_cleaned_data(df, output_file)
            print(f"Clean data saved to: {output_file}")
        
        print("=" * 60)
        print("Data formatting completed successfully!")
        print(f"Cleaned DataFrame has {len(df)} rows and {len(df.columns)} columns")
        print("=" * 60)
        
        return df
        
    except Exception as e:
        print(f"Error during data formatting: {e}")
        raise


def main():
    """
    Command-line interface for the data formatter.
    """
    if len(sys.argv) < 2:
        print("Usage: python formatter.py <input_file> [output_file]")
        print("Example: python formatter.py CommentsRAW_SAMPLE.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        cleaned_file = format_data(input_file, output_file)
        print(f"\nSuccess! Cleaned data available at: {cleaned_file}")
        print("This file is ready for Phase 1 (Initial Coding) processing.")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 