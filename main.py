#!/usr/bin/env python3
"""
AI-Assisted Qualitative Coding Agent

This script automates the initial open-coding phase of qualitative data analysis
using Google's Gemini API. It processes CSV files containing question-answer pairs
and generates initial codes for each entry using batch processing for efficiency.
"""

import os
import json
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Constants
BATCH_SIZE = 20
DEFAULT_QUESTION_COL = "question"
DEFAULT_ANSWER_COL = "answer"

def load_config() -> bool:
    """
    Load environment variables and configure the Gemini client.
    
    Returns:
        bool: True if configuration is successful, False otherwise
    """
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("Error: GEMINI_API_KEY not found in environment variables.")
            print("Please create a .env file with your API key:")
            print("GEMINI_API_KEY='your_api_key_here'")
            return False
        
        # Configure Gemini client
        genai.configure(api_key=api_key)
        print("✓ Gemini API configured successfully")
        return True
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False

def load_csv_data(file_path: str, question_col: str = DEFAULT_QUESTION_COL, 
                  answer_col: str = DEFAULT_ANSWER_COL) -> Optional[pd.DataFrame]:
    """
    Load CSV file and validate required columns.
    
    Args:
        file_path (str): Path to the CSV file
        question_col (str): Name of the question column
        answer_col (str): Name of the answer column
        
    Returns:
        pd.DataFrame or None: Loaded DataFrame or None if error
    """
    try:
        # Load CSV file
        df = pd.read_csv(file_path)
        print(f"✓ Loaded CSV file: {file_path}")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # Validate required columns
        if question_col not in df.columns:
            print(f"Error: Column '{question_col}' not found in CSV.")
            print(f"Available columns: {list(df.columns)}")
            return None
            
        if answer_col not in df.columns:
            print(f"Error: Column '{answer_col}' not found in CSV.")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Remove rows with missing data in required columns
        initial_rows = len(df)
        df = df.dropna(subset=[question_col, answer_col])
        final_rows = len(df)
        
        if initial_rows > final_rows:
            print(f"  Removed {initial_rows - final_rows} rows with missing data")
        
        print(f"✓ Validated columns: '{question_col}' and '{answer_col}'")
        return df
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def get_codes_for_batch(batch_df: pd.DataFrame, question_col: str, 
                       answer_col: str) -> List[Dict[str, Any]]:
    """
    Generate codes and sentiment analysis for a batch of question-answer pairs using Gemini API.
    
    Args:
        batch_df (pd.DataFrame): DataFrame chunk containing the batch
        question_col (str): Name of the question column
        answer_col (str): Name of the answer column
        
    Returns:
        List[Dict]: List of analysis objects with 'id', 'code', and 'sentiment' keys
    """
    try:
        # Convert batch to list of dictionaries
        batch_data = []
        for idx, row in batch_df.iterrows():
            batch_data.append({
                "id": idx,
                "question": str(row[question_col]),
                "answer": str(row[answer_col])
            })
        
        # Construct the refined batch prompt
        prompt = """You are an expert qualitative research assistant. Your task is to perform open coding for grounded theory analysis on student feedback.

Your goal is to generate a brief, informative initial code (3-6 words) that captures the central theme of each student's response. You must also determine the sentiment of each response.

**Instructions:**
1.  **Distill the Essence:** For long responses with multiple points, distill the answer to its single most important underlying issue or idea. Do not just summarize one part of it.
2.  **Be Direct:** For very short or simple answers, provide a straightforward code.
3.  **Format:** Return your analysis as a JSON array where each object has three keys: "id" (matching the input), "code" (3-6 words), and "sentiment" (values: "Positive", "Negative", "Neutral").

**Examples of Correct Analysis:**

---
**Example 1:**
* **Question:** What did you like the least about the course?
* **Answer:** nothing, i love it.
* **Analysis:**
    {
      "id": 0,
      "code": "Positive Overall Feedback",
      "sentiment": "Positive"
    }

---
**Example 2:**
* **Question:** What did you like the least about the course?
* **Answer:** The teacher is never present in any of the videos, nor does she seem to prioritize our success or concerns...
* **Analysis:**
    {
      "id": 1,
      "code": "Lack of Instructor Feedback/Support",
      "sentiment": "Negative"
    }
---

**Now, perform the analysis on the following data:**

Input:
""" + json.dumps(batch_data, indent=2) + """

Return your analysis as a JSON array:

```json"""
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Clean up response text (remove markdown code blocks if present)
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        analysis_results = json.loads(response_text)
        
        print(f"✓ Generated codes and sentiment analysis for {len(analysis_results)} items")
        return analysis_results
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response text: {response_text}")
        return []
    except Exception as e:
        print(f"Error generating analysis for batch: {e}")
        return []

def process_csv_in_batches(df: pd.DataFrame, question_col: str, 
                          answer_col: str, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """
    Process the entire DataFrame in batches and generate codes with sentiment analysis.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        question_col (str): Name of the question column
        answer_col (str): Name of the answer column
        batch_size (int): Number of rows to process in each batch
        
    Returns:
        pd.DataFrame: DataFrame with added 'Initial_Code' and 'Sentiment' columns
    """
    all_codes = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    print(f"Processing {len(df)} rows in {total_batches} batches of {batch_size}")
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        
        # Get codes for this batch
        batch_codes = get_codes_for_batch(batch_df, question_col, answer_col)
        all_codes.extend(batch_codes)
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    # Create DataFrame from analysis results
    analysis_df = pd.DataFrame(all_codes)
    
    # Merge with original DataFrame
    if not analysis_df.empty:
        analysis_df = analysis_df.set_index('id')
        result_df = df.copy()
        result_df['Initial_Code'] = analysis_df['code']
        result_df['Sentiment'] = analysis_df['sentiment']
    else:
        result_df = df.copy()
        result_df['Initial_Code'] = None
        result_df['Sentiment'] = None
    
    return result_df

def save_output_csv(df: pd.DataFrame, input_file_path: str) -> str:
    """
    Save the processed DataFrame to a new CSV file.
    
    Args:
        df (pd.DataFrame): Processed DataFrame with codes
        input_file_path (str): Path to the original input file
        
    Returns:
        str: Path to the output file
    """
    # Generate output filename
    base_name = os.path.splitext(input_file_path)[0]
    output_path = f"{base_name}_coded.csv"
    
    # Save without index
    df.to_csv(output_path, index=False)
    print(f"✓ Saved results to: {output_path}")
    
    return output_path

def main():
    """
    Main function to orchestrate the qualitative coding process.
    """
    print("=== AI-Assisted Qualitative Coding Agent ===\n")
    
    # Load configuration
    if not load_config():
        return
    
    # Get input file path from user
    input_file = input("Enter the path to your CSV file: ").strip()
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return
    
    # Get column names from user (with defaults)
    question_col = input(f"Enter question column name (default: '{DEFAULT_QUESTION_COL}'): ").strip()
    if not question_col:
        question_col = DEFAULT_QUESTION_COL
        
    answer_col = input(f"Enter answer column name (default: '{DEFAULT_ANSWER_COL}'): ").strip()
    if not answer_col:
        answer_col = DEFAULT_ANSWER_COL
    
    # Load CSV data
    df = load_csv_data(input_file, question_col, answer_col)
    if df is None:
        return
    
    # Process data in batches
    print(f"\nStarting batch processing...")
    result_df = process_csv_in_batches(df, question_col, answer_col)
    
    # Save output
    output_file = save_output_csv(result_df, input_file)
    
    print(f"\n=== Processing Complete ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Total rows processed: {len(result_df)}")
    
    # Show sample of results
    coded_count = result_df['Initial_Code'].notna().sum()
    print(f"Successfully coded: {coded_count}/{len(result_df)} rows")
    
    if coded_count > 0:
        print("\nSample of generated analysis:")
        sample_df = result_df[result_df['Initial_Code'].notna()].head(3)
        for _, row in sample_df.iterrows():
            print(f"  Question: {row[question_col][:50]}...")
            print(f"  Answer: {row[answer_col][:50]}...")
            print(f"  Code: {row['Initial_Code']}")
            print(f"  Sentiment: {row['Sentiment']}")
            print()

if __name__ == "__main__":
    main() 