#!/usr/bin/env python3
"""
AI-Assisted Qualitative Coding Agent

This script automates the initial open-coding phase of qualitative data analysis
using Google's Gemini API. It processes CSV files containing question-answer pairs
and generates initial codes for each entry using batch processing for efficiency.

Enhanced with Context-Aware Coding and Batch Summaries for improved consistency.
"""

import os
import json
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Callable, Set

# Constants
BATCH_SIZE = 20
DEFAULT_QUESTION_COL = "question"
DEFAULT_ANSWER_COL = "answer"
CONTEXT_MEMORY_THRESHOLD = 30  # Start including context when we have 30+ codes
CONTEXT_SAMPLE_SIZE = 25  # Include 20-30 recent codes in context

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

def get_codes_for_batch_with_context(batch_df: pd.DataFrame, question_col: str, 
                                   answer_col: str, context_codes: List[str] = None) -> Dict[str, Any]:
    """
    Generate codes and sentiment analysis for a batch of question-answer pairs using Gemini API,
    with optional context from previously generated codes.
    
    Args:
        batch_df (pd.DataFrame): DataFrame chunk containing the batch
        question_col (str): Name of the question column
        answer_col (str): Name of the answer column
        context_codes (List[str], optional): List of recently generated codes for context
        
    Returns:
        Dict: Dictionary with 'analysis' (list of analysis objects) and 'batch_summary' (str)
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
        
        # Build context section if codes are provided
        context_section = ""
        if context_codes and len(context_codes) >= CONTEXT_MEMORY_THRESHOLD:
            recent_codes = context_codes[-CONTEXT_SAMPLE_SIZE:]  # Get most recent codes
            context_section = f"""
**CONTEXT - Recently Used Codes:**
Here are codes you have used recently in this analysis session: {', '.join(f'"{code}"' for code in recent_codes)}

**IMPORTANT:** Before creating a new code, check whether one of these existing codes accurately represents the response. Prioritize reuse of existing codes when they fit. Only create a new code when none of the existing codes accurately capture the essence of the response.
"""
        
        # Construct the enhanced batch prompt with context awareness
        prompt = f"""You are an expert qualitative research assistant. Your task is to perform open coding for grounded theory analysis on student feedback.

Your goal is to generate a brief, informative initial code (3-6 words) that captures the central theme of each student's response. You must also determine the sentiment of each response.
{context_section}
**Instructions:**
1. **Distill the Essence:** For long responses with multiple points, distill the answer to its single most important underlying issue or idea. Do not just summarize one part of it.
2. **Be Direct:** For very short or simple answers, provide a straightforward code.
3. **Consistency Priority:** {"If context codes are provided above, prioritize reusing an existing code that accurately fits the response rather than creating a new variant." if context_codes else ""}
4. **Format:** Return your analysis as a JSON object with two parts:
   - "analysis": Array where each object has three keys: "id" (matching the input), "code" (3-6 words), and "sentiment" (values: "Positive", "Negative", "Neutral")
   - "batch_summary": A single sentence summarizing the key concepts or themes found in this batch

**Examples of Correct Analysis:**

---
**Example 1:**
* **Question:** What did you like the least about the course?
* **Answer:** nothing, i love it.
* **Analysis:**
    {{
      "id": 0,
      "code": "Positive Overall Feedback",
      "sentiment": "Positive"
    }}

---
**Example 2:**
* **Question:** What did you like the least about the course?
* **Answer:** The teacher is never present in any of the videos, nor does she seem to prioritize our success or concerns...
* **Analysis:**
    {{
      "id": 1,
      "code": "Lack of Instructor Feedback/Support",
      "sentiment": "Negative"
    }}
---

**Now, perform the analysis on the following data:**

Input:
{json.dumps(batch_data, indent=2)}

Return your analysis as a JSON object with both "analysis" array and "batch_summary" string:

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
        result = json.loads(response_text)
        
        # Validate response structure
        if not isinstance(result, dict) or 'analysis' not in result:
            print("Warning: Unexpected response format, attempting to parse as legacy format")
            # Try to parse as legacy format (direct array)
            if isinstance(result, list):
                return {
                    'analysis': result,
                    'batch_summary': "Batch processed successfully"
                }
            else:
                raise ValueError("Invalid response format")
        
        analysis_results = result.get('analysis', [])
        batch_summary = result.get('batch_summary', "Batch processed successfully")
        
        print(f"✓ Generated codes and sentiment analysis for {len(analysis_results)} items")
        print(f"✓ Batch summary: {batch_summary}")
        
        return {
            'analysis': analysis_results,
            'batch_summary': batch_summary
        }
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response text: {response_text}")
        return {
            'analysis': [],
            'batch_summary': "Error processing batch"
        }
    except Exception as e:
        print(f"Error generating analysis for batch: {e}")
        return {
            'analysis': [],
            'batch_summary': "Error processing batch"
        }

def get_codes_for_batch(batch_df: pd.DataFrame, question_col: str, 
                       answer_col: str) -> List[Dict[str, Any]]:
    """
    Legacy function wrapper for backward compatibility.
    Generate codes and sentiment analysis for a batch without context.
    
    Args:
        batch_df (pd.DataFrame): DataFrame chunk containing the batch
        question_col (str): Name of the question column
        answer_col (str): Name of the answer column
        
    Returns:
        List[Dict]: List of analysis objects with 'id', 'code', and 'sentiment' keys
    """
    result = get_codes_for_batch_with_context(batch_df, question_col, answer_col)
    return result['analysis']

def update_context_memory(recently_generated_codes: List[str], new_codes: List[str], 
                         max_memory_size: int = 100) -> List[str]:
    """
    Update the context memory with new codes, maintaining deduplication and size limits.
    
    Args:
        recently_generated_codes (List[str]): Current list of recent codes
        new_codes (List[str]): New codes to add
        max_memory_size (int): Maximum number of codes to keep in memory
        
    Returns:
        List[str]: Updated list of recent codes
    """
    # Convert to set for deduplication, then back to list to preserve order
    seen = set()
    updated_codes = []
    
    # Add existing codes (in order)
    for code in recently_generated_codes:
        if code not in seen:
            updated_codes.append(code)
            seen.add(code)
    
    # Add new codes
    for code in new_codes:
        if code and code not in seen:  # Skip None/empty codes
            updated_codes.append(code)
            seen.add(code)
    
    # Limit size to prevent token bloat
    if len(updated_codes) > max_memory_size:
        updated_codes = updated_codes[-max_memory_size:]
    
    return updated_codes

def process_csv_in_batches_with_context(df: pd.DataFrame, question_col: str, 
                                      answer_col: str, batch_size: int = BATCH_SIZE, 
                                      progress_callback: Optional[Callable[[str], None]] = None,
                                      summary_callback: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    """
    Process the entire DataFrame in batches with context-aware coding and generate codes with sentiment analysis.
    Enhanced with context memory and batch summaries.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        question_col (str): Name of the question column
        answer_col (str): Name of the answer column
        batch_size (int): Number of rows to process in each batch
        progress_callback (callable): Optional callback function for progress updates
        summary_callback (callable): Optional callback function for batch summaries
        
    Returns:
        pd.DataFrame: DataFrame with added 'Initial_Code' and 'Sentiment' columns
    """
    all_codes = []
    recently_generated_codes = []  # Context memory
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    print(f"Processing {len(df)} rows in {total_batches} batches of {batch_size}")
    print(f"Context-aware coding enabled (threshold: {CONTEXT_MEMORY_THRESHOLD} codes)")
    
    # Process in batches with progress bar
    for batch_num, i in enumerate(tqdm(range(0, len(df), batch_size), desc="Processing batches")):
        batch_df = df.iloc[i:i+batch_size]
        
        # Send batch progress update if callback provided
        if progress_callback:
            batch_percentage = int((batch_num / total_batches) * 100)
            progress_callback(f"Processing batch {batch_num + 1} of {total_batches} ({batch_percentage}%)")
        
        # Determine whether to use context
        use_context = len(recently_generated_codes) >= CONTEXT_MEMORY_THRESHOLD
        context_codes = recently_generated_codes if use_context else None
        
        if use_context:
            print(f"  Using context memory ({len(recently_generated_codes)} codes available)")
        
        # Get codes for this batch with context
        batch_result = get_codes_for_batch_with_context(batch_df, question_col, answer_col, context_codes)
        batch_codes = batch_result['analysis']
        batch_summary = batch_result['batch_summary']
        
        # Send batch summary if callback provided
        if summary_callback and batch_summary:
            summary_callback(batch_summary)
        
        all_codes.extend(batch_codes)
        
        # Update context memory with new codes
        new_codes = [item.get('code') for item in batch_codes if item.get('code')]
        recently_generated_codes = update_context_memory(recently_generated_codes, new_codes)
        
        print(f"  Batch {batch_num + 1}: Generated {len(batch_codes)} codes, memory size: {len(recently_generated_codes)}")
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    # Send completion update if callback provided
    if progress_callback:
        progress_callback(f"Completed all {total_batches} batches (100%)")
    
    # Final context memory stats
    unique_codes = len(set(recently_generated_codes))
    total_comments = len(df)
    uniqueness_percentage = (unique_codes / total_comments) * 100 if total_comments > 0 else 0
    
    print(f"✓ Context-aware coding completed:")
    print(f"  Total comments: {total_comments}")
    print(f"  Unique codes generated: {unique_codes}")
    print(f"  Code uniqueness: {uniqueness_percentage:.1f}% (target: ≤40%)")
    
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

def process_csv_in_batches(df: pd.DataFrame, question_col: str, 
                          answer_col: str, batch_size: int = BATCH_SIZE, progress_callback=None) -> pd.DataFrame:
    """
    Legacy wrapper function for backward compatibility.
    Process the entire DataFrame in batches and generate codes with sentiment analysis.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        question_col (str): Name of the question column
        answer_col (str): Name of the answer column
        batch_size (int): Number of rows to process in each batch
        progress_callback (callable): Optional callback function for progress updates
        
    Returns:
        pd.DataFrame: DataFrame with added 'Initial_Code' and 'Sentiment' columns
    """
    return process_csv_in_batches_with_context(df, question_col, answer_col, batch_size, progress_callback)

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