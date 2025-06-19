#!/usr/bin/env python3
"""
Integrated Three-Phase Qualitative Analysis Pipeline (In-Memory Version)

This script provides a complete workflow for qualitative data analysis using DataFrames:
- Phase 0: Data Formatting & Cleaning
- Phase 1: Initial Coding with AI  
- Phase 2: Thematic Analysis & Report Generation
- Phase 4: Executive Summary Generation

Author: Qualitative Coding Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os
import sys
import json
import google.generativeai as genai

# Import our refactored modules
from formatter import format_data
from main import process_csv_in_batches, load_config
from thematic_analyzer import analyze_themes


def smart_column_mapping(df: pd.DataFrame, target_columns: list) -> Dict[str, str]:
    """
    Intelligently map DataFrame columns to expected column names.
    
    Args:
        df (pd.DataFrame): DataFrame to examine
        target_columns (list): List of expected column names
        
    Returns:
        Dict[str, str]: Mapping of existing columns to target columns
    """
    mapping = {}
    df_columns_lower = [col.lower() for col in df.columns]
    
    for target in target_columns:
        target_lower = target.lower()
        
        # Direct match
        if target in df.columns:
            mapping[target] = target
            continue
            
        # Case-insensitive match
        for i, col_lower in enumerate(df_columns_lower):
            if col_lower == target_lower:
                mapping[list(df.columns)[i]] = target
                break
        else:
            # Partial match
            for i, col_lower in enumerate(df_columns_lower):
                if target_lower in col_lower or col_lower in target_lower:
                    mapping[list(df.columns)[i]] = target
                    break
    
    return mapping


def merge_with_schedule(final_df: pd.DataFrame, schedule_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Merge themed data with schedule information with validation.
    
    Args:
        final_df (pd.DataFrame): Themed comments DataFrame
        schedule_df (pd.DataFrame): Schedule DataFrame
        
    Returns:
        tuple: (Merged DataFrame, validation_result dict)
    """
    print("Merging with schedule data...")
    
    # Find the section column in schedule data
    possible_section_cols = ['SectionNumber_ASU', 'Section', 'SectionNumber', 'section', 'SECTION', 'Number', 'number', 'CLASS_NBR', 'Class_Nbr']
    schedule_section_col = None
    
    for col in possible_section_cols:
        if col in schedule_df.columns:
            schedule_section_col = col
            break
    
    if not schedule_section_col:
        return final_df, {
            "status": "error",
            "message": f"No section column found in schedule data. Available columns: {list(schedule_df.columns)}",
            "type": "missing_column"
        }
    
    # Ensure both merge keys are numeric for consistent matching
    final_df['SectionNumber_ASU'] = pd.to_numeric(final_df['SectionNumber_ASU'], errors='coerce')
    schedule_df[schedule_section_col] = pd.to_numeric(schedule_df[schedule_section_col], errors='coerce')
    
    # Remove rows with NaN section numbers
    final_df_clean = final_df.dropna(subset=['SectionNumber_ASU'])
    schedule_df_clean = schedule_df.dropna(subset=[schedule_section_col])
    
    # Check for overlap before merging
    comments_sections = set(final_df_clean['SectionNumber_ASU'].astype(int))
    schedule_sections = set(schedule_df_clean[schedule_section_col].astype(int))
    overlapping_sections = comments_sections.intersection(schedule_sections)
    
    overlap_percentage = (len(overlapping_sections) / len(comments_sections)) * 100 if comments_sections else 0
    
    print(f"📊 Section overlap analysis:")
    print(f"   Comments sections: {len(comments_sections)}")
    print(f"   Schedule sections: {len(schedule_sections)}")
    print(f"   Overlapping sections: {len(overlapping_sections)}")
    print(f"   Overlap percentage: {overlap_percentage:.1f}%")
    
    # If overlap is too low, return warning instead of merging
    if overlap_percentage < 10:  # Less than 10% overlap
        return final_df, {
            "status": "warning",
            "message": f"Low section number overlap detected ({overlap_percentage:.1f}%). "
                      f"Comments data appears to be from a different time period than schedule data. "
                      f"Please verify that your files are from the same academic term/year.",
            "type": "time_period_mismatch",
            "details": {
                "comments_sections": len(comments_sections),
                "schedule_sections": len(schedule_sections),
                "overlapping_sections": len(overlapping_sections),
                "overlap_percentage": overlap_percentage,
                "sample_comments_sections": sorted(list(comments_sections))[:5],
                "sample_schedule_sections": sorted(list(schedule_sections))[:5],
                "sample_overlapping": sorted(list(overlapping_sections))[:5] if overlapping_sections else []
            }
        }
    
    # Perform merge only if good overlap
    merged_df = pd.merge(
        final_df, 
        schedule_df, 
        left_on='SectionNumber_ASU', 
        right_on=schedule_section_col, 
        how='left'
    )
    
    merged_count = len(merged_df[merged_df[schedule_section_col].notna()])
    
    print(f"✅ Merged with schedule data using '{schedule_section_col}' column")
    print(f"✅ Successfully merged {merged_count} rows with schedule information")
    
    return merged_df, {
        "status": "success",
        "message": f"Successfully merged schedule data. {merged_count} rows matched.",
        "type": "successful_merge",
        "details": {
            "merged_rows": merged_count,
            "total_rows": len(final_df),
            "merge_column": schedule_section_col
        }
    }


def merge_with_grades(final_df: pd.DataFrame, grades_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Merge data with grades information with validation.
    
    Args:
        final_df (pd.DataFrame): Current DataFrame
        grades_df (pd.DataFrame): Grades DataFrame
        
    Returns:
        tuple: (Merged DataFrame, validation_result dict)
    """
    print("Merging with grades data...")
    
    # Find the section column in grades data
    possible_section_cols = ['SectionNumber_ASU', 'Section', 'SectionNumber', 'section', 'SECTION', 'Class Nbr', 'Class_Nbr', 'CLASS_NBR', 'Number', 'number']
    grades_section_col = None
    
    for col in possible_section_cols:
        if col in grades_df.columns:
            grades_section_col = col
            break
    
    if not grades_section_col:
        return final_df, {
            "status": "error",
            "message": f"No section column found in grades data. Available columns: {list(grades_df.columns)}",
            "type": "missing_column"
        }
    
    # Ensure both merge keys are numeric for consistent matching
    final_df['SectionNumber_ASU'] = pd.to_numeric(final_df['SectionNumber_ASU'], errors='coerce')
    grades_df[grades_section_col] = pd.to_numeric(grades_df[grades_section_col], errors='coerce')
    
    # Remove rows with NaN section numbers
    final_df_clean = final_df.dropna(subset=['SectionNumber_ASU'])
    grades_df_clean = grades_df.dropna(subset=[grades_section_col])
    
    # Check for overlap before merging
    comments_sections = set(final_df_clean['SectionNumber_ASU'].astype(int))
    grades_sections = set(grades_df_clean[grades_section_col].astype(int))
    overlapping_sections = comments_sections.intersection(grades_sections)
    
    overlap_percentage = (len(overlapping_sections) / len(comments_sections)) * 100 if comments_sections else 0
    
    print(f"📊 Section overlap analysis:")
    print(f"   Comments sections: {len(comments_sections)}")
    print(f"   Grades sections: {len(grades_sections)}")
    print(f"   Overlapping sections: {len(overlapping_sections)}")
    print(f"   Overlap percentage: {overlap_percentage:.1f}%")
    
    # If overlap is too low, return warning instead of merging
    if overlap_percentage < 10:  # Less than 10% overlap
        return final_df, {
            "status": "warning",
            "message": f"Low section number overlap detected ({overlap_percentage:.1f}%). "
                      f"Comments data appears to be from a different time period than grades data. "
                      f"Please verify that your files are from the same academic term/year.",
            "type": "time_period_mismatch",
            "details": {
                "comments_sections": len(comments_sections),
                "grades_sections": len(grades_sections),
                "overlapping_sections": len(overlapping_sections),
                "overlap_percentage": overlap_percentage,
                "sample_comments_sections": sorted(list(comments_sections))[:5],
                "sample_grades_sections": sorted(list(grades_sections))[:5],
                "sample_overlapping": sorted(list(overlapping_sections))[:5] if overlapping_sections else []
            }
        }
    
    # Perform merge only if good overlap
    merged_df = pd.merge(
        final_df, 
        grades_df, 
        left_on='SectionNumber_ASU', 
        right_on=grades_section_col, 
        how='left'
    )
    
    merged_count = len(merged_df[merged_df[grades_section_col].notna()])
    
    print(f"✅ Merged with grades data using '{grades_section_col}' column")
    print(f"✅ Successfully merged {merged_count} rows with grades information")
    
    return merged_df, {
        "status": "success",
        "message": f"Successfully merged grades data. {merged_count} rows matched.",
        "type": "successful_merge",
        "details": {
            "merged_rows": merged_count,
            "total_rows": len(final_df),
            "merge_column": grades_section_col
        }
    }


def clean_for_json_serialization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame for JSON serialization.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    clean_df = df.copy()
    
    # Handle different data types appropriately
    for col in clean_df.columns:
        if clean_df[col].dtype == 'object':
            # Text columns: convert to string and fill NaN with empty string
            clean_df[col] = clean_df[col].fillna('').astype(str)
        elif np.issubdtype(clean_df[col].dtype, np.number):
            # Numeric columns: fill NaN with 0 and handle inf values
            clean_df[col] = clean_df[col].replace([np.inf, -np.inf], np.nan)
            clean_df[col] = clean_df[col].fillna(0)
        else:
            # Other types: convert to string
            clean_df[col] = clean_df[col].fillna('').astype(str)
    
    return clean_df


def generate_executive_summary(final_df: pd.DataFrame) -> str:
    """
    Generate an executive summary based on the final merged dataset.
    
    Args:
        final_df (pd.DataFrame): Final merged DataFrame with all analysis data
        
    Returns:
        str: Executive summary as markdown text
    """
    try:
        print("\n📝 PHASE 4: EXECUTIVE SUMMARY GENERATION")
        print("-" * 50)
        
        # Convert DataFrame to JSON for the prompt
        data_json = final_df.to_json(orient='records', indent=2)
        
        # Create the executive summary prompt
        executive_summary_prompt = f"""Role: You are an expert educational data analyst and strategist. Your task is to write a concise executive summary based on a comprehensive dataset of student feedback, course details, and grade distributions. The audience for this report is a university department head who needs to quickly understand the key findings and make data-driven decisions.

Task: Analyze the provided JSON data and generate a structured report. The report should identify high-level trends, compare different course sections or modalities, highlight what is working well, and pinpoint areas for improvement.

Instructions:
1. Analyze Holistically: Review the entire dataset to understand the relationships between qualitative feedback (comments, themes, sentiment) and quantitative data (grades, enrollment, modality).
2. Structure Your Report: Generate the report in markdown format with the following sections:
    * ### Overall Summary: A brief, top-level paragraph summarizing the most significant findings from the data.
    * ### Key Strengths: Identify 2-3 aspects that are working well. Use positive-sentiment themes and specific examples as evidence. Mention if these strengths are consistent across all sections or concentrated in specific ones.
    * ### Areas for Improvement: Identify the 2-3 most critical areas needing attention. Use negative-sentiment themes and grade data (e.g., high DEW rates) as evidence. Quote specific, illustrative comments to support your points.
    * ### Patterns and Comparisons: Highlight any interesting patterns. For example:
        * Does one instructor consistently receive more positive feedback on a specific theme?
        * Is there a difference in feedback between "Online" and "In-Person" modalities?
        * Is there a correlation between high DEW (Drop/Fail/Withdraw) rates and negative course evaluations?
        * Does student feedback differ by course modality suggesting a preference or perceived effectiveness of one mode over another?
        * Does student feedback vary by session length (Session A/B vs. Session C), indicating a preference for 7.5-week vs. 16-week formats?
    * ### Actionable Recommendations: Based on your analysis, suggest 1-2 concrete, actionable steps or areas for future investigation. These should be based in the students comments as evidence.

Data for Analysis:
You will be provided with a JSON object representing the final, merged dataset. Here is the data:
{data_json}"""

        # Call Gemini API to generate the executive summary
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(executive_summary_prompt)
        
        executive_summary = response.text.strip()
        
        print(f"✅ Executive summary generated ({len(executive_summary)} characters)")
        return executive_summary
        
    except Exception as e:
        print(f"Warning: Failed to generate executive summary: {e}")
        return "## Executive Summary\n\nExecutive summary generation is currently unavailable. Please refer to the thematic analysis above for key insights."


def run_pipeline(
    comments_file_path: str,
    schedule_file_path: Optional[str] = None,
    grades_file_path: Optional[str] = None,
    question_col: str = "question",
    answer_col: str = "response"
) -> Dict[str, Any]:
    """
    Runs the entire analysis pipeline in-memory.

    Args:
        comments_file_path (str): Path to comments file
        schedule_file_path (str, optional): Path to schedule file
        grades_file_path (str, optional): Path to grades file
        question_col (str): Name of question column
        answer_col (str): Name of answer column

    Returns:
        Dict[str, Any]: Dictionary containing the final DataFrame, markdown report, and executive summary
    """
    print("🚀 Starting robust in-memory pipeline...")

    # Load API configuration
    if not load_config():
        raise Exception("Failed to load API configuration")

    # Phase 0: Clean the data
    print("\n📋 PHASE 0: DATA FORMATTING & CLEANING")
    print("-" * 50)
    
    cleaned_df = format_data(comments_file_path)
    print(f"✅ Phase 0 Complete: Cleaned data has {len(cleaned_df)} rows.")

    # Phase 1: Initial Coding
    print("\n🤖 PHASE 1: AI-ASSISTED INITIAL CODING")
    print("-" * 50)
    
    coded_df = process_csv_in_batches(cleaned_df, question_col, answer_col)
    print(f"✅ Phase 1 Complete: Coded data has {len(coded_df)} rows.")
    
    # Phase 2: Thematic Analysis
    print("\n🎯 PHASE 2: THEMATIC ANALYSIS")
    print("-" * 50)
    
    final_df, markdown_report = analyze_themes(coded_df)
    print(f"✅ Phase 2 Complete: Themed data has {len(final_df)} rows.")

    # Phase 3: Merge with other data sources
    print("\n🔗 PHASE 3: DATA INTEGRATION")
    print("-" * 50)
    
    validation_results = []
    
    if schedule_file_path and os.path.exists(schedule_file_path):
        try:
            schedule_df = pd.read_csv(schedule_file_path, on_bad_lines='skip')
            final_df, schedule_result = merge_with_schedule(final_df, schedule_df)
            validation_results.append({
                "file_type": "schedule",
                "result": schedule_result
            })
        except Exception as e:
            print(f"Warning: Failed to merge schedule data: {e}")
            validation_results.append({
                "file_type": "schedule",
                "result": {
                    "status": "error",
                    "message": f"Failed to process schedule data: {str(e)}",
                    "type": "processing_error"
                }
            })

    if grades_file_path and os.path.exists(grades_file_path):
        try:
            grades_df = pd.read_csv(grades_file_path, on_bad_lines='skip')
            final_df, grades_result = merge_with_grades(final_df, grades_df)
            validation_results.append({
                "file_type": "grades",
                "result": grades_result
            })
        except Exception as e:
            print(f"Warning: Failed to merge grades data: {e}")
            validation_results.append({
                "file_type": "grades",
                "result": {
                    "status": "error",
                    "message": f"Failed to process grades data: {str(e)}",
                    "type": "processing_error"
                }
            })
    
    # Clean up final data for JSON serialization
    final_df = clean_for_json_serialization(final_df)

    # Phase 4: Generate Executive Summary
    executive_summary = generate_executive_summary(final_df)

    print("\n🎉 Pipeline finished successfully!")
    print(f"Final dataset: {len(final_df)} rows, {len(final_df.columns)} columns")
    
    # Check for any warnings to report
    warnings = [vr for vr in validation_results if vr["result"]["status"] == "warning"]
    if warnings:
        print(f"\n⚠️  {len(warnings)} data validation warning(s) detected")
    
    return {
        "final_dataframe": final_df,
        "markdown_report": markdown_report,
        "executive_summary": executive_summary,
        "validation_results": validation_results
    }


def run_pipeline_legacy(input_file: str, skip_phase0: bool = False, skip_phase1: bool = False, 
                       skip_phase2: bool = False, output_dir: str = None, question_col: str = "question",
                       answer_col: str = "response") -> Dict[str, Any]:
    """
    Legacy pipeline function for backward compatibility.
    
    Args:
        input_file (str): Path to the raw input file
        skip_phase0 (bool): Skip Phase 0 (data cleaning)
        skip_phase1 (bool): Skip Phase 1 (initial coding)
        skip_phase2 (bool): Skip Phase 2 (thematic analysis)
        output_dir (str): Directory for output files
        question_col (str): Name of the question column
        answer_col (str): Name of the answer column
        
    Returns:
        dict: Results dictionary with file paths and analysis data
    """
    print("=" * 80)
    print("🔬 QUALITATIVE ANALYSIS PIPELINE (LEGACY MODE)")
    print("=" * 80)
    
    # For legacy mode, we'll run the new pipeline and save files
    try:
        results = run_pipeline(input_file, None, None, question_col, answer_col)
        
        # Save outputs for legacy compatibility
        final_df = results["final_dataframe"]
        markdown_report = results["markdown_report"]
        
        # Generate legacy output files
        from pathlib import Path
        input_path = Path(input_file)
        base_name = input_path.stem
        
        # Save markdown report
        report_path = f"{base_name}_themes_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        # Save themed CSV
        themed_path = f"{base_name}_themed.csv"
        final_df.to_csv(themed_path, index=False, quoting=1)
        
        return {
            "input_file": input_file,
            "coded_file": f"{base_name}_coded.csv",  # Would be generated in full legacy mode
            "analysis_file": themed_path,
            "themes": [],  # Would extract from markdown_report if needed
            "summary": markdown_report
        }
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    """
    Command-line interface for the pipeline.
    """
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <comments_file> [schedule_file] [grades_file]")
        print("Example: python pipeline.py comments.csv schedule.csv grades.csv")
        sys.exit(1)
    
    comments_file = sys.argv[1]
    schedule_file = sys.argv[2] if len(sys.argv) > 2 else None
    grades_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        results = run_pipeline(comments_file, schedule_file, grades_file)
        print(f"\nSuccess! Pipeline completed.")
        print(f"Final dataset shape: {results['final_dataframe'].shape}")
        print(f"Report length: {len(results['markdown_report'])} characters")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1) 