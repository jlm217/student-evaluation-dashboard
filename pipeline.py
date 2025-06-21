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
import asyncio


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


def create_hybrid_summary_json(final_df: pd.DataFrame) -> dict:
    """
    Create a hybrid JSON payload that efficiently combines qualitative and quantitative data.
    
    This function implements the PRD strategy of separating:
    1. qualitative_data: ALL comment records with essential columns
    2. quantitative_summary: Aggregated section-level data (unique sections only)
    
    Args:
        final_df (pd.DataFrame): Final merged DataFrame with all analysis data
        
    Returns:
        dict: Hybrid JSON object with qualitative_data and quantitative_summary
    """
    print("📊 Creating hybrid summary JSON payload...")
    
    # Phase 1.2: Build qualitative_data payload
    qualitative_columns = [
        'crs_number', 'SectionNumber_ASU', 'Instructor', 'Instruction_Mode', 
        'Term', 'question', 'response', 'Theme', 'Sentiment'
    ]
    
    # Select available qualitative columns
    available_qual_cols = [col for col in qualitative_columns if col in final_df.columns]
    
    # Handle column name variations
    column_mapping = {
        'Instruction Mode': 'Instruction_Mode',
        'Modality': 'Instruction_Mode',
        'Term/Session': 'Term'
    }
    
    # Apply column mapping if needed
    for old_col, new_col in column_mapping.items():
        if old_col in final_df.columns and new_col not in final_df.columns:
            final_df[new_col] = final_df[old_col]
            if new_col not in available_qual_cols and new_col in qualitative_columns:
                available_qual_cols.append(new_col)
    
    print(f"   📝 Qualitative columns: {available_qual_cols}")
    
    # Create qualitative data
    qualitative_data = final_df[available_qual_cols].to_dict('records')
    
    # Phase 1.3: Build quantitative_summary payload
    quantitative_columns = [
        'SectionNumber_ASU', 'Instructor', 'Instruction_Mode', 'Term', 
        'Total_Enrollment', 'A', 'B', 'C', 'D', 'E', 'W'
    ]
    
    # Handle column name variations for quantitative data
    quant_column_mapping = {
        'Total Enrollment': 'Total_Enrollment',
        'Instruction Mode': 'Instruction_Mode',
        'Modality': 'Instruction_Mode',
        'Term/Session': 'Term'
    }
    
    # Apply quantitative column mapping
    for old_col, new_col in quant_column_mapping.items():
        if old_col in final_df.columns and new_col not in final_df.columns:
            final_df[new_col] = final_df[old_col]
    
    # Select available quantitative columns
    available_quant_cols = [col for col in quantitative_columns if col in final_df.columns]
    
    print(f"   📊 Quantitative columns: {available_quant_cols}")
    
    # Create unique sections dataframe
    if available_quant_cols:
        sections_df = final_df[available_quant_cols].drop_duplicates(subset=['SectionNumber_ASU'])
        
        # Calculate DEW rate and grade distribution for each section
        quantitative_summary = []
        
        for _, section in sections_df.iterrows():
            section_data = {
                'SectionNumber_ASU': int(section['SectionNumber_ASU']) if pd.notna(section['SectionNumber_ASU']) else None,
                'Instructor': section.get('Instructor', ''),
                'Instruction_Mode': section.get('Instruction_Mode', ''),
                'Term': section.get('Term', ''),
                'Total_Enrollment': int(section.get('Total_Enrollment', 0)) if pd.notna(section.get('Total_Enrollment', 0)) else 0
            }
            
            # Calculate grade distribution
            grade_columns = ['A', 'B', 'C', 'D', 'E', 'W']
            grade_distribution = {}
            total_graded = 0
            
            for grade in grade_columns:
                if grade in section and pd.notna(section[grade]):
                    count = int(section[grade])
                    grade_distribution[grade] = count
                    total_graded += count
                else:
                    grade_distribution[grade] = 0
            
            # Calculate DEW rate (D, E, W grades)
            dew_count = grade_distribution.get('D', 0) + grade_distribution.get('E', 0) + grade_distribution.get('W', 0)
            dew_rate = (dew_count / total_graded * 100) if total_graded > 0 else 0
            
            section_data['DEW_Rate_Percent'] = round(dew_rate, 1)
            section_data['Grade_Distribution'] = grade_distribution
            
            quantitative_summary.append(section_data)
    else:
        quantitative_summary = []
    
    # Phase 1.4: Assemble final hybrid object
    hybrid_payload = {
        'qualitative_data': qualitative_data,
        'quantitative_summary': quantitative_summary
    }
    
    print(f"   ✅ Hybrid payload created:")
    print(f"      📝 Qualitative records: {len(qualitative_data)}")
    print(f"      📊 Quantitative sections: {len(quantitative_summary)}")
    
    return hybrid_payload


def generate_executive_summary(final_df: pd.DataFrame) -> str:
    """
    Generate an executive summary using the hybrid data model approach.
    
    Uses create_hybrid_summary_json to create a token-efficient payload that combines
    all qualitative data with aggregated quantitative data.
    
    Args:
        final_df (pd.DataFrame): Final merged DataFrame with all analysis data
        
    Returns:
        str: Executive summary as markdown text
    """
    try:
        print("\n📝 PHASE 4: EXECUTIVE SUMMARY GENERATION")
        print("-" * 50)
        
        # Phase 2.1: Create hybrid summary JSON
        hybrid_data = create_hybrid_summary_json(final_df)
        
        # Convert to JSON string for the prompt
        data_json = json.dumps(hybrid_data, indent=2, default=str)
        
        print(f"   📏 Payload size: ~{len(data_json):,} characters")
        
        # Create the updated executive summary prompt for hybrid data
        executive_summary_prompt = f"""Role: You are an expert educational data analyst and strategist.
Objective: Produce a concise, decision-ready executive summary that fuses student feedback (qualitative_data) with course performance metrics (quantitative_summary). The primary audience is a busy department head who needs key insights and next steps in ≤ 2 pages. Assume a team of experts will have great insight as the look closer at your observations. Your role is to spot important patterns and trends and recommed steps that point them in the right direction.

Task: Analyze the provided hybrid JSON data and generate a structured report. The data contains two parts:
1. qualitative_data: Complete student feedback with themes and sentiment analysis
2. quantitative_summary: Aggregated section-level performance metrics and grade distributions

Instructions:
1. Analyze Holistically: Review both qualitative feedback and quantitative data to understand correlations and patterns.
2. Compose Report (markdown): Generate the report in markdown format with the following sections (your response should start at the Overall Summary):
    * ### Overall Summary: A brief, top-level paragraph summarizing the most significant findings from both qualitative and quantitative data.
    * ### Key Strengths: 2–3 bullets, each containing: theme name, prevalence, use specific student question and response quotes, and supporting DEW or grade data.
    * ### Areas for Improvement: Identify the 2-3 most critical areas needing attention. Consider negative-sentiment themes and high DEW rates as evidence. Quote specific, illustrative comments and cite relevant performance metrics.
    * ### Patterns and Comparisons: 3–5 sentences (or a mini-table) noting instructor, mode, or section contrasts. For example:
        * Do sections with positive feedback themes also show better grade distributions?
        * Is there a correlation between instructor performance (DEW rates) and student sentiment?
        * Does student feedback differ by modaility (Online vs In-Person) and does this correlate with academic outcomes?
        * Are there specific themes that correlate with higher or lower student success rates?
    * ### Actionable Recommendations: Based on your analysis, suggest 2-3 concrete, actionable steps supported by both student comments and performance data.
Format & tone: Neutral, student-centric, evidence-first; avoid jargon; bold themes when used; attibute quote in text citation; use section-level end notes for other citations; total length ≈ 400–600 words.

Data Structure:
- qualitative_data: Array of student feedback records with crs_number, SectionNumber_ASU, Instructor, Instruction_Mode, Term, question, response, Theme, Sentiment
- quantitative_summary: Array of section performance records with SectionNumber_ASU, Instructor, Instruction_Mode, Term, Total_Enrollment, DEW_Rate_Percent, Grade_Distribution

Data for Analysis:
{data_json}"""

        # Call Gemini API with updated model
        model = genai.GenerativeModel('gemini-2.5-flash')
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
    
    final_df, markdown_report, themes_data = analyze_themes(coded_df)
    ai_memo = themes_data.get('memo', '') if themes_data else ''
    ai_analysis_memo = themes_data.get('ai_analysis_memo', '') if themes_data else ''
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
        "validation_results": validation_results,
        "ai_analysis_memo": ai_analysis_memo,
        "status": "success"
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


async def run_pipeline_with_progress(
    job_id: str,
    comments_file_path: str,
    schedule_file_path: Optional[str] = None,
    grades_file_path: Optional[str] = None,
    question_col: str = "question",
    answer_col: str = "response"
) -> Dict[str, Any]:
    """
    Enhanced pipeline that sends progress updates via WebSocket.
    This is the async version of run_pipeline for real-time tracking.
    """
    from app import progress_manager  # Import here to avoid circular imports
    
    try:
        print("🚀 Starting pipeline with progress tracking...")
        
        # Load and validate environment
        if not load_config():
            raise Exception("Failed to load configuration")
        
        # === PHASE 0: DATA FORMATTING & CLEANING ===
        await progress_manager.send_progress(job_id, "cleaning", "in_progress", "Loading and cleaning data...")
        print("📊 Phase 0: Data Formatting & Cleaning")
        
        comments_df = pd.read_csv(comments_file_path)
        print(f"✅ Loaded comments file: {len(comments_df)} rows")
        
        schedule_df = None
        if schedule_file_path and os.path.exists(schedule_file_path):
            schedule_df = pd.read_csv(schedule_file_path)
            print(f"✅ Loaded schedule file: {len(schedule_df)} rows")
        
        grades_df = None
        if grades_file_path and os.path.exists(grades_file_path):
            grades_df = pd.read_csv(grades_file_path)
            print(f"✅ Loaded grades file: {len(grades_df)} rows")
        
        # Format the data - just use the comments file path since format_data works with files
        formatted_df = format_data(comments_file_path)
        await progress_manager.send_progress(job_id, "cleaning", "complete", "Data formatting completed")
        
        # === PHASE 1: INITIAL CODING ===
        await progress_manager.send_progress(job_id, "coding", "in_progress", "Generating AI-powered codes and sentiment analysis...")
        print("🔥 Phase 1: Initial Coding with AI")
        
        # Apply context-aware initial coding using AI with real-time progress and batch summaries
        import math
        import time
        from main import get_codes_for_batch_with_context, update_context_memory, CONTEXT_MEMORY_THRESHOLD
        
        batch_size = 20
        total_batches = math.ceil(len(formatted_df) / batch_size)
        
        print(f"Processing {len(formatted_df)} rows in {total_batches} batches of {batch_size}")
        print(f"Context-aware coding enabled (threshold: {CONTEXT_MEMORY_THRESHOLD} codes)")
        
        all_codes = []
        recently_generated_codes = []  # Context memory for consistent coding
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(formatted_df))
            batch_df = formatted_df.iloc[start_idx:end_idx]
            
            # Send batch progress update in real-time
            batch_percentage = int((batch_num / total_batches) * 100)
            batch_message = f"Processing batch {batch_num + 1} of {total_batches} ({batch_percentage}%)"
            await progress_manager.send_progress(job_id, "coding", "in_progress", batch_message)
            print(f"Batch Progress: {batch_message}")
            
            # Determine whether to use context
            use_context = len(recently_generated_codes) >= CONTEXT_MEMORY_THRESHOLD
            context_codes = recently_generated_codes if use_context else None
            
            if use_context:
                print(f"  Using context memory ({len(recently_generated_codes)} codes available)")
            
            # Process this batch with context awareness
            batch_result = get_codes_for_batch_with_context(batch_df, question_col, answer_col, context_codes)
            batch_codes = batch_result['analysis']
            batch_summary = batch_result['batch_summary']
            
            # Stream batch summary to frontend (ephemeral UI update)
            if batch_summary:
                summary_message = f"💭 {batch_summary}"
                await progress_manager.send_progress(job_id, "coding", "batch_summary", summary_message)
                print(f"Batch Summary: {batch_summary}")
            
            all_codes.extend(batch_codes)
            
            # Update context memory with new codes for next batch
            new_codes = [item.get('code') for item in batch_codes if item.get('code')]
            recently_generated_codes = update_context_memory(recently_generated_codes, new_codes)
            
            print(f"  Batch {batch_num + 1}: Generated {len(batch_codes)} codes, memory size: {len(recently_generated_codes)}")
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
        
        # Send completion update with final stats
        unique_codes = len(set(recently_generated_codes))
        total_comments = len(formatted_df)
        uniqueness_percentage = (unique_codes / total_comments) * 100 if total_comments > 0 else 0
        
        completion_message = f"Completed all {total_batches} batches - {unique_codes} unique codes generated ({uniqueness_percentage:.1f}% uniqueness)"
        await progress_manager.send_progress(job_id, "coding", "in_progress", completion_message)
        print(f"Batch Progress: {completion_message}")
        
        # Create the coded DataFrame
        analysis_df = pd.DataFrame(all_codes)
        if not analysis_df.empty:
            analysis_df = analysis_df.set_index('id')
            coded_df = formatted_df.copy()
            coded_df['Initial_Code'] = analysis_df['code']
            coded_df['Sentiment'] = analysis_df['sentiment']
        else:
            coded_df = formatted_df.copy()
            coded_df['Initial_Code'] = None
            coded_df['Sentiment'] = None
        await progress_manager.send_progress(job_id, "coding", "complete", "Initial coding completed")
        
        # === PHASE 2: THEMATIC ANALYSIS ===
        await progress_manager.send_progress(job_id, "themes", "in_progress", "Synthesizing themes and generating insights...")
        print("🎯 Phase 2: Thematic Analysis")
        
        # Generate themes with enhanced methodology
        final_df, markdown_report, themes_data = analyze_themes(coded_df, job_id=job_id)
        ai_memo = themes_data.get('memo', '') if themes_data else ''
        ai_analysis_memo = themes_data.get('ai_analysis_memo', '') if themes_data else ''
        await progress_manager.send_progress(job_id, "themes", "complete", "Methodologically-aligned thematic analysis completed")
        
        # === PHASE 3: MERGING WITH ADDITIONAL DATA ===
        validation_results = []
        
        if schedule_df is not None:
            print("📅 Merging with schedule data...")
            final_df, schedule_validation = merge_with_schedule(final_df, schedule_df)
            validation_results.append({
                "file_type": "schedule",
                "result": schedule_validation
            })
        
        if grades_df is not None:
            print("📈 Merging with grades data...")
            final_df, grades_validation = merge_with_grades(final_df, grades_df)
            validation_results.append({
                "file_type": "grades",
                "result": grades_validation
            })
        
        # Clean up final data for JSON serialization
        final_df = clean_for_json_serialization(final_df)
        
        # === PHASE 4: EXECUTIVE SUMMARY ===
        await progress_manager.send_progress(job_id, "summary", "in_progress", "Creating executive summary...")
        print("📋 Phase 4: Executive Summary Generation")
        
        executive_summary = generate_executive_summary(final_df)
        await progress_manager.send_progress(job_id, "summary", "complete", "Executive summary generated")
        
        print("✨ Pipeline completed successfully!")
        
        return {
            "final_dataframe": final_df,
            "markdown_report": markdown_report,
            "executive_summary": executive_summary,
            "validation_results": validation_results,
            "ai_analysis_memo": ai_analysis_memo,
            "status": "success"
        }
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        print(f"❌ {error_msg}")
        await progress_manager.send_progress(job_id, "error", "error", error_msg)
        raise e 