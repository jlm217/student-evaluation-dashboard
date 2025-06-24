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
from data_cleaner import clean_likert_file, analyze_response_rates, enhance_likert_analysis_with_response_rates
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


def create_hybrid_summary_json(final_df: pd.DataFrame, likert_analysis: Optional[Dict[str, Any]] = None) -> dict:
    """
    Create a hybrid JSON payload that efficiently combines qualitative and quantitative data.
    
    This function implements the PRD strategy of separating:
    1. qualitative_data: ALL comment records with essential columns
    2. quantitative_summary: Aggregated section-level data (unique sections only)
    3. likert_summary: Quantitative survey benchmarks per section (if available)
    
    Args:
        final_df (pd.DataFrame): Final merged DataFrame with all analysis data
        likert_analysis (Dict[str, Any], optional): Results from analyze_quantitative_questions()
        
    Returns:
        dict: Hybrid JSON object with qualitative_data and quantitative_summary (with likert_summary)
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
            
            # Add Likert survey analysis if available
            if likert_analysis and 'sections' in likert_analysis:
                section_id = int(section['SectionNumber_ASU']) if pd.notna(section['SectionNumber_ASU']) else None
                
                if section_id in likert_analysis['sections']:
                    likert_section_data = likert_analysis['sections'][section_id]
                    
                    # Create summarized Likert data for this section
                    likert_summary = {
                        'total_students_surveyed': likert_section_data['total_students'],
                        'questions_answered': likert_section_data['questions_answered'],
                        'key_metrics': {}
                    }
                    
                    # Add top performance metrics (questions with notable deviations)
                    questions_data = likert_section_data.get('questions', {})
                    notable_questions = []
                    
                    for question_key, q_data in questions_data.items():
                        deviation = q_data.get('deviation')
                        performance_tier = q_data.get('performance_tier', 'unknown')
                        
                        # Include questions with significant deviations or interesting patterns
                        if deviation is not None and (abs(deviation) > 0.3 or performance_tier != 'average'):
                            # Get question text from metadata if available
                            question_text = q_data.get('question_text')
                            if not question_text and likert_analysis and 'question_metadata' in likert_analysis:
                                question_text = likert_analysis['question_metadata'].get(question_key, {}).get('question_text')
                            
                            # Fallback to formatted key if no question text available
                            if not question_text:
                                question_text = question_key.replace('q_', '').replace('_', ' ').title()
                            
                            notable_questions.append({
                                'question_key': question_key,
                                'question_text': question_text,
                                'section_score': q_data['this_section_score'],
                                'peer_average': q_data['peer_group_average'],
                                'deviation': deviation,
                                'performance_tier': performance_tier,
                                'student_count': q_data['student_count']
                            })
                    
                    # Sort by absolute deviation (most significant first)
                    notable_questions.sort(key=lambda x: abs(x['deviation']), reverse=True)
                    
                    # Keep top 5 most notable questions to avoid payload bloat
                    likert_summary['notable_questions'] = notable_questions[:5]
                    
                    # Add summary statistics
                    if notable_questions:
                        avg_deviation = sum(abs(q['deviation']) for q in notable_questions) / len(notable_questions)
                        likert_summary['key_metrics'] = {
                            'average_deviation_magnitude': round(avg_deviation, 2),
                            'above_average_count': len([q for q in notable_questions if q['performance_tier'] == 'above_average']),
                            'below_average_count': len([q for q in notable_questions if q['performance_tier'] == 'below_average'])
                        }
                    
                    section_data['likert_summary'] = likert_summary
                else:
                    # Section exists in academic data but not in Likert data
                    section_data['likert_summary'] = {
                        'total_students_surveyed': 0,
                        'questions_answered': 0,
                        'key_metrics': {},
                        'notable_questions': [],
                        'note': 'No quantitative survey data available for this section'
                    }
            
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


def generate_executive_summary(final_df: pd.DataFrame, likert_analysis: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate an executive summary using the hybrid data model approach with optional quantitative survey integration.
    
    Uses create_hybrid_summary_json to create a token-efficient payload that combines
    all qualitative data with aggregated quantitative data and Likert survey benchmarks.
    
    Args:
        final_df (pd.DataFrame): Final merged DataFrame with all analysis data
        likert_analysis (Dict[str, Any], optional): Results from analyze_quantitative_questions()
        
    Returns:
        str: Executive summary as markdown text
    """
    try:
        print("\n📝 PHASE 4: EXECUTIVE SUMMARY GENERATION")
        print("-" * 50)
        
        # Phase 2.1: Create hybrid summary JSON with Likert integration
        hybrid_data = create_hybrid_summary_json(final_df, likert_analysis)
        
        # Convert to JSON string for the prompt
        data_json = json.dumps(hybrid_data, indent=2, default=str)
        
        print(f"   📏 Payload size: ~{len(data_json):,} characters")
        
        # Enhanced executive summary prompt with quantitative survey integration
        likert_instruction = ""
        if likert_analysis:
            likert_instruction = """
3. Quantitative Survey Analysis: Each section may include 'likert_summary' data with:
   - Student survey responses with peer group benchmarks
   - Performance tiers (above_average/average/below_average compared to peer sections)
   - Notable deviations from peer group averages
   - Use this data to validate or contrast with qualitative themes
   - **IMPORTANT**: When referencing survey questions, use the 'question_text' field (actual question text) instead of 'question_key' (variable names)
"""
        
        executive_summary_prompt = f"""

Role: You are an expert educational data analyst and strategist.

Objective: Produce a concise, decision-ready executive summary that fuses student feedback (qualitative_data) with course performance metrics (quantitative_summary). The primary audience is a busy department head who needs key insights and next steps in ≤ 2 pages. Assume a team of experts will have great insight as the look closer at your observations. Your role is to spot important patterns and trends and recommed steps that point them in the right direction.

Task: Analyze the provided hybrid JSON data and generate a structured report. The data contains:
1. qualitative_data: Complete student feedback with themes and sentiment analysis
2. quantitative_summary: Aggregated section-level performance metrics and grade distributions
3. Quantitative Survey Analysis: Each section may include 'likert_summary' data with:
   - Student survey responses with peer group benchmarks
   - Performance tiers (above_average/average/below_average compared to peer sections)
   - Notable deviations from peer group averages
   - Use this data to validate or contrast with qualitative themes
   - **IMPORTANT**: When referencing survey questions, use the 'question_text' field (actual question text) instead of 'question_key' (variable names)

Instructions:
1. Analyze Holistically: Review qualitative feedback, quantitative data, and quantitative survey results to understand correlations and patterns.
2. Look for Convergent Evidence: Where qualitative themes align with (or contradict) quantitative survey scores and academic performance.
3. **When referencing survey questions**: Always use the actual question text from the 'question_text' field, never the technical 'question_key' variables.
4. Compose Report (markdown): Generate the report in markdown format with the following sections. For each point being made in this report the typical structure should include a clear articulatulation of the point or main idea, reference to direct supporting evidence, and then implication for student learning/success. Your response should start at the Overall Summary:
    * ### Overall Summary: A brief, top-level paragraph summarizing the most significant findings from all data sources.
    * ### Key Strengths: 2–3 bullets, each containing: theme name, prevalence, use specific student question and response quotes, and supporting DEW or grade data. Include quantitative survey validation where available using actual question text.
    * ### Areas for Improvement: Identify the 2-3 most critical areas needing attention. Consider negative-sentiment themes, high DEW rates, and below-average survey scores as convergent evidence. Quote specific comments and cite relevant performance metrics. Use actual question text when referencing surveys.
    * ### Patterns and Comparisons: 3–5 sentences noting instructor, mode, or section contrasts. For example:
        * Do sections with positive feedback themes also show better grade distributions AND higher survey scores?
        * Is there correlation between instructor performance (DEW rates), student sentiment, AND quantitative survey ratings?
        * Does student feedback differ by modality and does this correlate with both academic outcomes AND survey responses?
        * Are there specific themes that correlate with higher or lower student success rates AND survey satisfaction?
    * ### Actionable Recommendations: Based on convergent evidence from all data sources, suggest 2-3 concrete, actionable steps supported by student comments, performance data, and survey benchmarks.

Format & tone: Neutral, student-centric, based in the evidence you have been provided; bold themes when used; When referring to specific section, use the Instruction_Mode to identify the course, (eg Smith’s In Person section; or a student from an Jones’ Online section); total length ≈ 400–600 words.

Data Structure:
- qualitative_data: Array of student feedback records with crs_number, SectionNumber_ASU, Instructor, Instruction_Mode, Term, question, response, Theme, Sentiment
- quantitative_summary: Array of section performance records with SectionNumber_ASU, Instructor, Instruction_Mode, Term, Total_Enrollment, DEW_Rate_Percent, Grade_Distribution, and optional likert_summary with notable_questions containing both 'question_key' (technical variable) and 'question_text' (actual question text - USE THIS)

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


def analyze_quantitative_questions(cleaned_likert_df: pd.DataFrame, question_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze quantitative survey questions to create peer group benchmarks.
    
    This function implements the core peer group benchmarking logic:
    1. For each question, identifies all sections that were asked that question
    2. Calculates peer group averages across all students in that group
    3. Calculates section-specific averages and response distributions
    4. Provides comparison metrics (how far above/below peer average)
    
    Args:
        cleaned_likert_df (pd.DataFrame): Wide-format Likert data from clean_likert_file()
        question_metadata (Dict[str, Any], optional): Question metadata with text and response labels
        
    Returns:
        Dict[str, Any]: Analysis results structured as:
            {
                'peer_group_averages': {question_key: average_score, ...},
                'question_metadata': {question_key: {question_text, response_labels}, ...},
                'sections': {
                    section_id: {
                        'questions': {
                            question_key: {
                                'question_text': str,
                                'response_labels': Dict[str, str],
                                'this_section_score': float,
                                'peer_group_average': float,
                                'deviation': float,  # this_section - peer_group
                                'response_distribution': {score: count, ...},
                                'student_count': int,
                                'performance_tier': str  # 'above_average', 'average', 'below_average'
                            }
                        }
                    }
                }
            }
    """
    print("🎯 Analyzing quantitative questions and creating peer group benchmarks...")
    
    if cleaned_likert_df.empty:
        print("   ⚠️  No Likert data available for analysis")
        return {
            'peer_group_averages': {},
            'question_metadata': question_metadata or {},
            'sections': {}
        }
    
    # Get all question columns (those starting with 'q_')
    question_columns = [col for col in cleaned_likert_df.columns if col.startswith('q_')]
    
    if not question_columns:
        print("   ⚠️  No question columns found in Likert data")
        return {
            'peer_group_averages': {},
            'question_metadata': question_metadata or {},
            'sections': {}
        }
    
    print(f"   Found {len(question_columns)} unique questions to analyze")
    
    # Step 1: Calculate peer group averages for each question
    print("   📊 Calculating peer group averages...")
    peer_group_averages = {}
    
    for question_col in question_columns:
        # Filter out missing responses (-1) and calculate mean
        valid_responses = cleaned_likert_df[cleaned_likert_df[question_col] >= 0][question_col]
        
        if len(valid_responses) > 0:
            peer_avg = valid_responses.mean()
            peer_group_averages[question_col] = round(peer_avg, 2)
            print(f"      {question_col}: {peer_avg:.2f} (n={len(valid_responses)})")
        else:
            peer_group_averages[question_col] = None
            print(f"      {question_col}: No valid responses")
    
    # Step 2: Section-level analysis
    print("   🏫 Analyzing individual sections...")
    sections_analysis = {}
    
    # Group by section
    section_groups = cleaned_likert_df.groupby('SectionNumber_ASU')
    
    for section_id, section_data in section_groups:
        section_id = int(section_id)  # Ensure consistent type
        print(f"      Section {section_id}: {len(section_data)} students")
        
        section_questions = {}
        
        for question_col in question_columns:
            # Get valid responses for this question in this section
            valid_responses = section_data[section_data[question_col] >= 0][question_col]
            
            if len(valid_responses) == 0:
                # This section wasn't asked this question
                continue
            
            # Calculate section-specific metrics
            this_section_score = valid_responses.mean()
            peer_group_average = peer_group_averages[question_col]
            
            if peer_group_average is not None:
                deviation = this_section_score - peer_group_average
                
                # Determine performance tier
                if deviation > 0.5:
                    performance_tier = 'above_average'
                elif deviation < -0.5:
                    performance_tier = 'below_average'
                else:
                    performance_tier = 'average'
            else:
                deviation = None
                performance_tier = 'unknown'
            
            # Calculate response distribution
            response_distribution = valid_responses.value_counts().to_dict()
            
            # Get question metadata if available
            question_text = None
            response_labels = None
            if question_metadata and question_col in question_metadata:
                question_text = question_metadata[question_col]['question_text']
                response_labels = question_metadata[question_col]['response_labels']
            
            section_questions[question_col] = {
                'question_text': question_text,
                'response_labels': response_labels,
                'this_section_score': round(this_section_score, 2),
                'peer_group_average': peer_group_average,
                'deviation': round(deviation, 2) if deviation is not None else None,
                'response_distribution': response_distribution,
                'student_count': len(valid_responses),
                'performance_tier': performance_tier
            }
        
        sections_analysis[section_id] = {
            'questions': section_questions,
            'total_students': len(section_data),
            'questions_answered': len(section_questions)
        }
    
    # Step 3: Generate summary statistics
    total_sections = len(sections_analysis)
    total_questions_analyzed = len([q for q in peer_group_averages.values() if q is not None])
    
    print(f"   ✅ Analysis complete:")
    print(f"      - {total_sections} sections analyzed")
    print(f"      - {total_questions_analyzed} questions with valid data")
    print(f"      - {len(cleaned_likert_df)} total student responses")
    
    return {
        'peer_group_averages': peer_group_averages,
        'question_metadata': question_metadata or {},
        'sections': sections_analysis,
        'summary': {
            'total_sections': total_sections,
            'total_questions': len(question_columns),
            'questions_with_data': total_questions_analyzed,
            'total_student_responses': len(cleaned_likert_df)
        }
    }


def run_pipeline(
    comments_file_path: str,
    schedule_file_path: Optional[str] = None,
    grades_file_path: Optional[str] = None,
    likert_file_path: Optional[str] = None,
    question_col: str = "question",
    answer_col: str = "response"
) -> Dict[str, Any]:
    """
    Runs the entire analysis pipeline in-memory with optional quantitative data integration.

    Args:
        comments_file_path (str): Path to comments file
        schedule_file_path (str, optional): Path to schedule file
        grades_file_path (str, optional): Path to grades file
        likert_file_path (str, optional): Path to Likert survey data file
        question_col (str): Name of question column
        answer_col (str): Name of answer column

    Returns:
        Dict[str, Any]: Dictionary containing the final DataFrame, markdown report, executive summary,
                       and quantitative analysis results (if Likert data provided)
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
    
    # Phase 3.5: Process Likert Survey Data (Quantitative Analysis)
    likert_analysis = None
    if likert_file_path and os.path.exists(likert_file_path):
        try:
            print("📊 Processing Likert survey data...")
            cleaned_likert_df, question_metadata = clean_likert_file(likert_file_path)
            
            # Perform response rate analysis
            response_analysis = analyze_response_rates(cleaned_likert_df)
            
            # Perform quantitative question analysis
            likert_analysis = analyze_quantitative_questions(cleaned_likert_df, question_metadata)
            
            # Enhance Likert analysis with response rate information
            likert_analysis = enhance_likert_analysis_with_response_rates(likert_analysis, response_analysis)
            
            validation_results.append({
                "file_type": "likert",
                "result": {
                    "status": "success",
                    "message": f"Successfully processed Likert data: {likert_analysis['summary']['total_sections']} sections, "
                              f"{likert_analysis['summary']['questions_with_data']} questions analyzed, "
                              f"{response_analysis.get('overall_stats', {}).get('total_unique_respondents', 0)} unique respondents",
                    "type": "successful_processing",
                    "details": {
                        **likert_analysis['summary'],
                        "response_rate_summary": response_analysis.get('overall_stats', {})
                    }
                }
            })
            print(f"✅ Likert analysis complete: {likert_analysis['summary']['total_sections']} sections analyzed")
            
        except Exception as e:
            print(f"Warning: Failed to process Likert data: {e}")
            validation_results.append({
                "file_type": "likert", 
                "result": {
                    "status": "error",
                    "message": f"Failed to process Likert data: {str(e)}",
                    "type": "processing_error"
                }
            })
    elif likert_file_path:
        print(f"Warning: Likert file path provided but file not found: {likert_file_path}")
        validation_results.append({
            "file_type": "likert",
            "result": {
                "status": "error", 
                "message": f"Likert file not found: {likert_file_path}",
                "type": "file_not_found"
            }
        })
    
    # Clean up final data for JSON serialization
    final_df = clean_for_json_serialization(final_df)

    # Phase 4: Generate Executive Summary
    executive_summary = generate_executive_summary(final_df, likert_analysis)

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
        "likert_analysis": likert_analysis,
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
        results = run_pipeline(input_file, None, None, None, question_col, answer_col)
        
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
    likert_file_path: Optional[str] = None,
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
        
        # Phase 3.5: Process Likert Survey Data (Quantitative Analysis)
        likert_analysis = None
        if likert_file_path and os.path.exists(likert_file_path):
            try:
                await progress_manager.send_progress(job_id, "quantitative", "in_progress", "Processing quantitative survey data...")
                print("📊 Processing Likert survey data...")
                cleaned_likert_df, question_metadata = clean_likert_file(likert_file_path)
                
                # Perform response rate analysis
                response_analysis = analyze_response_rates(cleaned_likert_df)
                
                # Perform quantitative question analysis
                likert_analysis = analyze_quantitative_questions(cleaned_likert_df, question_metadata)
                
                # Enhance Likert analysis with response rate information
                likert_analysis = enhance_likert_analysis_with_response_rates(likert_analysis, response_analysis)
                
                validation_results.append({
                    "file_type": "likert",
                    "result": {
                        "status": "success",
                        "message": f"Successfully processed Likert data: {likert_analysis['summary']['total_sections']} sections, "
                                  f"{likert_analysis['summary']['questions_with_data']} questions analyzed, "
                                  f"{response_analysis.get('overall_stats', {}).get('total_unique_respondents', 0)} unique respondents",
                        "type": "successful_processing",
                        "details": {
                            **likert_analysis['summary'],
                            "response_rate_summary": response_analysis.get('overall_stats', {})
                        }
                    }
                })
                print(f"✅ Likert analysis complete: {likert_analysis['summary']['total_sections']} sections analyzed")
                await progress_manager.send_progress(job_id, "quantitative", "complete", "Quantitative analysis completed")
                
            except Exception as e:
                print(f"Warning: Failed to process Likert data: {e}")
                validation_results.append({
                    "file_type": "likert", 
                    "result": {
                        "status": "error",
                        "message": f"Failed to process Likert data: {str(e)}",
                        "type": "processing_error"
                    }
                })
        
        # Phase 4: Executive Summary
        print("\n📋 PHASE 4: EXECUTIVE SUMMARY GENERATION")
        print("-" * 50)
        
        executive_summary = generate_executive_summary(final_df, likert_analysis)
        await progress_manager.send_progress(job_id, "summary", "complete", "Executive summary generated")
        
        print("✨ Pipeline completed successfully!")
        
        return {
            "final_dataframe": final_df,
            "markdown_report": markdown_report,
            "executive_summary": executive_summary,
            "validation_results": validation_results,
            "ai_analysis_memo": ai_analysis_memo,
            "likert_analysis": likert_analysis,
            "status": "success"
        }
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        print(f"❌ {error_msg}")
        await progress_manager.send_progress(job_id, "error", "error", error_msg)
        raise e 