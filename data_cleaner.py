#!/usr/bin/env python3
"""
Data Cleaning Script for Qualitative Coding Agent

This script standardizes column names across different data files to enable proper merging:
- Comments file: SectionNumber_ASU (already standardized)
- Schedule file: Number -> SectionNumber_ASU  
- Grades file: Class Nbr -> SectionNumber_ASU

NEW: Quantitative Survey Data (Likert) cleaning and transformation

Usage:
    python data_cleaner.py --schedule schedule.csv --grades grades.csv --output-dir cleaned/
"""

import pandas as pd
import argparse
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any


def clean_schedule_file(input_path: str, output_path: str) -> None:
    """
    Clean schedule file by standardizing column names and data.
    
    Key changes:
    - Number -> SectionNumber_ASU
    - Instructor(s) -> Instructor
    - Extract modality from Location/Days columns
    """
    print(f"📋 Cleaning schedule file: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Rename key columns
    column_mapping = {
        'Number': 'SectionNumber_ASU',
        'Instructor(s)': 'Instructor',
        'Title': 'Course_Title',
        'Course': 'Course_Code'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Determine modality based on Location column
    def determine_modality(location, days):
        if pd.isna(location) or location == '':
            return 'Online'
        elif 'iCourse' in str(location):
            return 'iCourse'
        elif 'ASU Online' in str(location):
            return 'ASU Online'
        elif 'Tempe' in str(location):
            return 'In-Person'
        else:
            return 'Unknown'
    
    df['Modality'] = df.apply(lambda row: determine_modality(row.get('Location', ''), row.get('Days', '')), axis=1)
    
    # Extract term information from Dates column
    def extract_term(dates_str):
        if pd.isna(dates_str):
            return 'Unknown'
        dates_str = str(dates_str)
        if '8/17' in dates_str and '12/1' in dates_str:
            return 'Fall 2023'
        elif '1/9' in dates_str and '4/28' in dates_str:
            return 'Spring 2023'
        else:
            return 'Unknown'
    
    df['Term'] = df['Dates'].apply(extract_term)
    
    # Convert SectionNumber_ASU to integer for consistency
    df['SectionNumber_ASU'] = pd.to_numeric(df['SectionNumber_ASU'], errors='coerce')
    
    # Select and reorder columns for output
    output_columns = [
        'SectionNumber_ASU', 'Course_Code', 'Course_Title', 'Instructor', 
        'Modality', 'Term', 'Days', 'Start', 'End', 'Location', 'Units'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in output_columns if col in df.columns]
    df_clean = df[available_columns].copy()
    
    # Remove rows where SectionNumber_ASU is NaN
    df_clean = df_clean.dropna(subset=['SectionNumber_ASU'])
    
    # Save cleaned file
    df_clean.to_csv(output_path, index=False)
    print(f"   ✅ Saved cleaned schedule file: {output_path}")
    print(f"   📊 Output: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    print(f"   🔑 SectionNumber_ASU range: {df_clean['SectionNumber_ASU'].min():.0f} - {df_clean['SectionNumber_ASU'].max():.0f}")


def clean_grades_file(input_path: str, output_path: str) -> None:
    """
    Clean grades file by standardizing column names and data.
    
    Key changes:
    - Class Nbr -> SectionNumber_ASU
    - Primary Instructor -> Instructor
    - Calculate grade percentages
    """
    print(f"📊 Cleaning grades file: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Rename key columns
    column_mapping = {
        'Class Nbr': 'SectionNumber_ASU',
        'Primary Instructor': 'Instructor',
        'Instruction Mode': 'Modality',
        'Term/Session': 'Term'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Convert SectionNumber_ASU to integer for consistency
    df['SectionNumber_ASU'] = pd.to_numeric(df['SectionNumber_ASU'], errors='coerce')
    
    # Calculate grade percentages
    grade_columns = ['A', 'B', 'C', 'D', 'E', 'EN', 'EU', 'W', 'X']
    
    # Fill NaN values with 0 for grade calculations
    for col in grade_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate total graded students (excluding EN, EU, X if they exist)
    graded_columns = ['A', 'B', 'C', 'D', 'E', 'W']
    existing_graded_cols = [col for col in graded_columns if col in df.columns]
    
    if existing_graded_cols:
        df['Total_Graded'] = df[existing_graded_cols].sum(axis=1)
        
        # Calculate percentages
        for grade in existing_graded_cols:
            df[f'{grade}_Percent'] = (df[grade] / df['Total_Graded'] * 100).round(1)
    
    # Clean instructor names (remove quotes, standardize format)
    if 'Instructor' in df.columns:
        df['Instructor'] = df['Instructor'].str.replace('"', '').str.strip()
        # Convert "Last,First" to "First Last" format
        def clean_instructor_name(name):
            if pd.isna(name):
                return name
            if ',' in str(name):
                parts = str(name).split(',')
                if len(parts) >= 2:
                    last = parts[0].strip()
                    first = parts[1].strip()
                    return f"{first} {last}"
            return str(name)
        
        df['Instructor'] = df['Instructor'].apply(clean_instructor_name)
    
    # Standardize modality names
    if 'Modality' in df.columns:
        modality_mapping = {
            'In-Person': 'In-Person',
            'Hybrid': 'Hybrid', 
            'I-Course': 'iCourse',
            'ASU Online': 'ASU Online',
            'Online': 'Online'
        }
        df['Modality'] = df['Modality'].map(modality_mapping).fillna(df['Modality'])
    
    # Select and reorder columns for output
    base_columns = ['SectionNumber_ASU', 'Course', 'Session', 'Term', 'Modality', 'Instructor', 'Total Enrollment']
    grade_cols = [col for col in df.columns if col in grade_columns or col.endswith('_Percent') or col == 'Total_Graded']
    
    output_columns = base_columns + grade_cols
    available_columns = [col for col in output_columns if col in df.columns]
    df_clean = df[available_columns].copy()
    
    # Remove rows where SectionNumber_ASU is NaN
    df_clean = df_clean.dropna(subset=['SectionNumber_ASU'])
    
    # Save cleaned file
    df_clean.to_csv(output_path, index=False)
    print(f"   ✅ Saved cleaned grades file: {output_path}")
    print(f"   📊 Output: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    print(f"   🔑 SectionNumber_ASU range: {df_clean['SectionNumber_ASU'].min():.0f} - {df_clean['SectionNumber_ASU'].max():.0f}")


def clean_comments_file(input_path: str, output_path: str) -> None:
    """
    Clean comments file by ensuring SectionNumber_ASU is standardized.
    
    The comments file should already have SectionNumber_ASU, but we ensure
    it's properly formatted and consistent.
    """
    print(f"💬 Cleaning comments file: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Convert SectionNumber_ASU to integer for consistency
    if 'SectionNumber_ASU' in df.columns:
        df['SectionNumber_ASU'] = pd.to_numeric(df['SectionNumber_ASU'], errors='coerce')
        # Remove rows where SectionNumber_ASU is NaN
        df = df.dropna(subset=['SectionNumber_ASU'])
    else:
        print("   ⚠️  Warning: SectionNumber_ASU column not found in comments file")
        return
    
    # Save cleaned file
    df.to_csv(output_path, index=False)
    print(f"   ✅ Saved cleaned comments file: {output_path}")
    print(f"   📊 Output: {len(df)} rows, {len(df.columns)} columns")
    print(f"   🔑 SectionNumber_ASU range: {df['SectionNumber_ASU'].min():.0f} - {df['SectionNumber_ASU'].max():.0f}")


def validate_merge_compatibility(comments_path: str, schedule_path: str, grades_path: str) -> None:
    """
    Validate that the cleaned files can be properly merged by checking
    the overlap of SectionNumber_ASU values.
    """
    print(f"\n🔍 Validating merge compatibility...")
    
    # Load cleaned files
    comments_df = pd.read_csv(comments_path) if os.path.exists(comments_path) else None
    schedule_df = pd.read_csv(schedule_path) if os.path.exists(schedule_path) else None
    grades_df = pd.read_csv(grades_path) if os.path.exists(grades_path) else None
    
    if comments_df is not None and 'SectionNumber_ASU' in comments_df.columns:
        comments_sections = set(comments_df['SectionNumber_ASU'].dropna().astype(int))
        print(f"   💬 Comments: {len(comments_sections)} unique sections")
    else:
        comments_sections = set()
    
    if schedule_df is not None and 'SectionNumber_ASU' in schedule_df.columns:
        schedule_sections = set(schedule_df['SectionNumber_ASU'].dropna().astype(int))
        print(f"   📋 Schedule: {len(schedule_sections)} unique sections")
    else:
        schedule_sections = set()
    
    if grades_df is not None and 'SectionNumber_ASU' in grades_df.columns:
        grades_sections = set(grades_df['SectionNumber_ASU'].dropna().astype(int))
        print(f"   📊 Grades: {len(grades_sections)} unique sections")
    else:
        grades_sections = set()
    
    # Check overlaps
    if comments_sections and schedule_sections:
        schedule_overlap = comments_sections.intersection(schedule_sections)
        print(f"   🔗 Comments ∩ Schedule: {len(schedule_overlap)} sections")
        if schedule_overlap:
            print(f"       Sample overlapping sections: {sorted(list(schedule_overlap))[:5]}")
    
    if comments_sections and grades_sections:
        grades_overlap = comments_sections.intersection(grades_sections)
        print(f"   🔗 Comments ∩ Grades: {len(grades_overlap)} sections")
        if grades_overlap:
            print(f"       Sample overlapping sections: {sorted(list(grades_overlap))[:5]}")
    
    if schedule_sections and grades_sections:
        schedule_grades_overlap = schedule_sections.intersection(grades_sections)
        print(f"   🔗 Schedule ∩ Grades: {len(schedule_grades_overlap)} sections")


def clean_likert_file(input_path: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and transform Likert survey data from long format to wide format.
    
    This function handles the quantitative survey data integration by:
    1. Loading the raw CSV data
    2. Converting text responses to numerical values
    3. Standardizing question text into machine-readable column headers
    4. Pivoting from long format (one row per student per question) to wide format
    5. Handling missing data gracefully
    6. Creating question metadata with original text and response labels
    
    Args:
        input_path (str): Path to the raw Likert data CSV file
        
    Returns:
        tuple: (cleaned_df, question_metadata)
            - cleaned_df: DataFrame in wide format with columns:
                - SectionNumber_ASU: Section identifier
                - DummyID: Student identifier (for merging with comments)
                - q_[question_key]: Numerical responses for each question
            - question_metadata: Dict with question_key -> question info mapping
    """
    print(f"🔢 Cleaning Likert survey data: {input_path}")
    
    # Load the raw data
    try:
        df = pd.read_csv(input_path)
        print(f"   Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        raise Exception(f"Failed to load Likert data file: {e}")
    
    # Validate required columns
    required_columns = ['SectionNumber_ASU', 'QUESTION', 'responsevalue', 'DummyID']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise Exception(f"Missing required columns in Likert data: {missing_columns}")
    
    print(f"   Found {df['SectionNumber_ASU'].nunique()} unique sections")
    print(f"   Found {df['QUESTION'].nunique()} unique questions")
    print(f"   Found {df['DummyID'].nunique()} unique respondents")
    
    # Create mapping dictionary for text responses to numerical values
    response_mapping = {
        # 5-Point Likert Scale (High to Low)
        'Strongly Agree': 5,
        'Agree': 4,
        'Neither Disagree nor Agree': 3,
        'Neither Agree nor Disagree': 3,  # Alternative phrasing
        'Neutral': 3,
        'Disagree': 2,
        'Strongly Disagree': 1,
        
        # Alternative 5-Point Scale (Strongly Agree to Strongly Disagree)
        'Strongly agree': 5,  # Case variations
        'agree': 4,
        'neutral': 3,
        'disagree': 2,
        'Strongly disagree': 1,
        
        # Rating Scales (Very Good to Very Poor)
        'Very Good': 5,
        'Good': 4,
        'Average': 3,
        'Poor': 2,
        'Very Poor': 1,
        
        # Effort/Frequency Scales
        'Much Higher': 5,
        'Higher': 4,
        'About the Same': 3,
        'Lower': 2,
        'Much Lower': 1,
        
        # Access Frequency
        'Five or more': 5,
        'Four times a week': 4,
        'Three times a week': 3,
        'Twice a week': 2,
        'Once a week': 1,
        
        # Yes/No Questions
        'Yes': 1,
        'No': 0,
        
        # Grade Expectations (A=5 to F=1)
        'A': 5,
        'B': 4,
        'C': 3,
        'D': 2,
        'E': 1,
        'F': 1,
        
        # Required/Elective
        'Required': 1,
        'Elective': 0,
        
        # Course Learning
        'Advisor': 2,
        'Friend': 1,
        'Other': 0,
        
        # Working Hours
        'I am not working': 0,
        '1-10 hours': 1,
        '11-20 hours': 2,
        '21-30 hours': 3,
        '31-40 hours': 4,
        'More than 40 hours': 5,
        
        # Numeric string responses (already in proper format)
        '1': 1,
        '2': 2, 
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,  # Some scales might go higher
        '0': 0,
    }
    
    # Apply response mapping, but also handle cases where responses are already numeric
    print("   Converting text responses to numerical values...")
    
    def convert_response(response_value):
        """Convert response to numeric, handling both text and already-numeric values."""
        if pd.isna(response_value):
            return None
            
        # First try the mapping dictionary
        if response_value in response_mapping:
            return response_mapping[response_value]
            
        # If not in mapping, try to convert directly to numeric
        try:
            numeric_val = float(response_value)
            # Ensure it's a reasonable range (0-6 for most scales)
            if 0 <= numeric_val <= 10:  # Allow some flexibility
                return int(numeric_val)
        except (ValueError, TypeError):
            pass
            
        # If all else fails, return None (will be caught as unmapped)
        return None
    
    df['numeric_response'] = df['responsevalue'].apply(convert_response)
    
    # Check for unmapped responses
    unmapped_responses = df[df['numeric_response'].isna()]['responsevalue'].unique()
    if len(unmapped_responses) > 0:
        print(f"   ⚠️  Warning: Found {len(unmapped_responses)} unmapped response values:")
        for response in unmapped_responses[:10]:  # Show first 10
            print(f"      '{response}'")
        if len(unmapped_responses) > 10:
            print(f"      ... and {len(unmapped_responses) - 10} more")

    # Function to determine response labels based on the question and actual responses
    def determine_response_labels(question_text: str, response_values: set) -> Dict[str, str]:
        """
        Determine appropriate response labels for a question based on its text and actual response values.
        """
        question_lower = question_text.lower()
        unique_numeric = {v for v in response_values if pd.notna(v) and v >= 0}
        
        # Check if this is a grade expectation question
        if any(keyword in question_lower for keyword in ['grade', 'expect to earn']):
            return {
                '5': 'A',
                '4': 'B', 
                '3': 'C',
                '2': 'D',
                '1': 'E'
            }
        
        # Check if this is a yes/no question based on question text
        elif any(keyword in question_lower for keyword in ['did you', 'do you', 'have you', 'are you', 'go to']):
            # For yes/no questions, check if responses are primarily binary (even if on 1-5 scale)
            if len(unique_numeric) <= 3 and max(unique_numeric) <= 2:
                # Classic binary: 0=No, 1=Yes
                return {
                    '1': 'Yes',
                    '0': 'No'
                }
            elif len(unique_numeric) <= 3 and min(unique_numeric) >= 1 and max(unique_numeric) <= 2:
                # Binary on 1-2 scale: 1=No, 2=Yes (common in surveys)
                return {
                    '2': 'Yes',
                    '1': 'No'
                }
            elif len(unique_numeric) <= 3 and 1 in unique_numeric and max(unique_numeric) <= 5:
                # Yes/No question where "No" maps to 1 (Strongly Disagree) and "Yes" maps to higher values
                # This handles cases where Yes/No gets converted to agreement scale
                return {
                    '5': 'Yes',
                    '4': 'Yes',
                    '3': 'Somewhat',
                    '2': 'No',
                    '1': 'No'
                }
        
        # Check if this is a frequency question
        elif any(keyword in question_lower for keyword in ['how often', 'how many times', 'frequency']):
            # Check if this is an access frequency question (course access)
            if any(access_keyword in question_lower for access_keyword in ['access', 'visit', 'log in', 'login']):
                return {
                    '5': 'Five or more',
                    '4': 'Four times a week',
                    '3': 'Three times a week', 
                    '2': 'Twice a week',
                    '1': 'Once a week'
                }
            # Generic frequency scales for other questions
            elif max(unique_numeric) <= 5:
                return {
                    '5': 'Very Often',
                    '4': 'Often',
                    '3': 'Sometimes', 
                    '2': 'Rarely',
                    '1': 'Never'
                }
        
        # Check if this is a workload comparison question
        elif any(keyword in question_lower for keyword in ['compared to', 'relative to', 'workload']):
            return {
                '5': 'Much Higher',
                '4': 'Higher',
                '3': 'About the Same',
                '2': 'Lower',
                '1': 'Much Lower'
            }
        
        # Check if this is a quality rating question
        elif any(keyword in question_lower for keyword in ['quality', 'rate the', 'how would you rate']):
            return {
                '5': 'Excellent',
                '4': 'Good',
                '3': 'Average',
                '2': 'Poor',
                '1': 'Very Poor'
            }
        
        # Default to Likert scale for agreement-based questions
        else:
            return {
                '5': 'Strongly Agree',
                '4': 'Agree',
                '3': 'Neutral',
                '2': 'Disagree',
                '1': 'Strongly Disagree'
            }
    
    # Function to standardize question text into clean column headers
    def create_question_key(question_text: str) -> str:
        """
        Convert question text to standardized column header.
        
        Examples:
        "The instructor was an effective teacher." -> "q_instructor_effective"
        "How would you rate the course as a whole?" -> "q_rate_course_whole"
        """
        if pd.isna(question_text):
            return "q_unknown"
        
        # Convert to lowercase and clean
        clean_text = str(question_text).lower()
        
        # Remove common question prefixes
        prefixes_to_remove = [
            'how would you rate the ',
            'how would you rate ',
            'how did you ',
            'how many ',
            'would you ',
            'do you ',
            'did you ',
            'are you ',
            'is this ',
            'what grade ',
            'relative to other courses you have taken, ',
            'if your class had a lab component, ',
            'on average, ',
        ]
        
        for prefix in prefixes_to_remove:
            if clean_text.startswith(prefix):
                clean_text = clean_text[len(prefix):]
                break
        
        # Remove punctuation and special characters
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        
        # Replace multiple spaces with single space
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Split into words and take meaningful ones
        words = clean_text.split()
        
        # Remove common words that don't add meaning
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'was', 'were', 'is', 'are', 'be', 'been', 'have', 'has', 'had', 'this', 'that',
            'these', 'those', 'you', 'your', 'my', 'our', 'their', 'his', 'her', 'its'
        }
        
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 1]
        
        # Take first 4-5 most meaningful words to keep column names reasonable
        key_words = meaningful_words[:5]
        
        # Join with underscores and add prefix
        question_key = 'q_' + '_'.join(key_words)
        
        # Ensure reasonable length (max 50 chars)
        if len(question_key) > 50:
            question_key = question_key[:50].rstrip('_')
            
        return question_key
    
    # Create standardized question keys and build metadata
    print("   Creating standardized question keys and metadata...")
    df['question_key'] = df['QUESTION'].apply(create_question_key)
    
    # Create question metadata dictionary
    question_metadata = {}
    unique_questions = df[['QUESTION', 'question_key']].drop_duplicates()
    
    for _, row in unique_questions.iterrows():
        question_text = row['QUESTION']
        question_key = row['question_key']
        
        # Get all response values for this question to determine labels
        question_responses = df[df['question_key'] == question_key]['responsevalue'].unique()
        response_labels = determine_response_labels(question_text, set(question_responses))
        
        question_metadata[question_key] = {
            'question_text': question_text,
            'response_labels': response_labels
        }
    
    # Show some examples of question key mapping
    question_examples = unique_questions.head(5)
    print("   Example question key mappings:")
    for _, row in question_examples.iterrows():
        print(f"      '{row['QUESTION'][:60]}...' -> '{row['question_key']}'")
    
    # Convert SectionNumber_ASU to string for consistency
    df['SectionNumber_ASU'] = df['SectionNumber_ASU'].astype(str)
    
    # Create unique student-section identifier for pivoting
    df['student_section_id'] = df['DummyID'].astype(str) + '_' + df['SectionNumber_ASU'].astype(str)
    
    # Pivot the data from long to wide format
    print("   Pivoting data from long to wide format...")
    try:
        # Use pivot_table to handle potential duplicates
        pivot_df = df.pivot_table(
            index=['student_section_id', 'SectionNumber_ASU', 'DummyID'],
            columns='question_key',
            values='numeric_response',
            aggfunc='first'  # Take first value if duplicates exist
        )
        
        # Reset index to make it a regular DataFrame
        wide_df = pivot_df.reset_index()
        
        # Fill NaN values with appropriate defaults
        # For questions, use -1 to indicate "not asked" vs 0 which could be a valid response
        question_columns = [col for col in wide_df.columns if col.startswith('q_')]
        wide_df[question_columns] = wide_df[question_columns].fillna(-1)
        
        print(f"   ✅ Successfully pivoted data:")
        print(f"      - {len(wide_df)} unique student-section combinations")
        print(f"      - {len(question_columns)} question columns")
        print(f"      - Sections: {sorted(wide_df['SectionNumber_ASU'].unique())}")
        print(f"      - Question metadata created for {len(question_metadata)} questions")
        
        # Convert SectionNumber_ASU back to numeric for consistency with other datasets
        wide_df['SectionNumber_ASU'] = pd.to_numeric(wide_df['SectionNumber_ASU'], errors='coerce')
        
        # Remove rows where SectionNumber_ASU conversion failed
        wide_df = wide_df.dropna(subset=['SectionNumber_ASU'])
        
        return wide_df, question_metadata
        
    except Exception as e:
        raise Exception(f"Failed to pivot Likert data: {e}")


def analyze_response_rates(likert_df: pd.DataFrame) -> dict:
    """
    Analyze survey response rates from Likert data.
    
    This function calculates response rates per section using unique DummyID counts
    and compares with built-in enrollment data when available.
    
    Args:
        likert_df (pd.DataFrame): Cleaned Likert data in wide format
        
    Returns:
        dict: Response rate analysis with section-level statistics
    """
    print("📊 Analyzing survey response rates...")
    
    if likert_df.empty:
        print("   ⚠️  No Likert data available for response rate analysis")
        return {}
    
    # Calculate unique respondents per section
    section_stats = []
    
    for section_num in sorted(likert_df['SectionNumber_ASU'].unique()):
        section_data = likert_df[likert_df['SectionNumber_ASU'] == section_num]
        
        # Count unique respondents using DummyID
        unique_respondents = section_data['DummyID'].nunique()
        
        # Get built-in enrollment data if available (from original raw data)
        # Note: This would require passing additional enrollment data
        # For now, we'll focus on respondent counts and response patterns
        
        section_stats.append({
            'SectionNumber_ASU': int(section_num),
            'unique_respondents': unique_respondents,
            'total_responses': len(section_data),
            'avg_questions_per_student': len(section_data) / unique_respondents if unique_respondents > 0 else 0
        })
    
    # Calculate overall statistics
    total_respondents = likert_df['DummyID'].nunique()
    total_sections = len(section_stats)
    avg_respondents_per_section = total_respondents / total_sections if total_sections > 0 else 0
    
    response_analysis = {
        'overall_stats': {
            'total_unique_respondents': total_respondents,
            'total_sections_with_surveys': total_sections,
            'avg_respondents_per_section': round(avg_respondents_per_section, 1)
        },
        'section_details': section_stats,
        'data_quality_indicators': {
            'sections_with_low_response': len([s for s in section_stats if s['unique_respondents'] < 5]),
            'sections_with_high_response': len([s for s in section_stats if s['unique_respondents'] >= 20]),
            'response_distribution': {
                'min_respondents': min([s['unique_respondents'] for s in section_stats]) if section_stats else 0,
                'max_respondents': max([s['unique_respondents'] for s in section_stats]) if section_stats else 0,
                'median_respondents': sorted([s['unique_respondents'] for s in section_stats])[len(section_stats)//2] if section_stats else 0
            }
        }
    }
    
    # Print summary
    print(f"   📈 Response Rate Analysis Summary:")
    print(f"      - Total unique survey respondents: {total_respondents}")
    print(f"      - Sections with survey data: {total_sections}")
    print(f"      - Average respondents per section: {avg_respondents_per_section:.1f}")
    print(f"      - Response range: {response_analysis['data_quality_indicators']['response_distribution']['min_respondents']}-{response_analysis['data_quality_indicators']['response_distribution']['max_respondents']} respondents per section")
    
    # Highlight potential data quality concerns
    low_response_sections = response_analysis['data_quality_indicators']['sections_with_low_response']
    if low_response_sections > 0:
        print(f"      ⚠️  {low_response_sections} sections have <5 respondents (low statistical confidence)")
    
    return response_analysis


def enhance_likert_analysis_with_response_rates(likert_analysis: dict, response_analysis: dict) -> dict:
    """
    Enhance existing Likert analysis results with response rate information.
    
    Args:
        likert_analysis (dict): Existing quantitative analysis results
        response_analysis (dict): Response rate analysis results
        
    Returns:
        dict: Enhanced analysis with response rate context
    """
    if not likert_analysis or not response_analysis:
        return likert_analysis
    
    # Add response rate context to each section's analysis
    enhanced_analysis = likert_analysis.copy()
    
    # Create lookup for response rates by section
    response_lookup = {
        section['SectionNumber_ASU']: section 
        for section in response_analysis.get('section_details', [])
    }
    
    # Enhance section-level analysis
    sections_dict = enhanced_analysis.get('sections', {})
    for section_num, section_data in sections_dict.items():
        response_info = response_lookup.get(section_num, {})
        
        if response_info:
            section_data['response_rate_info'] = {
                'unique_respondents': response_info.get('unique_respondents', 0),
                'statistical_confidence': 'high' if response_info.get('unique_respondents', 0) >= 20 else 
                                        'medium' if response_info.get('unique_respondents', 0) >= 10 else 'low',
                'avg_questions_per_student': response_info.get('avg_questions_per_student', 0)
            }
    
    # Add overall response rate summary
    enhanced_analysis['response_rate_summary'] = response_analysis.get('overall_stats', {})
    
    return enhanced_analysis


def main():
    parser = argparse.ArgumentParser(description='Clean and standardize data files for merging')
    parser.add_argument('--comments', help='Path to comments CSV file')
    parser.add_argument('--schedule', help='Path to schedule CSV file')  
    parser.add_argument('--grades', help='Path to grades CSV file')
    parser.add_argument('--output-dir', default='cleaned', help='Output directory for cleaned files')
    parser.add_argument('--validate', action='store_true', help='Validate merge compatibility after cleaning')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🧹 Data Cleaning Script for Qualitative Coding Agent")
    print("=" * 60)
    
    # Clean files
    cleaned_files = {}
    
    if args.comments:
        output_path = output_dir / "comments_cleaned.csv"
        clean_comments_file(args.comments, str(output_path))
        cleaned_files['comments'] = str(output_path)
    
    if args.schedule:
        output_path = output_dir / "schedule_cleaned.csv"
        clean_schedule_file(args.schedule, str(output_path))
        cleaned_files['schedule'] = str(output_path)
    
    if args.grades:
        output_path = output_dir / "grades_cleaned.csv"
        clean_grades_file(args.grades, str(output_path))
        cleaned_files['grades'] = str(output_path)
    
    # Validate merge compatibility if requested
    if args.validate and len(cleaned_files) >= 2:
        validate_merge_compatibility(
            cleaned_files.get('comments', ''),
            cleaned_files.get('schedule', ''),
            cleaned_files.get('grades', '')
        )
    
    print(f"\n✅ Data cleaning completed!")
    print(f"📁 Cleaned files saved to: {output_dir}")
    for file_type, path in cleaned_files.items():
        print(f"   {file_type}: {path}")


if __name__ == "__main__":
    main() 