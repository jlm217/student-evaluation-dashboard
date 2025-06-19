#!/usr/bin/env python3
"""
Data Cleaning Script for Qualitative Coding Agent

This script standardizes column names across different data files to enable proper merging:
- Comments file: SectionNumber_ASU (already standardized)
- Schedule file: Number -> SectionNumber_ASU  
- Grades file: Class Nbr -> SectionNumber_ASU

Usage:
    python data_cleaner.py --schedule schedule.csv --grades grades.csv --output-dir cleaned/
"""

import pandas as pd
import argparse
import os
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