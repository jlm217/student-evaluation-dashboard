# AI-Assisted Qualitative Analysis Workflow

This project provides a complete Python-based workflow for automating qualitative data analysis using AI. The system processes raw survey export data through three integrated phases: data cleaning, initial coding, and thematic analysis.

## Overview

The workflow consists of three main phases:

1. **Phase 0: Data Formatting & Cleaning Agent** - Transforms raw, poorly structured survey exports into clean, standardized CSV format
2. **Phase 1: Initial Coding Agent** - Automatically generates initial codes and sentiment analysis for question-answer pairs using Google's Gemini API
3. **Phase 2: Thematic Analysis Agent** - Synthesizes initial codes into higher-level themes and generates comprehensive reports

## Features

### Phase 0: Data Cleaning
- **Multi-format Support**: Handles CSV, TXT, and space-delimited files
- **Header Management**: Automatically detects or assigns proper column headers
- **Column Selection**: Extracts only essential columns (course number, section, question, response)
- **Data Deduplication**: Removes duplicate question-response pairs
- **Whitespace Cleaning**: Trims and standardizes text formatting
- **Flexible Parsing**: Robust handling of various delimiter formats

### Phase 1: Initial Coding
- **AI-Powered Coding**: Uses Google's Gemini API for intelligent code generation
- **Batch Processing**: Processes data in configurable batches for cost efficiency
- **Sentiment Analysis**: Automatically determines sentiment (Positive, Negative, Neutral)
- **Error Handling**: Robust error handling for API issues and malformed responses
- **Progress Tracking**: Real-time progress bars for batch processing

### Phase 2: Thematic Analysis
- **Theme Generation**: Groups related codes into overarching themes
- **Multiple Outputs**: Generates markdown reports, CSV summaries, and themed datasets
- **Comprehensive Mapping**: Links each original response to its final theme
- **Statistical Summary**: Provides counts and distributions of codes and themes

### Integrated Pipeline
- **Seamless Workflow**: Automated progression from raw data to final analysis
- **Flexible Execution**: Run individual phases or the complete pipeline
- **Command-line Interface**: Easy-to-use CLI with multiple options
- **Error Recovery**: Robust error handling with clear feedback

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a `.env` file in the project root:
   ```bash
   touch .env
   ```
3. Add your API key to `.env`:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## Usage

### Complete Pipeline (Recommended)

Run the full three-phase pipeline on raw data:

```bash
python pipeline.py your_raw_data.csv
```

This will automatically:
1. Clean and format your raw data
2. Generate initial codes and sentiment analysis
3. Create themes and comprehensive reports

### Individual Phases

#### Phase 0 Only (Data Cleaning)
```bash
python pipeline.py raw_data.csv --only-phase0
# or
python formatter.py raw_data.csv
```

#### Phase 1 Only (Initial Coding)
```bash
python pipeline.py clean_data.csv --only-phase1 --skip-phase0
# or
python main.py  # Interactive mode
```

#### Phase 2 Only (Thematic Analysis)
```bash
python pipeline.py coded_data.csv --only-phase2 --skip-phase0 --skip-phase1
# or
python thematic_analyzer.py  # Interactive mode
```

### Skip Specific Phases

```bash
# Skip data cleaning (if data is already clean)
python pipeline.py clean_data.csv --skip-phase0

# Skip thematic analysis (stop after coding)
python pipeline.py raw_data.csv --skip-phase2
```

## Input Formats

### Raw Data (Phase 0 Input)
Phase 0 can handle various formats:
- Standard CSV files with headers
- Space-delimited text files
- Tab-delimited files
- Files without headers (will assign automatically)

### Clean Data (Phase 1 Input)
CSV files with these essential columns:
- `crs_number`: Course identifier
- `SectionNumber_ASU`: Section number
- `question`: The survey question
- `response`: The student's response

### Coded Data (Phase 2 Input)
CSV files from Phase 1 with additional columns:
- `Initial_Code`: Generated codes from Phase 1
- `Sentiment`: Sentiment analysis (Positive/Negative/Neutral)

## Output Files

The pipeline generates multiple output files:

### Phase 0 Outputs
- `[filename]_cleaned.csv`: Clean, standardized data ready for Phase 1

### Phase 1 Outputs
- `[filename]_coded.csv`: Original data with initial codes and sentiment analysis

### Phase 2 Outputs
- `[filename]_themed.csv`: Complete dataset with theme assignments
- `themes_report.md`: Detailed markdown report of all themes
- `themes_summary.csv`: Summary table of themes and descriptions

## Example Workflow

```bash
# Complete pipeline from raw export to final analysis
python pipeline.py CommentsRAW_SAMPLE.csv
```

**Sample Output:**
```
================================================================================
🔬 QUALITATIVE ANALYSIS PIPELINE
================================================================================

📋 PHASE 0: DATA FORMATTING & CLEANING
--------------------------------------------------
Loading raw data from: CommentsRAW_SAMPLE.csv
Successfully loaded CSV with 94 rows and 12 columns
Selected 4 essential columns: ['crs_number', 'SectionNumber_ASU', 'question', 'response']
Removed 10 duplicate entries (94 -> 84 rows)
✅ Phase 0 completed successfully!

🤖 PHASE 1: AI-ASSISTED INITIAL CODING
--------------------------------------------------
Processing 80 rows in 4 batches of 20
✅ Phase 1 completed successfully!
   Successfully coded: 80/80 rows

🎯 PHASE 2: THEMATIC ANALYSIS & REPORTING
--------------------------------------------------
Generated 8 themes from 76 unique codes
✅ Phase 2 completed successfully!

🎉 PIPELINE COMPLETED SUCCESSFULLY!
```

## Configuration

### Batch Size
Modify `BATCH_SIZE` in `pipeline.py` to adjust API batch processing (default: 20)

### Column Names
The system expects these column names after Phase 0:
- `question`: Survey questions
- `response`: Student responses

## Error Handling

The pipeline handles various error conditions:
- **Phase 0**: Invalid file formats, missing data, parsing errors
- **Phase 1**: Missing API key, network issues, malformed responses
- **Phase 2**: Missing codes, API failures, file I/O errors

Each phase provides clear error messages and stops execution if critical errors occur.

## Performance Tips

1. **Use appropriate batch sizes**: Larger batches are more efficient but may hit API limits
2. **Clean data first**: Run Phase 0 on raw data to improve downstream processing
3. **Monitor API usage**: The system includes rate limiting to prevent API abuse
4. **Process in stages**: For large datasets, consider running phases separately

## File Structure

```
qualitative-coding-agent/
├── pipeline.py              # Main integrated pipeline
├── formatter.py             # Phase 0: Data cleaning
├── main.py                  # Phase 1: Initial coding
├── thematic_analyzer.py     # Phase 2: Thematic analysis
├── requirements.txt         # Python dependencies
├── .env                     # API configuration (create this)
├── README.md               # This file
└── [output files]          # Generated analysis results
```

## License

This project is provided as-is for research purposes. 