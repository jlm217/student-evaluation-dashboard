#!/usr/bin/env python3
"""
FastAPI application for the Qualitative Coding Agent v2

This serves the React frontend and provides API endpoints for:
- In-memory pipeline processing
- Dashboard data in JSON format
- On-demand CSV downloads
"""

import os
import tempfile
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from io import StringIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

from pipeline import run_pipeline

# Initialize FastAPI app
app = FastAPI(title="Qualitative Coding Agent v2", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for processed dataframes with TTL
DATA_CACHE = {}  # {job_id: {"data": df, "timestamp": datetime, "metadata": dict}}

class ProcessRequest(BaseModel):
    comments_file_path: str
    schedule_file_path: Optional[str] = None
    grades_file_path: Optional[str] = None
    question_col: str = "question"
    answer_col: str = "response"


def cleanup_cache():
    """Remove cache entries older than 1 hour"""
    while True:
        try:
            current_time = datetime.now()
            expired_keys = [
                job_id for job_id, entry in DATA_CACHE.items()
                if current_time - entry["timestamp"] > timedelta(hours=1)
            ]
            for key in expired_keys:
                del DATA_CACHE[key]
                print(f"Cleaned up expired cache entry: {key}")
        except Exception as e:
            print(f"Cache cleanup error: {e}")
        
        time.sleep(300)  # Check every 5 minutes


def clean_for_json_response(data):
    """Clean data for JSON serialization."""
    if isinstance(data, dict):
        return {k: clean_for_json_response(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json_response(item) for item in data]
    elif isinstance(data, (np.integer, np.floating)):
        return float(data)
    elif pd.isna(data) or data in [np.inf, -np.inf]:
        return ""
    else:
        return data


# Start cleanup thread
threading.Thread(target=cleanup_cache, daemon=True).start()


@app.get("/")
async def serve_frontend():
    """Serve the React frontend"""
    return FileResponse("frontend/index.html")


@app.get("/CommentsRAW_SAMPLE_cleaned_coded_themed.csv")
async def serve_demo_csv():
    """Serve demo CSV file for frontend demo"""
    # Check if we have the processed file, otherwise use the raw sample
    demo_files = [
        "CommentsRAW_SAMPLE_cleaned_coded_themed.csv",
        "CommentsRAW_SAMPLE_coded_themed.csv", 
        "CommentsRAW_SAMPLE_themed.csv",
        "CommentsRAW_SAMPLE.csv"
    ]
    
    for filename in demo_files:
        if os.path.exists(filename):
            return FileResponse(filename, media_type="text/csv")
    
    # If no files exist, create a simple demo response
    demo_data = """question,response,Initial_Code,Sentiment,Theme,crs_number,SectionNumber_ASU
"What did you like the least about the course?","The workload was too much for a 100-level class and the professor was unresponsive.","Excessive workload/poor professor","Negative","Course Workload and Difficulty","AST 111","12345"
"What did you like the most about the course?","I liked that it was organized. I really appreciated the fact that the grades were submitted immediately too.","Organization and grading","Positive","Course Structure and Organization","AST 111","12345"
"What do you think are the greatest strengths of this TA?","The TA was very responsive and helpful with questions.","Responsive and helpful TA","Positive","Teaching Assistant (TA) Performance and Support","AST 112","12346"
"""
    
    from fastapi.responses import Response
    return Response(content=demo_data, media_type="text/csv")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}


@app.post("/api/v2/analyze")
async def process_files_endpoint(
    comments_file: UploadFile = File(...),
    schedule_file: Optional[UploadFile] = File(None),
    grades_file: Optional[UploadFile] = File(None)
):
    """
    This single endpoint handles upload, processing, and returning JSON data.
    """
    # Use a context manager for temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            print(f"Processing files in temporary directory: {tmpdir}")
            
            # Save files temporarily
            comments_path = os.path.join(tmpdir, comments_file.filename)
            with open(comments_path, "wb") as buffer:
                shutil.copyfileobj(comments_file.file, buffer)
            print(f"Saved comments file: {comments_file.filename}")

            schedule_path = None
            if schedule_file and schedule_file.filename:
                schedule_path = os.path.join(tmpdir, schedule_file.filename)
                with open(schedule_path, "wb") as buffer:
                    shutil.copyfileobj(schedule_file.file, buffer)
                print(f"Saved schedule file: {schedule_file.filename}")

            grades_path = None
            if grades_file and grades_file.filename:
                grades_path = os.path.join(tmpdir, grades_file.filename)
                with open(grades_path, "wb") as buffer:
                    shutil.copyfileobj(grades_file.file, buffer)
                print(f"Saved grades file: {grades_file.filename}")
            
            # Run the in-memory pipeline
            print("Starting pipeline processing...")
            results = run_pipeline(
                comments_file_path=comments_path,
                schedule_file_path=schedule_path,
                grades_file_path=grades_path
            )

            final_df = results["final_dataframe"]
            markdown_report = results["markdown_report"]
            validation_results = results.get("validation_results", [])
            
            print(f"Pipeline completed. Final DataFrame shape: {final_df.shape}")
            
            # Generate a unique ID for this job and cache the result
            job_id = str(uuid.uuid4())
            
            # Cache the data with metadata
            DATA_CACHE[job_id] = {
                "data": final_df.copy(),
                "timestamp": datetime.now(),
                "metadata": {
                    "comments_file": comments_file.filename,
                    "schedule_file": schedule_file.filename if schedule_file else None,
                    "grades_file": grades_file.filename if grades_file else None,
                    "total_rows": len(final_df),
                    "total_columns": len(final_df.columns)
                }
            }
            
            print(f"Cached results with job ID: {job_id}")

            # Convert dataframe to JSON for the response
            dashboard_json = final_df.to_dict(orient='records')
            
            # Clean the JSON data
            dashboard_json = clean_for_json_response(dashboard_json)
            
            response_data = {
                "jobId": job_id,
                "dashboardData": dashboard_json,
                "markdownReport": markdown_report,
                "validationResults": validation_results,
                "metadata": {
                    "totalRows": len(final_df),
                    "totalColumns": len(final_df.columns),
                    "columnsIncluded": list(final_df.columns),
                    "hasScheduleData": schedule_file is not None,
                    "hasGradesData": grades_file is not None
                }
            }
            
            print(f"Returning response with {len(dashboard_json)} records")
            return response_data

        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")


@app.get("/api/v2/download/{job_id}")
async def download_themed_csv(job_id: str):
    """
    Generates and serves the final themed CSV on-demand.
    """
    if job_id not in DATA_CACHE:
        raise HTTPException(status_code=404, detail="Job ID not found or data has expired.")

    try:
        cache_entry = DATA_CACHE[job_id]
        final_df = cache_entry["data"]
        metadata = cache_entry["metadata"]
        
        print(f"Generating CSV download for job {job_id}")
        print(f"DataFrame shape: {final_df.shape}")
        
        # Convert DataFrame to CSV string in memory
        stream = StringIO()
        final_df.to_csv(stream, index=False, quoting=1)  # QUOTE_ALL for safety
        csv_content = stream.getvalue()
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"themed_analysis_report_{timestamp}.csv"
        
        response = StreamingResponse(
            iter([csv_content]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        
        print(f"CSV download prepared: {filename} ({len(csv_content)} bytes)")
        
        # Optionally, clear the cache after download (uncomment if desired)
        # del DATA_CACHE[job_id]
        
        return response
        
    except Exception as e:
        print(f"Download error for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating CSV download: {str(e)}")


@app.get("/api/v2/job/{job_id}/status")
async def get_job_status(job_id: str):
    """
    Get the status and metadata for a job.
    """
    if job_id not in DATA_CACHE:
        raise HTTPException(status_code=404, detail="Job ID not found or data has expired.")
    
    cache_entry = DATA_CACHE[job_id]
    metadata = cache_entry["metadata"]
    
    return {
        "jobId": job_id,
        "status": "completed",
        "timestamp": cache_entry["timestamp"].isoformat(),
        "metadata": metadata
    }


@app.get("/api/v2/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics (for debugging/monitoring).
    """
    return {
        "totalJobs": len(DATA_CACHE),
        "jobs": [
            {
                "jobId": job_id,
                "timestamp": entry["timestamp"].isoformat(),
                "rowCount": entry["metadata"]["total_rows"],
                "columnCount": entry["metadata"]["total_columns"]
            }
            for job_id, entry in DATA_CACHE.items()
        ]
    }


# Mount static files for frontend
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


# Legacy endpoints for backward compatibility
@app.post("/api/upload-multiple")
async def upload_multiple_files_legacy(
    comments_file: UploadFile = File(...),
    schedule_file: Optional[UploadFile] = File(None),
    grades_file: Optional[UploadFile] = File(None)
):
    """
    Legacy upload-multiple endpoint for backward compatibility
    """
    try:
        uploaded_files = {}
        
        # Save comments file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            shutil.copyfileobj(comments_file.file, tmp_file)
            uploaded_files['comments'] = {
                "filename": comments_file.filename,
                "temp_path": tmp_file.name
            }
        
        # Save schedule file if provided
        if schedule_file and schedule_file.filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                shutil.copyfileobj(schedule_file.file, tmp_file)
                uploaded_files['schedule'] = {
                    "filename": schedule_file.filename,
                    "temp_path": tmp_file.name
                }
        
        # Save grades file if provided
        if grades_file and grades_file.filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                shutil.copyfileobj(grades_file.file, tmp_file)
                uploaded_files['grades'] = {
                    "filename": grades_file.filename,
                    "temp_path": tmp_file.name
                }
        
        return {
            "uploaded_files": uploaded_files,
            "message": "Files uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")


@app.post("/api/process-multiple")
async def process_multiple_files_legacy(request: dict):
    """
    Legacy process-multiple endpoint for backward compatibility - redirects to new v2 API
    """
    try:
        # Extract file paths from request
        comments_file_path = request.get("comments_file_path")
        schedule_file_path = request.get("schedule_file_path")
        grades_file_path = request.get("grades_file_path")
        question_col = request.get("question_col", "question")
        answer_col = request.get("answer_col", "response")
        
        if not comments_file_path:
            raise HTTPException(status_code=400, detail="comments_file_path is required")
        
        # Run the pipeline
        results = run_pipeline(
            comments_file_path=comments_file_path,
            schedule_file_path=schedule_file_path,
            grades_file_path=grades_file_path,
            question_col=question_col,
            answer_col=answer_col
        )
        
        final_df = results["final_dataframe"]
        markdown_report = results["markdown_report"]
        
        # Convert to the legacy format expected by the frontend
        themed_data = final_df.to_dict(orient='records')
        themed_data = clean_for_json_response(themed_data)
        
        # Extract themes from the markdown report (simplified)
        themes = []
        if "## Themes" in markdown_report:
            theme_section = markdown_report.split("## Themes")[1]
            for line in theme_section.split('\n'):
                if line.startswith('### '):
                    theme_name = line.replace('### ', '').split('.', 1)[-1].strip()
                    if theme_name:
                        themes.append(theme_name)
        
        return {
            "themed_data": themed_data,
            "themes": themes,
            "summary": markdown_report,
            "has_schedule": schedule_file_path is not None,
            "has_grades": grades_file_path is not None,
            "coded_file": None,  # Not used in new architecture
            "analysis_file": None  # Not used in new architecture
        }
        
    except Exception as e:
        print(f"Legacy process endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@app.post("/api/upload")
async def upload_file_legacy(file: UploadFile = File(...)):
    """
    Legacy upload endpoint for backward compatibility
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Load and validate CSV
        df = pd.read_csv(tmp_path)
        
        # Clean the preview data
        preview_df = df.head(3).copy()
        preview_df = preview_df.replace([np.inf, -np.inf], '')
        preview_df = preview_df.fillna('')
        
        # Convert to JSON-safe format
        preview_records = []
        for record in preview_df.to_dict('records'):
            clean_record = {}
            for key, value in record.items():
                if isinstance(value, (int, float)) and (np.isnan(value) if isinstance(value, float) else False):
                    clean_record[key] = ""
                elif isinstance(value, float) and (np.isinf(value)):
                    clean_record[key] = ""
                else:
                    clean_record[key] = str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
            preview_records.append(clean_record)
        
        return {
            "filename": file.filename,
            "temp_path": tmp_path,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": preview_records
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Qualitative Coding Agent v2...")
    print("Dashboard will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 