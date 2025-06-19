# To-Do List: Building the AI-Assisted Qualitative Analysis Workflow

This to-do list breaks down the development of the complete three-phase Python workflow into actionable steps.

## Phase 0: Data Formatting & Cleaning Agent (NEW)

This section provides an updated plan for creating the Python script to clean the raw data, including column selection and deduplication.

### Phase 1: Setup and Data Loading

    [ ] 1. Set Up Project Environment:

        [ ] Create a new project directory (e.g., phase0_data_cleaner).

        [ ] Set up and activate a Python virtual environment.

        [ ] Create a requirements.txt file and add pandas to it.

        [ ] Run pip install -r requirements.txt.

    [ ] 2. Script Scaffolding (formatter.py):

        [ ] Create the main Python file and import pandas and csv.

    [ ] 3. Load the Raw Data:

        [ ] Use pandas.read_csv() to load the data.

        [ ] Use a regular expression separator (sep='\s\s+') to handle the space-delimited format.

        [ ] Set header=None since the file has no header row.

        [ ] Example: df = pd.read_csv('your_raw_file.txt', sep='\s\s+', header=None, engine='python')

### Phase 2: Data Transformation

    [ ] 4. Define and Assign Headers:

        [ ] Create a list of strings for the column headers that matches the structure of the raw file.

        [ ] Assign the headers to the DataFrame: df.columns = headers.

    [ ] 5. Select Essential Columns:

        [ ] Create a new DataFrame containing only the four required columns.

        [ ] Rename columns for simplicity and consistency (e.g., crs_number to CourseNumber).

        [ ] Example: selected_df = df[['crs_number', 'SectionNumber_ASU', 'question', 'response']].

    [ ] 6. Clean Data Cells:

        [ ] Apply a function to trim leading/trailing whitespace from all columns.

        [ ] Example: clean_df = selected_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x).

    [ ] 7. Deduplicate Data:

        [ ] Remove duplicate rows based on the question and response columns to ensure each unique comment is only present once.

        [ ] Example: clean_df.drop_duplicates(subset=['question', 'response'], inplace=True, keep='first').

### Phase 3: Output

    [ ] 8. Save to a Clean CSV:

        [ ] Use df.to_csv() to save the final clean_df DataFrame.

        [ ] Set index=False to avoid writing the pandas index as a column.

        [ ] Use quoting=csv.QUOTE_ALL to ensure all text fields are properly quoted.

        [ ] Example: clean_df.to_csv('CommentsAST2024_cleaned.csv', index=False, quoting=1).

    [ ] 9. Finalize and Document:

        [ ] Add comments to the script explaining the cleaning logic.

        [ ] Create a README.md explaining the script's purpose and how to run it.

---

## Phase 1: Initial Coding Agent (COMPLETED)

### Phase 1: Setup & Core Logic

    [x] 1. Set Up Project Environment:

        [x] Create a new project directory and a Python virtual environment (python -m venv venv).

        [x] Create a requirements.txt file.

    [x] 2. Install Dependencies:

        [x] Add pandas, google-generativeai, python-dotenv, and tqdm to requirements.txt.

        [x] Install the packages: pip install -r requirements.txt.

    [x] 3. Secure API Key:

        [x] Create a .env template file for GEMINI_API_KEY.

        [x] Add .env to your .gitignore file.

    [x] 4. Script Scaffolding (main.py):

        [x] Create a main.py file and import necessary libraries.

        [x] Write the main function structure to orchestrate the script's flow.

        [x] Define constants, such as BATCH_SIZE = 20.

### Phase 2: File Handling & API Integration

    [x] 5. Load Configuration & API Key:

        [x] In main.py, load environment variables and configure the Gemini client.

    [x] 6. Read Input CSV:

        [x] Write a function to load the source CSV into a pandas DataFrame and validate required columns.

    [x] 7. Implement Batch API Call Function:

        [x] Create a function get_codes_for_batch(batch_df) that accepts a DataFrame chunk.

        [x] Inside the function:

            [x] Convert the batch_df into a list of dictionaries.

            [x] Construct the JSON input for the prompt.

            [x] Build the full prompt using the frame defined in the PRD.

            [x] Call the model.generate_content() API.

            [x] Implement a try...except block to parse the JSON response from the model. Handle JSON decoding errors.

            [x] Return the parsed list of code objects.

### Phase 3: Processing & Output

    [x] 8. Iterate and Process in Batches:

        [x] Create an empty list to hold all the generated codes.

        [x] Loop through the main DataFrame in chunks of BATCH_SIZE.

        [x] Use tqdm to create a progress bar for the batch loop.

        [x] For each batch:

            [x] Call the get_codes_for_batch() function.

            [x] Extend the master list of codes with the results from the batch.

            [x] Include a small time.sleep() to avoid hitting rate limits, if necessary.

    [x] 9. Create and Save Output CSV:

        [x] After the loop, create a new DataFrame from the list of code objects.

        [x] Merge this new DataFrame with the original DataFrame based on the index.

        [x] Save the final, merged DataFrame to a new CSV file (..._coded.csv), ensuring the index is not written to the file.

    [x] 10. Final Polish & Documentation:

        [x] Add clear comments to the code.

        [x] Create a README.md explaining setup, configuration (like BATCH_SIZE), and execution.

        [x] Perform a test run on a small slice of the data before processing the entire file.

---

## Phase 2: Thematic Analysis Agent (COMPLETED)

This section breaks down the development of the Phase 2 Python script for thematic synthesis.

### Phase 1: Setup & Data Ingestion

    [x] 1. Set Up Project Environment:

        [x] Create a new project directory (e.g., phase2_thematic_analysis) or integrate into existing project.

        [x] Set up a Python virtual environment and activate it (if not using existing).

        [x] Update requirements.txt file with any additional dependencies.

    [x] 2. Install Dependencies:

        [x] Add pandas, google-generativeai, and python-dotenv to requirements.txt (if not already present).

        [x] Install the packages: pip install -r requirements.txt.

    [x] 3. API Key & Configuration:

        [x] Reuse the .env file from Phase 1 or create a new one to store the Gemini API key.

    [x] 4. Script Scaffolding (thematic_analyzer.py):

        [x] Create the main Python file.

        [x] Import libraries and load the API key from the environment.

        [x] Configure the Gemini client.

    [x] 5. Read Coded CSV:

        [x] Write a function to load SAMPLEComments_coded.csv into a pandas DataFrame.

        [x] Extract the Initial_Code and the original student Answer columns into a list of strings or a list of dictionaries. This list will be the input for the prompt.

### Phase 2: Thematic Synthesis & API Call

    [x] 6. Craft the Thematic Synthesis Prompt:

        [x] Create a separate function or a multi-line string variable to hold the detailed prompt.

        [x] The function should take the list of initial codes as an argument and insert it into the prompt template.

    [x] 7. Implement the API Call:

        [x] Create a function generate_themes(codes_list).

        [x] Inside this function, construct the full prompt.

        [x] Call the model.generate_content() method.

        [x] Crucially, configure the generation_config to ask for a JSON response, which makes parsing reliable.

        [x] Wrap the API response parsing in a try...except block to handle potential JSON errors.

        [x] Return the parsed Python dictionary representing the themes.

### Phase 3: Report Generation

    [x] 8. Write Markdown/Text Report:

        [x] Create a function write_markdown_report(themes_data).

        [x] Loop through the themes data returned from the API.

        [x] For each theme, write the theme name (as a heading), the description, and a bulleted list of the supporting initial codes to a .md file.

    [x] 9. Write CSV Summary:

        [x] Create a function write_csv_summary(themes_data).

        [x] Convert the list of themes (just the name and description) into a pandas DataFrame.

        [x] Save the DataFrame to themes_summary.csv.

    [x] 10. Create Themed CSV Output:

        [x] Create a function create_themed_csv(original_df, themes_data).

        [x] Map each Initial_Code to its corresponding theme name.

        [x] Add a new 'Theme' column to the original DataFrame.

        [x] Save as [original_filename]_themed.csv.

    [x] 11. Orchestrate Main Function:

        [x] In the if __name__ == "__main__": block:

            [x] Call the function to read the coded CSV.

            [x] Extract unique initial codes for thematic analysis.

            [x] Call the function to generate themes via the API.

            [x] If the API call is successful, call all report-writing functions.

            [x] Add print statements to inform the user that the process is complete and the files have been created.

### Phase 4: Integration & Testing

    [x] 12. Error Handling & Edge Cases:

        [x] Handle cases where codes don't map to themes cleanly.

        [x] Add logging for troubleshooting API issues.

        [x] Validate input file format and required columns.

    [x] 13. Testing & Validation:

        [x] Test with the sample coded data.

        [x] Validate that all three output files are generated correctly.

        [x] Verify theme-to-code mappings are accurate.

    [ ] 14. Documentation:

        [ ] Update README.md to include Phase 2 usage instructions.

        [ ] Document the complete three-phase workflow.

        [ ] Add examples of expected outputs.

---

## Phase 3: Interactive Qualitative Analysis Dashboard (Robust Architecture)

This updated section reflects the shift to an in-memory, JSON-based data flow as outlined in the updated PRD.

### Part 1: Backend Development (Python/FastAPI)

    [ ] 1. Refactor Core Logic (VERY IMPORTANT):

        [ ] Modify formatter.py: format_data function should return a pandas DataFrame, not a file path.

        [ ] Modify main.py (or its logic in pipeline.py): The initial coding function should accept a DataFrame and return a DataFrame with new columns.

        [ ] Modify thematic_analyzer.py: analyze_themes should accept a DataFrame, perform the analysis, and return the final DataFrame with the 'Theme' column added, along with the markdown report content.

    [ ] 2. Update the Pipeline Orchestrator (pipeline.py):

        [ ] Rewrite run_pipeline to manage the flow of DataFrames between the refactored functions.

        [ ] The function will now take initial file paths and return a final, fully processed DataFrame and the markdown report.

    [ ] 3. Update the FastAPI App (app.py):

        [ ] Modify the /process-multiple endpoint:

            It will call the updated run_pipeline orchestrator.

            It will receive the final DataFrame and markdown report back.

            It will convert the DataFrame to JSON (df.to_dict('records')).

            It will return a single JSON response to the front-end, e.g., {"dashboardData": [...], "markdownReport": "..."}.

            It should store the final DataFrame in a temporary location (e.g., a simple cache or temp file) associated with a unique job ID, to be used for the download request.

    [ ] 4. Create a Robust Download Endpoint (app.py):

        [ ] Create a /download-csv/{job_id} endpoint.

        [ ] This endpoint will retrieve the final DataFrame using the job ID.

        [ ] It will convert the DataFrame to a CSV string in memory (using df.to_csv()).

        [ ] It will return this string as a StreamingResponse with the correct Content-Disposition header to trigger a browser download.

### Part 2: Frontend Development (React)

    [ ] 1. Update API Call Logic:

        [ ] The handleAnalysis function will now expect a single, large JSON object from the /process-multiple endpoint.

        [ ] Update the state management (useState) to store this JSON data directly.

    [ ] 2. Implement Download Button:

        [ ] When the user clicks the "Download" button, make a GET request to the /download-csv/{job_id} endpoint.

        [ ] The browser will handle the response from this endpoint as a file download automatically.

    [ ] 3. (No Change Needed) Component Rendering:

        [ ] The rest of the component logic remains the same, as it will just be mapping over the JSON data it now receives. 