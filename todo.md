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

---

## Phase 4: Quantitative Survey Data Integration (NEW MAJOR FEATURE)

This section implements the comprehensive quantitative data stream integration with dynamic peer group benchmarking as outlined in the updated PRD.

### Part 1: Backend Development (Python)

#### Section 1: Data Cleaning & Transformation

    [ ] 1. Create the Likert Cleaning Function (data_cleaner.py):

        [ ] Define a new function: `clean_likert_file(input_path)`.

        [ ] Inside, load the CSV into a pandas DataFrame.

        [ ] Create a mapping dictionary to convert Likert text responses to standardized numerical scale:
            - "Strongly Agree" → 5
            - "Agree" → 4  
            - "Neutral" / "Neither Agree nor Disagree" → 3
            - "Disagree" → 2
            - "Strongly Disagree" → 1
            - "Yes" → 1, "No" → 0 (for non-Likert questions)

        [ ] Create a standardization function to generate clean "question keys" from raw QUESTION text:
            - Convert to lowercase
            - Remove punctuation and special characters
            - Replace spaces with underscores
            - Add "q_" prefix (e.g., "The instructor was effective" → "q_instructor_effective")

        [ ] Use pandas `pivot_table()` function to transform from long format to wide format:
            - Index: ['eval_username', 'SectionNumber_ASU'] (to maintain student-section relationship)
            - Columns: clean question keys  
            - Values: numeric responses
            - Aggfunc: 'first' (in case of duplicates)

        [ ] Handle missing data gracefully (fill NaN values appropriately).

        [ ] Return the cleaned, wide-format DataFrame.

#### Section 2: Peer Group Analysis & Benchmarking

    [ ] 2. Create the Analysis & Benchmarking Function (pipeline.py):

        [ ] Define a new function: `analyze_quantitative_questions(cleaned_likert_df)`.

        [ ] **Peer Group Averages Calculation**:
            - Loop through all question columns in the cleaned DataFrame
            - For each question, identify all sections that asked it (non-null values)
            - Calculate the overall mean score across all students who answered that question
            - Store results in a dictionary: `{'q_instructor_effective': 4.2, 'q_workload_manageable': 3.1, ...}`

        [ ] **Section-Level Analysis**:
            - Group the cleaned_likert_df by 'SectionNumber_ASU'
            - For each section and each question column:
                * Calculate the section-specific average score
                * Calculate response distribution (count of each response value)
                * Convert counts to percentages for easier interpretation
            - Handle sections with insufficient data (< 3 responses) appropriately

        [ ] **Benchmarking Logic**:
            - For each section-question combination, compare section score to relevant peer group average
            - Identify significant deviations (e.g., >1.0 point difference)
            - Flag questions that were unique to a section (no peer comparison available)

        [ ] Return a structured dictionary mapping each section number to its detailed quantitative analysis:
            ```python
            {
                "47666": {
                    "questions_analyzed": ["q_instructor_effective", "q_workload_manageable"],
                    "question_details": {
                        "q_instructor_effective": {
                            "this_section_score": 4.2,
                            "peer_group_average": 4.1,
                            "response_distribution": {"5": 40, "4": 30, "3": 20, "2": 10, "1": 0},
                            "response_count": 20,
                            "significant_deviation": False
                        }
                    }
                }
            }
            ```

#### Section 3: Pipeline Integration

    [ ] 3. Update the Main Pipeline Orchestrator (pipeline.py):

        [ ] Add a new optional parameter `likert_file_path` to the `run_pipeline` function signature.

        [ ] **Conditional Processing Logic**:
            - If `likert_file_path` is provided and file exists:
                * Call `clean_likert_file(likert_file_path)` 
                * Call `analyze_quantitative_questions(cleaned_likert_df)`
                * Store quantitative analysis results for later integration
            - If not provided, set quantitative analysis results to None

        [ ] **Error Handling**:
            - Handle file format errors gracefully
            - Log issues with Likert data processing
            - Ensure pipeline continues even if quantitative analysis fails

    [ ] 4. Update the Hybrid JSON Payload Creation (pipeline.py):

        [ ] Modify `create_hybrid_summary_json` function signature to accept quantitative analysis results.

        [ ] **Payload Enrichment Logic**:
            - For each section in the `quantitative_summary` array:
                * Look up the section's quantitative analysis data
                * If found, add new `likert_summary` key with structured data:
                ```json
                "likert_summary": [
                    {
                        "question_text": "The workload for this course was manageable.",
                        "question_key": "q_workload_manageable", 
                        "this_section_score": 2.2,
                        "peer_group_average": 4.1,
                        "response_distribution": {
                            "Strongly Agree": 2, "Agree": 8, "Neutral": 10, 
                            "Disagree": 15, "Strongly Disagree": 10
                        },
                        "response_count": 45,
                        "significant_deviation": true,
                        "deviation_magnitude": -1.9
                    }
                ]
                ```
                * If no quantitative data available, set `likert_summary: []`

        [ ] **Token Optimization**:
            - Ensure enriched payload remains within model token limits
            - Prioritize questions with significant deviations
            - Consider truncating response distributions if necessary

#### Section 4: Executive Summary Enhancement

    [ ] 5. Update the Executive Summary Prompt (pipeline.py):

        [ ] **Enhanced Prompt Instructions**:
            - Add specific instruction to analyze `likert_summary` data in correlation with qualitative themes
            - Direct the AI to identify significant deviations (where `significant_deviation: true`)
            - Request explicit correlation between quantitative deviations and qualitative sentiment patterns
            - Ask for statistical validation of qualitative themes where applicable

        [ ] **Example Prompt Addition**:
            ```
            "QUANTITATIVE CORRELATION ANALYSIS:
            For each section, examine the likert_summary data. Pay special attention to:
            1. Questions where this_section_score deviates significantly from peer_group_average (deviation_magnitude > 1.0 or < -1.0)
            2. Correlate these quantitative deviations with the qualitative themes for the same section
            3. Validate or contradict qualitative findings with quantitative evidence
            4. Highlight cases where qualitative sentiment aligns or conflicts with quantitative ratings
            
            Example analysis: 'Section 47666 showed negative qualitative themes around workload (15 comments coded as Excessive Workload), which is statistically validated by their quantitative rating of 2.2/5 on workload manageability, significantly below the peer average of 4.1.'"
            ```

### Part 2: Frontend Development (index.html)

#### Section 1: Upload Interface Enhancement

    [ ] 1. Update the Upload UI:

        [ ] Add a fourth file input element for "Quantitative Survey Data (Optional)":
            - Clear labeling: "Quantitative Survey Data (Likert Scale Responses) - Optional"
            - Accept .csv files only
            - Include help text explaining the expected format
            - Visual indication that this field is optional

        [ ] Update the `handleFileSelect` JavaScript function:
            - Add handling for the new Likert file input
            - Validate file format (CSV only)
            - Update form validation to allow submission with or without Likert data
            - Provide user feedback on file selection status

#### Section 2: Dashboard UI Components

    [ ] 2. Create the Quantitative Results Module HTML:

        [ ] Add new container in main dashboard: `<div id="quantitative-results-container" style="display: none;"></div>`

        [ ] **Module Structure**:
            ```html
            <div id="quantitative-results-container" class="dashboard-module">
                <h3>📊 Quantitative Survey Results</h3>
                <div id="quantitative-charts-wrapper">
                    <!-- Dynamic charts will be inserted here -->
                </div>
                <div id="no-quantitative-data" style="display: none;">
                    <p>No quantitative survey data available for this section.</p>
                </div>
            </div>
            ```

#### Section 3: Dynamic Visualization Logic

    [ ] 3. Update the Main Rendering Logic:

        [ ] Modify the main `render()` function to call `renderQuantitativeResults()` after analysis completion.

        [ ] Update section selection logic to refresh quantitative results when user changes selected section.

    [ ] 4. Create the renderQuantitativeResults Function:

        [ ] **Core Function Logic**:
            ```javascript
            function renderQuantitativeResults(selectedSection) {
                const container = document.getElementById('quantitative-results-container');
                const chartsWrapper = document.getElementById('quantitative-charts-wrapper');
                const noDataDiv = document.getElementById('no-quantitative-data');
                
                // Find likert_summary for selected section
                const sectionData = analysisData.quantitative_summary.find(
                    section => section.SectionNumber_ASU === selectedSection
                );
                
                if (!sectionData || !sectionData.likert_summary || sectionData.likert_summary.length === 0) {
                    // Show no data message
                    container.style.display = 'block';
                    chartsWrapper.style.display = 'none';
                    noDataDiv.style.display = 'block';
                    return;
                }
                
                // Clear previous charts and render new ones
                chartsWrapper.innerHTML = '';
                sectionData.likert_summary.forEach(question => renderQuestionChart(question));
                
                container.style.display = 'block';
                chartsWrapper.style.display = 'block';
                noDataDiv.style.display = 'none';
            }
            ```

        [ ] **Individual Question Chart Rendering**:
            ```javascript
            function renderQuestionChart(questionData) {
                const chartDiv = document.createElement('div');
                chartDiv.className = 'question-chart';
                
                // Chart HTML with bars for response distribution
                chartDiv.innerHTML = `
                    <div class="question-header">
                        <h4>${questionData.question_text}</h4>
                        <div class="score-comparison">
                            <span class="section-score ${questionData.significant_deviation ? 'significant' : ''}">
                                This Section: ${questionData.this_section_score.toFixed(1)}
                            </span>
                            ${questionData.peer_group_average ? 
                                `<span class="peer-average">Peer Average: ${questionData.peer_group_average.toFixed(1)}</span>` 
                                : '<span class="unique-question">Unique to this section</span>'
                            }
                        </div>
                    </div>
                    <div class="response-distribution">
                        ${createDistributionBars(questionData.response_distribution, questionData.response_count)}
                    </div>
                `;
                
                document.getElementById('quantitative-charts-wrapper').appendChild(chartDiv);
            }
            ```

        [ ] **Response Distribution Visualization**:
            - Create horizontal bar chart showing percentage breakdown of responses
            - Color-code bars (green for positive, red for negative, gray for neutral)
            - Include percentage labels and response counts
            - Highlight significant deviations with visual indicators

#### Section 4: CSS Styling

    [ ] 5. Add Quantitative Module Styling:

        [ ] **Module Styling**:
            ```css
            .dashboard-module {
                margin: 20px 0;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background: white;
            }
            
            .question-chart {
                margin-bottom: 30px;
                padding: 15px;
                border-left: 4px solid #007bff;
                background: #f8f9fa;
            }
            
            .score-comparison {
                display: flex;
                gap: 20px;
                margin: 10px 0;
            }
            
            .section-score.significant {
                color: #dc3545;
                font-weight: bold;
            }
            
            .response-distribution {
                margin-top: 15px;
            }
            
            .distribution-bar {
                display: flex;
                align-items: center;
                margin: 5px 0;
            }
            ```

### Part 3: Testing & Quality Assurance

    [ ] 6. Comprehensive Testing Strategy:

        [ ] **Unit Testing**:
            - Test `clean_likert_file()` with various input formats
            - Test `analyze_quantitative_questions()` with edge cases (single section, missing data)
            - Test peer group calculation accuracy
            - Test JSON payload structure and token limits

        [ ] **Integration Testing**:
            - Test full pipeline with and without Likert data
            - Test dashboard rendering with various data scenarios
            - Test error handling when Likert file is malformed

        [ ] **User Acceptance Testing**:
            - Test upload workflow with optional Likert file
            - Verify visualization accuracy and clarity
            - Test correlation insights in executive summary

        [ ] **Performance Testing**:
            - Test with large Likert datasets (1000+ responses)
            - Monitor memory usage during pivot operations
            - Ensure dashboard responsiveness with complex visualizations

### Part 4: Documentation & Deployment

    [ ] 7. Enhanced Documentation:

        [ ] Update README.md with quantitative integration instructions
        [ ] Document expected Likert file format and requirements
        [ ] Add troubleshooting guide for common quantitative data issues
        [ ] Include examples of quantitative insights in executive summaries

    [ ] 8. Deployment Considerations:

        [ ] Ensure backward compatibility (system works without Likert data)
        [ ] Test memory requirements with new data processing
        [ ] Update error logging for quantitative analysis failures
        [ ] Consider data validation and user feedback for file format issues

---

## Success Criteria & Validation

- [ ] **Quantitative Integration**: Pipeline successfully processes Likert data and calculates peer group benchmarks
- [ ] **Correlation Analysis**: Executive summaries demonstrate clear correlations between qualitative themes and quantitative deviations  
- [ ] **Dashboard Enhancement**: Users can visualize quantitative results with peer comparisons for selected sections
- [ ] **Performance Maintenance**: Processing time increases by <20% with quantitative data integration
- [ ] **User Experience**: Upload workflow accommodates optional Likert file without confusion
- [ ] **Statistical Accuracy**: Peer group calculations and deviations are mathematically correct and meaningful 