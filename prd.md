Product Requirements Document (PRD): Interactive Qualitative Analysis Dashboard

1. Introduction & Vision

This document outlines the requirements for the Interactive Qualitative Analysis Dashboard, a web-based application designed to provide a seamless, end-to-end workflow for analyzing student course feedback. The application will allow users to upload raw data files, trigger a sophisticated AI-powered analysis pipeline, and explore the synthesized results through an intuitive, interactive interface. The vision is to empower educators, administrators, and researchers to move from raw data to actionable insights in minutes, not days.

The system encompasses three distinct but interconnected phases of qualitative data analysis, all orchestrated through a single web interface:

**Phase 0: Data Formatting & Cleaning Agent** serves as the crucial first step, transforming raw, poorly structured survey export data into clean, standardized format for analysis.

**Phase 1: Initial Coding Agent** automates the open-coding phase by processing question-answer pairs and generating relevant initial codes for each entry, along with sentiment analysis.

**Phase 2: Thematic Analysis Agent** takes the coded output and synthesizes the initial codes into higher-level themes, identifying patterns and relationships among the codes.

**Phase 3: Interactive Dashboard** provides the web-based interface that orchestrates the entire pipeline and presents results through an intuitive, interactive dashboard with filtering, visualization, and export capabilities.

This integrated approach significantly accelerates the research workflow, reduces manual effort, and provides consistent, systematic analysis that researchers can refine and build upon.

2. User Personas

    Primary User: The Course Coordinator / Department Head (e.g., "Dr. Anya Sharma")

        Background: University administrator, department head, or senior faculty member responsible for course quality and instructor support. May not have deep technical expertise but needs to make data-driven decisions about curriculum and instruction.

        Goals: To get a high-level overview of student sentiment and recurring issues across multiple course sections. Needs to easily compare feedback for different instructors, modalities, and terms. Wants to share clear, data-backed findings with stakeholders without requiring technical expertise.

        Pain Points: Raw survey data is disconnected from other course information (like grades and enrollment). Manually reading hundreds of comments and identifying themes is not feasible. Lacks a tool to quickly filter and find specific examples of feedback related to a known issue. Current analysis outputs are technical files that require additional work to make presentable.

    Secondary User: The Qualitative Researcher (e.g., "Dr. Sarah Chen")

        Background: University professor, PhD student, or market researcher. Familiar with qualitative methods like grounded theory, interviews, and thematic analysis.

        Goals: To efficiently analyze large volumes of text data and identify patterns while maintaining methodological rigor. Wants to spend more time on deep, interpretive work and less on the tedious initial coding process.

        Pain Points: Initial coding is time-consuming, repetitive, and can be prone to inconsistency. Manually sorting and grouping codes is difficult and mapping final themes back to individual responses requires significant extra work.

3. Core User Flow

    Upload: User arrives at the dashboard and is prompted to upload three separate files: the raw student comments, a course schedule file, and a grades/enrollment file.

    Process: User clicks an "Analyze & Build Dashboard" button. This triggers a backend process that runs the entire analysis pipeline in memory, using pandas DataFrames to pass data between stages.

    Display: The backend sends a single, clean JSON object to the front-end containing all the merged and analyzed data. The front-end renders the interactive dashboard from this JSON.

    Explore & Analyze: The user can select a course section, view metrics, and filter comments by theme.

    Export: The user clicks a "Download Themed CSV" button. The front-end requests the CSV from the backend, which generates it on-demand from the final DataFrame and serves it as a file download.

4. Features & Functionality

    ## Phase 0: Data Formatting & Cleaning Agent

    ### Problem Statement

    The raw data exported from the survey system is not machine-readable in its current state. Key problems include:

    - No Header Row: The file lacks a clear header, making it impossible to reference columns by name.
    - Inconsistent Structure: The data appears to be fixed-width or space-delimited, not comma-separated, making standard CSV parsing fail.
    - Multi-line Cells: Some responses contain line breaks, which can be misinterpreted as new rows.
    - Redundant/Useless Data: The file contains many columns that are not needed for the qualitative analysis, adding noise.
    - Duplicate Entries: The same comments may appear multiple times in the raw export.

    ### V1.0 (Core Functionality)

        File Input: Accepts a raw .txt or .csv file as input.

        Header Injection: The script will programmatically define and apply a correct header row to the data.

        Data Parsing: It will be robust enough to parse the space-delimited or fixed-width format of the raw data, correctly identifying the boundaries between columns.

        Column Selection: The script will discard irrelevant columns and only keep the essential ones for analysis: crs_number, SectionNumber_ASU, question, and response.

        Deduplication: The script will identify and remove duplicate rows based on the content of the question and response columns. This ensures that identical feedback is not processed multiple times in the downstream analysis.

        Whitespace Trimming: The agent will trim leading/trailing whitespace from all data cells to ensure consistency.

        Standardized CSV Output: The final output will be a clean, properly formatted CSV file (e.g., CommentsAST2024_cleaned.csv) that can be directly fed into the Phase 1 agent.

    ## Phase 1: Initial Coding Agent
    
    ### V1.0 (Core Functionality)

        CSV Input: The agent must accept a CSV file as input. The user will specify which columns contain the 'question' and the 'answer' to be analyzed.

        Gemini API Integration: The agent will connect to the Gemini API using a user-provided API key.

        Batch Processing: To optimize for cost and speed, the agent will process the data in batches. It will group a configurable number of rows (e.g., 20) into a single API request.

        Prompt Engineering: The agent will construct a specialized prompt for each batch. The prompt will provide a list of question-answer pairs and instruct the model to return a structured response (e.g., a JSON array of objects). This allows for efficient and reliable parsing of the batched response.

            Example Prompt Frame: "You are a qualitative research assistant. Your task is to perform open coding. For the following list of question-answer pairs, provide a short, descriptive code (2-5 words) for each. Return your response as a JSON array where each object has an 'id' matching the input and a 'code' key. \n\nInput:\n[{\"id\": 0, \"question\": \"Q1...\", \"answer\": \"A1...\"}, {\"id\": 1, \"question\": \"Q2...\", \"answer\": \"A2...\"}]"

        Sentiment Analysis: Along with coding, the agent will determine the sentiment of each response (Positive, Negative, Neutral).

        Output Generation: The agent will create a new CSV file named [original_filename]_coded.csv. This file will contain all the original data plus new columns: Initial_Code and Sentiment.

        Error Handling: The agent will gracefully handle potential errors, such as API connection issues, malformed JSON responses from the API, or missing data in rows, and log these errors for user review.

    ## Phase 2: Thematic Analysis Agent

    ### V1.0 (Core Functionality)

        CSV Input: The agent must accept the coded CSV file from Phase 1 (e.g., [filename]_coded.csv) as input. It will specifically use the Initial_Code and Original_Response columns for its analysis.

        Gemini API Integration: The agent will connect to the Gemini API using a user-provided API key (same configuration as Phase 1).

        Holistic Analysis: The agent will read the entire list of unique initial codes and send them to the Gemini API in a single, comprehensive request for thematic synthesis.

        Prompt Engineering for Synthesis: The agent will use a specialized prompt designed for thematic synthesis. The prompt will instruct the model to act as a senior researcher, review the list of codes, group related codes, and generate a set of overarching themes.

        Structured Thematic Output: The API call will be configured to return a structured JSON object. This object will contain a list of themes, where each theme includes a name, a description, and the list of initial codes that support it.

        Report Generation: The agent will generate a comprehensive suite of outputs for full analysis:

            **Appended CSV File**: A new file, [original_filename]_themed.csv, containing all the original data plus a new Theme column. This column will map each row's Initial_Code to the appropriate theme name generated by the AI.

            **Markdown Theme Report**: A file (themes_report.md) that clearly lists each theme, its description, and the supporting codes.

            **CSV Theme Summary**: A simple CSV file (themes_summary.csv) listing just the themes and their descriptions.

        Error Handling: The agent will handle potential errors in thematic analysis, including API issues, incomplete code mappings, and edge cases where codes don't fit neatly into themes.

    ## Phase 3: Interactive Qualitative Analysis Dashboard

    ### Core User Flow

        Upload: User arrives at the dashboard and is prompted to upload three separate files: the raw student comments, a course schedule file, and a grades/enrollment file.

        Process: User clicks an "Analyze & Build Dashboard" button. This triggers a backend process that cleans the data, runs the AI-powered initial coding and thematic analysis, and merges the results with the schedule and grade information using in-memory DataFrames.

        Display: The backend sends a single, clean JSON object to the front-end containing all the merged and analyzed data. The front-end renders the interactive dashboard from this JSON.

        Explore & Analyze: The user can select a course section, view metrics, and filter comments by theme.

        Export: The user clicks a "Download Themed CSV" button. The front-end requests the CSV from the backend, which generates it on-demand from the final DataFrame and serves it as a file download.

    ### V1.0 (Core Functionality)

        File Upload Interface:

            Three distinct, clearly labeled file input fields for comments, schedule, and grades.

            A single button to initiate the entire analysis process.

            The interface should provide feedback during processing (e.g., a loading spinner with progress indicators).

        Data Processing Backend:

            Executes the three-phase Python pipeline in memory, passing DataFrames between functions without creating intermediate files.

            Merges the data from the three uploaded files based on the SectionNumber_ASU key.

            Primary API Endpoint: Returns a single JSON object containing all data needed for the dashboard (themed comments, grade info, etc.) and the markdown theme report.

            Download Endpoint: An endpoint that generates a CSV file from the final DataFrame on-demand.

        Interactive Dashboard:

            Built in React, receiving all initial data via a single API call returning JSON.

            Contains all components for comprehensive data exploration:

                Course Section Filter: A dropdown menu populated with all unique course sections found in the data.

                Thematic Takeaways Module: A collapsible section displaying the formatted markdown report with themes, descriptions, and supporting codes.

                Course Details Card: Displays key metadata for the selected section (Instructor Name, Modality, Term, Total Enrollment).

                Grade Distribution Chart: A bar chart visualizing the grade distribution for the selected section.

                Student Feedback Module:
                    - Theme Filters: Dynamically generated buttons for each theme plus an "All Themes" button.
                    - Comment Cards: Scrollable list showing question, response, theme tags, and sentiment.
                    - Real-time filtering that instantly updates displayed comments when theme buttons are clicked.

            Data Export: A "Download Themed CSV" button.

    ### V2.0 (Future Enhancements)

        Customizable Prompts: Allow the user to edit or provide their own prompt template via a configuration file for all phases.

        Interactive Review Mode: Before saving the final CSV, present the generated codes/themes in a simple terminal interface, allowing the user to accept, reject, or edit each analysis.

        Support for Other File Types: Add support for .xlsx, .txt, and .docx files.

        Advanced Error Recovery: Implement logic to retry failed batches or individual items within a batch.

        Theme Refinement: Allow users to iteratively refine themes by providing feedback and regenerating the thematic analysis.

        Cross-Phase Integration: Seamless pipeline where each phase automatically begins after the previous completion, with user confirmation.

5. Technical Requirements

    Language: Python (Backend), JavaScript/React (Frontend)

    Key Libraries: 
    - **Backend**: FastAPI (web framework), pandas (data manipulation), google-generativeai (Gemini API), uvicorn (ASGI server), python-multipart (file uploads), python-dotenv (environment management)
    - **Frontend**: React.js, Axios (API calls), Chart.js (visualizations), Tailwind CSS (styling)

    Architecture:
    - **Data Processing**: In-memory pipeline using pandas DataFrames passed between analysis phases
    - **API Communication**: JSON for all data transfer between backend and frontend
    - **File Handling**: Temporary file processing with immediate cleanup, no persistent intermediate files

    Environment: 
    - **Backend**: FastAPI application deployable with uvicorn, compatible with standard Python hosting platforms
    - **Frontend**: React application buildable for static hosting or served alongside FastAPI backend
    - **Development**: Local development with hot reload for both frontend and backend

    Data Transfer: JSON for all API communication between backend and frontend. Intermediate CSV files are eliminated in favor of in-memory DataFrame processing.

    API Key Management: Secure handling of Gemini API keys through environment variables, not hardcoded in application.

    Performance Considerations:
    - Asynchronous processing for file uploads and AI API calls
    - Progress indicators for long-running analysis pipeline
    - Efficient data serialization for large datasets
    - On-demand CSV generation to minimize memory usage

6. Success Metrics

    **Overall System Metrics**:
    - **User Experience**: Users can successfully upload files, trigger analysis, and explore results within 5 minutes of first use without training.
    - **Data Integration**: Successfully merges comments, schedule, and grade data with 99%+ accuracy based on section matching.
    - **Performance**: Dashboard loads and renders within 3 seconds for datasets up to 1000 comments.
    - **End-to-End Efficiency**: The complete system should reduce total analysis time by at least 85% compared to fully manual analysis.
    - **Accessibility**: Interface is intuitive for non-technical stakeholders to independently use and interpret results in 90%+ of cases.

    **Technical Performance**:
    - **Pipeline Execution**: Complete three-phase analysis completes within 2 minutes for typical datasets (100-500 comments).
    - **Memory Efficiency**: System handles datasets up to 2000 comments without performance degradation.
    - **API Reliability**: 99%+ uptime for dashboard functionality with graceful error handling for AI API issues.

    **Research Quality**:
    - **Code Quality**: Generated codes are relevant and useful as starting points in over 85% of cases.
    - **Theme Coherence**: Generated themes are logical and accurately reflect underlying codes.
    - **Traceability**: Clear mapping from individual comments to themes maintained throughout the process.
    - **User Adoption**: Successfully used by at least 5 educators/researchers in first quarter of release. 