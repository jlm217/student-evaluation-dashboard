# Product Requirements Document (PRD): Qualitative Coding Agent with Quantitative Survey Data Integration

## 1. Vision

To create a comprehensive analytical tool that combines qualitative student feedback analysis with quantitative survey data integration. The system will intelligently correlate subjective student comments with objective, scaled responses (Likert scales, multiple choice) while dynamically handling inconsistent survey questions across different sections through intelligent peer group benchmarking. This will provide a holistic understanding of the student experience with both narrative insights and statistical validation.

## 2. Background / Problem Statement

### Current Limitations
- **Lack of Quantitative Dimension**: The current analysis relies solely on open-ended comments and cannot answer questions like "Do negative comments about workload correlate with low student ratings on a 'workload was manageable' Likert scale question?"
- **Data Heterogeneity**: Quantitative survey data is not uniform across sections. Different course sections may have been asked different sets of survey questions, making simple global aggregation misleading and methodologically flawed.
- **Missing Statistical Validation**: Qualitative themes lack quantitative backing to validate or contradict narrative insights.

### Opportunity
By integrating quantitative survey data with dynamic peer group benchmarking, we can provide fair "apples-to-apples" comparisons and enable powerful correlational analysis between qualitative themes and quantitative metrics.

## 3. Goals & Success Metrics

| Goal | Metric | Baseline | Target |
|------|--------|----------|--------|
| Integrate New Data Source | Pipeline successfully cleans, analyzes, and merges Likert data file | N/A | 100% success rate |
| Enable Deeper Insights | Executive Summary correlates qualitative themes with quantitative survey data | No correlation | ≥3 significant correlations identified per report |
| Provide Clear Visualizations | Dashboard dynamically displays quantitative survey results with peer benchmarks | No display | All relevant questions displayed with peer comparisons |
| Maintain Performance | Processing time remains acceptable with new data source | Current baseline | <20% increase in processing time |

## 4. Functional Requirements

### F1. Dynamic Data Cleaning & Transformation

**F1.1** The system SHALL include a new `clean_likert_file()` function capable of parsing the "long" format of the raw Likert data CSV.

**F1.2** This function MUST pivot the data into a "wide" format, where each row represents a unique respondent and each column represents a unique survey question.

**F1.3** It MUST convert text-based Likert responses (e.g., "Strongly Agree") into a standardized numerical scale (e.g., 5, 4, 3, 2, 1) and handle non-Likert questions (Yes/No → 1/0).

**F1.4** It MUST standardize question text into short, machine-readable column headers (e.g., "The instructor was an effective teacher." → `q_instructor_effective`).

**F1.5** The function MUST handle missing data gracefully and maintain data integrity during the transformation process.

### F2. Peer Group Benchmarking

**F2.1** The pipeline SHALL include a new `analyze_quantitative_questions()` function that identifies "peer groups"—sets of sections that were asked the exact same question.

**F2.2** For each common question, this function MUST calculate a `peer_group_average` score across all students in that group.

**F2.3** For each individual section, the function MUST calculate the section's specific average score (`this_section_score`) for each question it asked.

**F2.4** The function MUST calculate response distributions for each question within each section.

### F3. Token-Efficient Payload Enrichment

**F3.1** The `create_hybrid_summary_json` function SHALL be updated to enrich the `quantitative_summary` payload.

**F3.2** For each course section, a new key `likert_summary` will be added containing a list of objects with `question_text`, `this_section_score`, `peer_group_average`, and `response_distribution`.

**F3.3** The enriched payload MUST remain within token limits for the AI model while providing comprehensive quantitative context.

### F4. Dynamic Dashboard Visualization

**F4.1** The front-end dashboard MUST dynamically render a new "Quantitative Survey Results" module when a course section is selected.

**F4.2** This module MUST display visualizations (bar charts) for each quantitative question answered in that section, showing both the section's score and the peer group average for comparison.

**F4.3** The visualization MUST clearly indicate when a question was unique to a section (no peer comparison available).

### F5. Correlation Analysis

**F5.1** The system SHALL identify significant deviations (>1.0 point difference) between section scores and peer group averages.

**F5.2** The AI prompt MUST be enhanced to correlate these quantitative deviations with qualitative themes.

## 5. Technical Requirements

### Input Requirements
- **New Data Source**: Optional Likert data CSV file with columns: `SectionNumber_ASU`, `QUESTION`, `responsevalue`
- **File Format**: CSV with "long" format (one row per student per question)
- **Upload Interface**: Extended to handle the additional file upload

### Backend Requirements
- **Data Processing**: Extended `data_cleaner.py` with `clean_likert_file()` function
- **Analysis Pipeline**: Enhanced `pipeline.py` with `analyze_quantitative_questions()` function
- **Statistical Calculations**: Peer group benchmarking and section-level aggregations
- **Memory Management**: Efficient processing of potentially large datasets

### API Requirements
- **Endpoint Enhancement**: `/analyze` endpoint JSON response enriched with `likert_summary` data
- **No New Endpoints**: Leverage existing API structure
- **Response Format**: Maintain backward compatibility while adding new data fields

### Frontend Requirements
- **UI Enhancement**: Updated `index.html` JavaScript to parse and display `likert_summary` data
- **Dynamic Rendering**: New UI components for quantitative survey results
- **Visualization**: Bar charts with peer group comparisons
- **Responsive Design**: Maintain user experience across devices

### Data Structure Requirements

#### Cleaned Likert Data Format
```python
# Wide format after cleaning
{
    'SectionNumber_ASU': str,
    'q_instructor_effective': float,
    'q_workload_manageable': float,
    'q_final_project_valuable': float,
    # ... additional question columns
}
```

#### Enhanced Quantitative Summary Format
```json
{
  "SectionNumber_ASU": "47666",
  "Instructor": "A. Smith",
  "Instruction_Mode": "Online",
  "Term": "Fall 2023",
  "Total_Enrollment": 45,
  "DEW_Rate_Percent": 21.5,
  "Grade_Distribution": {...},
  "likert_summary": [
    {
      "question_text": "The workload for this course was manageable.",
      "this_section_score": 2.2,
      "peer_group_average": 4.1,
      "response_distribution": {
        "Strongly Agree": 2,
        "Agree": 8,
        "Neutral": 10,
        "Disagree": 15,
        "Strongly Disagree": 10
      }
    }
  ]
}
```

## 6. Implementation Strategy

### Phase 1: Data Integration Foundation
1. Implement `clean_likert_file()` function in `data_cleaner.py`
2. Create `analyze_quantitative_questions()` function in `pipeline.py`
3. Test data cleaning and peer group calculation logic

### Phase 2: Executive Summary Enhancement
1. Update `create_hybrid_summary_json()` to include `likert_summary`
2. Enhance AI prompt to leverage quantitative data for correlational analysis
3. Test end-to-end pipeline with sample data

### Phase 3: Dashboard Integration
1. Update frontend JavaScript to parse new data structure
2. Implement quantitative survey results visualization
3. Add peer group comparison displays

### Phase 4: Testing & Optimization
1. Comprehensive testing with various data scenarios
2. Performance optimization for large datasets
3. User acceptance testing and feedback incorporation

## 7. Risk Mitigation

| Risk | Impact | Mitigation Strategy |
|------|--------|-------------------|
| Large dataset performance issues | High | Implement efficient pandas operations and consider chunked processing |
| Inconsistent question mapping | Medium | Robust text standardization and manual review capability |
| Token limit exceeded with enriched data | High | Careful payload design and compression techniques |
| Complex peer group logic errors | Medium | Comprehensive unit testing and validation checks |

## 8. Success Criteria

- [ ] Pipeline successfully processes Likert data and identifies peer groups
- [ ] Executive summaries demonstrate clear correlations between qualitative and quantitative data
- [ ] Dashboard displays quantitative results with peer comparisons
- [ ] Processing time remains within acceptable limits
- [ ] User feedback indicates improved analytical insights

## 9. Future Enhancements

- **Advanced Statistical Analysis**: Correlation coefficients, significance testing
- **Interactive Filtering**: Filter peer groups by instruction mode, term, etc.
- **Export Capabilities**: Download quantitative analysis results
- **Historical Trending**: Track quantitative metrics across multiple terms 