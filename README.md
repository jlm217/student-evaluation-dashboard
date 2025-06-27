# Qualitative Coding Agent

An AI-powered tool for automated qualitative data analysis using Google's Gemini API. This application provides automated open coding, thematic analysis, and executive summary generation for survey response data.

## Features

- **Automated Open Coding**: Uses AI to generate initial codes for qualitative responses
- **Thematic Analysis**: Groups codes into meaningful themes  
- **Executive Summary Generation**: Creates comprehensive narrative summaries
- **Real-time Progress Tracking**: WebSocket-based progress updates
- **Interactive Dashboard**: Web-based interface for data exploration
- **Conversational Data Exploration**: Chat with your analyzed data

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd qualitative-coding-agent
   ```

2. **Set up environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

4. **Run the application**
   ```bash
   python3 -m uvicorn app:app --host 0.0.0.0 --port 8080 --reload
   ```

5. **Access the application**
   - Open http://localhost:8080 in your browser
   - Upload your CSV files and start analysis

### Deployment on Render

This application is configured for easy deployment on Render.com:

1. **Fork this repository** to your GitHub account

2. **Connect to Render**
   - Go to [render.com](https://render.com) and sign up
   - Connect your GitHub account
   - Create a new Web Service from your forked repository

3. **Set Environment Variables**
   - In Render dashboard, add the following environment variable:
     - `GEMINI_API_KEY`: Your Google Gemini API key

4. **Deploy**
   - Render will automatically deploy using the `render.yaml` configuration
   - Your app will be available at your Render URL

## API Documentation

### Core Endpoints

- `POST /api/v2/analyze-with-progress` - Upload files and start analysis with WebSocket progress
- `GET /api/v2/job/{job_id}/status` - Check analysis job status  
- `GET /api/v2/download/{job_id}` - Download processed CSV results
- `POST /api/v2/chat` - Chat with analyzed data
- `GET /api/health` - Health check endpoint

### WebSocket Endpoint

- `WS /ws/progress/{job_id}` - Real-time progress updates during analysis

## File Structure

```
qualitative-coding-agent/
├── app.py                 # FastAPI web application
├── main.py               # CLI version for direct processing
├── pipeline.py           # Core processing pipeline
├── thematic_analyzer.py  # Thematic analysis logic
├── data_cleaner.py       # Data cleaning utilities
├── formatter.py          # Output formatting
├── frontend/            # React frontend files
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── render.yaml         # Render deployment configuration
```

## Data Privacy & Security

### What's Included in Repository
- ✅ Application code and configuration
- ✅ Sample/demo data (anonymized)
- ✅ Documentation and setup instructions

### What's Excluded (.gitignore)
- ❌ API keys and secrets (.env files)
- ❌ Real survey data (*.csv files)
- ❌ Virtual environments and cache files
- ❌ IDE configuration files

### Security Best Practices
- API keys are loaded from environment variables only
- Real data files are excluded from version control
- Sample data is anonymized and suitable for public viewing
- CORS is configured for production deployment

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for AI processing | Yes |
| `ENVIRONMENT` | Runtime environment (development/production) | No |
| `PORT` | Server port (default: 8080) | No |
| `HOST` | Server host (default: 0.0.0.0) | No |

## Sample Data Format

The application expects CSV files with these columns:
- `question`: The survey question text
- `response`: The student/participant response
- Additional columns for metadata (course info, demographics, etc.)

## Development

### Running Tests
```bash
# Add test commands here when implemented
python -m pytest
```

### Code Style
This project follows Python best practices:
- FastAPI for API development
- Async/await for I/O operations
- Type hints throughout
- Environment-based configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the GitHub Issues page
- Review the documentation above
- Ensure your environment variables are correctly configured 