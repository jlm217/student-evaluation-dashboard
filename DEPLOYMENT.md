# Deployment Guide

## Pre-Deployment Security Checklist

### ✅ Environment Variables
- [ ] API keys moved to environment variables (not hardcoded)
- [ ] `.env.example` file created with template
- [ ] `.env` files properly excluded in `.gitignore`
- [ ] No secrets committed to repository

### ✅ Data Security  
- [ ] Real survey data excluded from repository
- [ ] Sample data is anonymized and safe for public viewing
- [ ] No personal information in committed files
- [ ] Temporary files and logs excluded

### ✅ Code Security
- [ ] No hardcoded passwords or API keys
- [ ] CORS properly configured for production
- [ ] Error messages don't expose sensitive information
- [ ] Input validation in place

## GitHub Preparation

### 1. Review Files Being Committed
```bash
# Check what files will be committed
git status
git add .
git status

# Review the diff before committing
git diff --cached
```

### 2. Verify .gitignore is Working
```bash
# These commands should show no sensitive files
git ls-files | grep -E "\.(env|csv)$" | grep -v "example\|sample"
```

### 3. Initial Commit
```bash
git init
git add .
git commit -m "Initial commit: Qualitative Coding Agent v2"
git branch -M main
git remote add origin https://github.com/yourusername/qualitative-coding-agent.git
git push -u origin main
```

## Render Deployment

### 1. Create Render Account
- Go to [render.com](https://render.com)
- Sign up with GitHub account

### 2. Create New Web Service
- Connect GitHub repository
- Select your `qualitative-coding-agent` repository
- Render will auto-detect the `render.yaml` configuration

### 3. Configure Environment Variables
In the Render dashboard, add:
- `GEMINI_API_KEY`: Your actual Google Gemini API key

### 4. Deploy
- Render will automatically build and deploy
- Monitor the deploy logs for any issues
- Your app will be available at `https://your-app-name.onrender.com`

## Post-Deployment Testing

### 1. Health Check
```bash
curl https://your-app-name.onrender.com/api/health
```

### 2. Upload Test
- Go to your deployed URL
- Try uploading the sample CSV file
- Verify the analysis completes successfully

### 3. WebSocket Test
- Check that real-time progress updates work
- Monitor browser developer console for errors

## Security Considerations for Production

### Environment Variables
- `GEMINI_API_KEY`: Keep secret, set in Render dashboard only
- `ENVIRONMENT`: Set to "production" 
- `PORT`: Automatically set by Render

### CORS Configuration
The app is currently configured with `allow_origins=["*"]` for development.
For production, consider restricting to your domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Data Handling
- Uploaded files are processed in-memory and temporary directories
- No persistent storage of user data
- Cache automatically expires after 1 hour
- Consider adding rate limiting for production use

## Troubleshooting

### Common Issues

1. **Build Fails**
   - Check that `requirements.txt` is complete
   - Verify Python version compatibility

2. **App Won't Start**  
   - Check environment variables are set correctly
   - Review Render logs for specific errors

3. **API Errors**
   - Verify GEMINI_API_KEY is valid and has quota
   - Check network connectivity and API limits

4. **Frontend Issues**
   - Ensure static files are served correctly
   - Check browser console for JavaScript errors

### Render-Specific Tips

- Free tier sleeps after 15 minutes of inactivity
- Cold starts may take 30-60 seconds
- Logs are available in the Render dashboard
- Automatic deploys on GitHub pushes

## Monitoring

After deployment, monitor:
- Application health at `/api/health`
- Error rates in Render logs
- API usage and quotas
- User feedback and issues

## Updates

To deploy updates:
1. Make changes locally
2. Test thoroughly
3. Commit and push to GitHub
4. Render will automatically redeploy 