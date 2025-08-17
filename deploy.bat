@echo off
echo Initializing Git repository...
git init

echo Adding all files...
git add .

echo Checking status...
git status

echo Committing changes...
git commit -m "Add deployment files for public launch"

echo Adding GitLab remote...
git remote add origin https://gitlab.com/sixteenpython/vriddhi-core.git

echo Pushing to GitLab...
git push -u origin main

echo.
echo Deployment files pushed to GitLab!
echo Next step: Go to https://share.streamlit.io to deploy your app
pause
