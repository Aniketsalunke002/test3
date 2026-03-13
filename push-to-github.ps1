# Push to GitHub - Run this after installing Git and GitHub CLI
# 1. Install Git: https://git-scm.com/download/win
# 2. Install GitHub CLI: https://cli.github.com/ (or: winget install GitHub.cli)

$ErrorActionPreference = "Stop"
$repoName = "test-3"  # GitHub repo names use hyphens, not spaces

Write-Host "Pushing to GitHub repo: $repoName" -ForegroundColor Cyan

# Navigate to script directory
Set-Location $PSScriptRoot

# Initialize git if needed
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
}

# Add all files
Write-Host "Adding files..." -ForegroundColor Yellow
git add .

# Commit
Write-Host "Committing..." -ForegroundColor Yellow
git commit -m "Initial commit - assignment 3" 2>$null
if ($LASTEXITCODE -ne 0) {
    # Might already be committed
    Write-Host "Nothing new to commit (or already committed)" -ForegroundColor Gray
}

# Create repo on GitHub and push (requires: gh auth login)
Write-Host "Creating GitHub repo '$repoName' and pushing..." -ForegroundColor Yellow
gh repo create $repoName --public --source=. --remote=origin --push

Write-Host "`nDone! Your code is at: https://github.com/$((gh api user -q .login))/test-3" -ForegroundColor Green
