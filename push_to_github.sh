#!/bin/bash

echo "========================================"
echo "GitHub Repository Setup Helper"
echo "========================================"
echo ""
echo "This script will help you push your trading bot to GitHub."
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Error: Git repository not initialized!"
    exit 1
fi

echo "Step 1: Create a new repository on GitHub"
echo "----------------------------------------"
echo "1. Go to https://github.com/new"
echo "2. Name it: crypto-iq-burst-bot"
echo "3. Make it public or private (your choice)"
echo "4. DON'T initialize with README, .gitignore, or license"
echo "5. Click 'Create repository'"
echo ""
read -p "Press Enter when you've created the repository..."

echo ""
echo "Step 2: Add GitHub remote"
echo "----------------------------------------"
echo "Enter your GitHub username:"
read github_username

if [ -z "$github_username" ]; then
    echo "Error: GitHub username is required!"
    exit 1
fi

# Add remote origin
git remote add origin "https://github.com/$github_username/crypto-iq-burst-bot.git"

echo ""
echo "Remote added: https://github.com/$github_username/crypto-iq-burst-bot.git"
echo ""

echo "Step 3: Push to GitHub"
echo "----------------------------------------"
echo "Pushing your code to GitHub..."
echo ""

# Push to GitHub
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ SUCCESS!"
    echo "========================================"
    echo ""
    echo "Your repository is now available at:"
    echo "https://github.com/$github_username/crypto-iq-burst-bot"
    echo ""
    echo "Next steps:"
    echo "1. Add a description on GitHub"
    echo "2. Add topics: crypto, trading-bot, bitcoin, ai"
    echo "3. Consider adding GitHub Actions for testing"
    echo "4. Update README with your specific setup"
    echo ""
else
    echo ""
    echo "⚠️  Push failed. Common solutions:"
    echo ""
    echo "1. If authentication failed:"
    echo "   - Create a Personal Access Token at:"
    echo "     https://github.com/settings/tokens"
    echo "   - Use the token as your password"
    echo ""
    echo "2. If repository doesn't exist:"
    echo "   - Make sure you created it on GitHub first"
    echo "   - Check the repository name matches"
    echo ""
    echo "3. Manual push commands:"
    echo "   git remote set-url origin https://github.com/$github_username/crypto-iq-burst-bot.git"
    echo "   git push -u origin main"
    echo ""
fi
