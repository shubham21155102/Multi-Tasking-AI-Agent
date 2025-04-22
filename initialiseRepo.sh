content=$(cat <<EOF
# Multi-Tasking AI Agent

This repository contains a multi-tasking AI agent capable of performing various tasks using different tools. The agent is designed with a modular and extensible architecture, allowing for the easy addition of new tools and functionalities.

## Features
- Modular architecture  
- Extensible toolset  
- Support for multiple tasks  
- Easy to use  
EOF
)


echo "$content" > README.md
git init
git add README.md
git commit -m "Initial commit with README"
git branch -M main
git remote add origin https://github.com/shubham21155102/Multi-Tasking-AI-Agent.git
git push -u origin main