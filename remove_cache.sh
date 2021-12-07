find . -name ".ipynb_checkpoints" -type d -exec rm -r {} \; 
find . -name "__pycache__" -type d -exec rm -r {} \; 
find . -name ".vscode" -type d -exec rm -r {} \; 
find . -name ".lh" -type d -exec rm -r {} \; 
find . -name ".history" -type d -exec rm -r {} \;
find ./data -name "*.mp4" -type f -exec rm -r {} \;
find ./data -name "*.webm" -type f -exec rm -r {} \;
# find ./data -type f -exec rm -r {} \;