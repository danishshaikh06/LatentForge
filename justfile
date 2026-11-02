# Add, commit and push
acp message="Update":
    git add .
    git commit -m "{{message}}"
    git push
