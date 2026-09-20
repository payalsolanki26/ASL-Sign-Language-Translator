#!/usr/bin/env bash
# Sign Reader - Live ASL Hand Sign Translator Launcher (Bash / WSL / Git Bash)

cd "$(dirname "$0")"

echo "========================================================"
echo "  Starting Sign Reader - Live ASL Translator"
echo "========================================================"
echo ""
echo "URL: http://127.0.0.1:8000"
echo "Press CTRL+C to stop the server anytime."
echo ""

# Try opening the default browser based on OS
if command -v start &> /dev/null; then
    start http://127.0.0.1:8000 &
elif command -v xdg-open &> /dev/null; then
    xdg-open http://127.0.0.1:8000 &
elif command -v open &> /dev/null; then
    open http://127.0.0.1:8000 &
fi

# Run the FastAPI Uvicorn Server
python -m uvicorn asl.serve.api:app --app-dir src --host 127.0.0.1 --port 8000
