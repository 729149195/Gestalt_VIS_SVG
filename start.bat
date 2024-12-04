@echo off
start powershell -NoExit -Command "cd Gestalt_SVG;  npm run dev"
start powershell -NoExit -Command "cd Gestalt_API; python app.py"
