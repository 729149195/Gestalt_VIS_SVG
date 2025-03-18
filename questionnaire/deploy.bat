@echo off
start cmd /k "git add . && git commit -m "auto push" && npm run build && git push && npm run deploy && node resetCount.js"