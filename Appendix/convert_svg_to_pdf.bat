@echo off
echo Enhanced SVG to PDF conversion with color preservation...
echo.

REM Check if Inkscape is installed
where inkscape >nul 2>nul
if %errorlevel% NEQ 0 (
    echo ERROR: Inkscape is not installed or not in PATH
    echo Please install Inkscape from https://inkscape.org/
    echo Then add it to your system PATH
    pause
    exit /b 1
)

echo Method 1: Using enhanced Inkscape parameters...
echo.

REM Convert Train_set SVG files with enhanced parameters
echo Converting Train_set SVG files with color preservation...
if exist "Train_set" (
    cd Train_set
    for %%f in (*.svg) do (
        echo Converting %%f with enhanced parameters...
        inkscape --export-type=pdf --export-pdf-version=1.5 --export-text-to-path --export-ignore-filters "%%f"
    )
    cd ..
) else (
    echo Train_set folder not found
)

REM Convert Test_set SVG files with enhanced parameters
echo Converting Test_set SVG files with color preservation...
if exist "Test_set" (
    cd Test_set
    for %%f in (*.svg) do (
        echo Converting %%f with enhanced parameters...
        inkscape --export-type=pdf --export-pdf-version=1.5 --export-text-to-path --export-ignore-filters "%%f"
    )
    cd ..
) else (
    echo Test_set folder not found
)

echo.
echo Enhanced conversion completed!
echo If colors are still missing, try Method 2 (PNG conversion)
echo.

REM Method 2: Convert to high-resolution PNG as backup
echo Method 2: Converting to high-resolution PNG as backup...
echo.

if exist "Train_set" (
    cd Train_set
    for %%f in (*.svg) do (
        echo Converting %%f to PNG (300 DPI)...
        inkscape --export-type=png --export-dpi=300 "%%f"
    )
    cd ..
)

if exist "Test_set" (
    cd Test_set
    for %%f in (*.svg) do (
        echo Converting %%f to PNG (300 DPI)...
        inkscape --export-type=png --export-dpi=300 "%%f"
    )
    cd ..
)

echo.
echo All conversions completed!
echo You can use either PDF or PNG files in your LaTeX document.
pause 