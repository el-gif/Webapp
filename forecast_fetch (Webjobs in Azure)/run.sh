#!/bin/bash

echo "=============================="
echo " Starting WebJob Execution "
echo "=============================="

# 1️⃣ Debugging: Print Environment Info
echo "🔍 Checking Environment Variables..."
echo "Current PATH: $PATH"
echo "Current User: $(whoami)"
echo "Home Directory: $HOME"
echo "Current Working Directory: $(pwd)"

# 2️⃣ Debugging: Check if Python and Pip Exist
echo "🔍 Checking Python and Pip Availability..."
PYTHON_PATH=$(command -v python3)
PIP_PATH=$(command -v pip)

if [ -z "$PYTHON_PATH" ]; then
    echo "⚠️ Python3 not found in PATH! Trying to locate..."
    PYTHON_PATH="/usr/bin/python3"
fi

if [ -z "$PIP_PATH" ]; then
    echo "⚠️ Pip not found in PATH! Trying to locate..."
    PIP_PATH="/usr/bin/pip"
fi

# Verify Python and Pip Paths
echo "✅ Using Python: $PYTHON_PATH"
echo "✅ Using Pip: $PIP_PATH"

# 3️⃣ Debugging: Check Installed Python Versions
echo "🔍 Listing Installed Python Versions..."
ls -l /usr/bin/python* /usr/local/bin/python* 2>/dev/null

# 4️⃣ Debugging: Check Installed Pip Versions
echo "🔍 Listing Installed Pip Versions..."
ls -l /usr/bin/pip* /usr/local/bin/pip* 2>/dev/null

# 5️⃣ Try Activating Virtual Environment
echo "🔍 Checking Virtual Environment..."
ANTENV_PATH=$(find /home/site/wwwroot /tmp -maxdepth 3 -type d -name "antenv" 2>/dev/null | head -n 1)

if [ -n "$ANTENV_PATH" ]; then
    echo "✅ Found Virtual Environment at: $ANTENV_PATH"
    if [ -f "$ANTENV_PATH/bin/activate" ]; then
        echo "✅ Activating Virtual Environment..."
        source "$ANTENV_PATH/bin/activate"
        PYTHON_PATH=$(command -v python3)  # Update path after activation
        PIP_PATH=$(command -v pip)         # Update path after activation
    else
        echo "⚠️ Virtual environment found, but activation script is missing!"
    fi
else
    echo "⚠️ Virtual environment 'antenv' not found! Proceeding without it."
fi

# 6️⃣ Debugging: Ensure Python is Installed
if [ ! -f "$PYTHON_PATH" ]; then
    echo "⚠️ Python3 is missing! Attempting to install..."
    apt-get update && apt-get install -y python3 python3-pip
    PYTHON_PATH=$(command -v python3)
    PIP_PATH=$(command -v pip)
    echo "✅ Installed Python: $PYTHON_PATH"
    echo "✅ Installed Pip: $PIP_PATH"
fi

# 7️⃣ Debugging: List Running Processes
echo "🔍 Listing Running Processes..."
ps aux | grep python

# 8️⃣ Debugging: Check Open Ports
echo "🔍 Checking Open Ports..."
netstat -tulnp | grep LISTEN

# 9️⃣ Install Dependencies
if [ -n "$PIP_PATH" ]; then
    echo "🔍 Installing Required Packages..."
    $PIP_PATH install --user ecmwf-opendata pandas
else
    echo "❌ Error: Pip is not available! Skipping package installation."
fi

# 🔟 Run the Python Script
if [ -n "$PYTHON_PATH" ]; then
    echo "🚀 Running forecast_fetch.py..."
    $PYTHON_PATH forecast_fetch.py
else
    echo "❌ Error: Python3 is not available! Exiting."
    exit 1
fi

echo "=============================="
echo " WebJob Execution Completed "
echo "=============================="
