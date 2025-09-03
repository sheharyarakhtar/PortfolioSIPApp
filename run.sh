#!/bin/bash

echo "🎯 Starting MPT-Based SIP Portfolio Optimizer..."

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "🐳 Running in Docker container"
    streamlit run main.py --server.port=1000 --server.address=0.0.0.0
else
    echo "💻 Running locally"
    
    # Check if Python and pip are available
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
        exit 1
    fi
    
    # Install dependencies if needed
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    
    echo "🚀 Starting Streamlit app..."
    streamlit run main.py
fi
