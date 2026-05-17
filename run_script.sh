#!/bin/bash

# Function to run the ingest pipeline
prepare() {
    echo "🚀 Running data ingestion pipeline (ingest.py)..."
    python ingest.py
    echo "✅ Ingestion complete!"
}

# Function to run the Streamlit application
run() {
    echo "🚀 Starting ContextIQ Streamlit App..."
    streamlit run app.py
}

# Function to stop the Streamlit application
stop() {
    echo "🛑 Stopping ContextIQ Streamlit App..."
    pkill -f "streamlit run app.py"
    echo "✅ App stopped!"
}

# Command line argument handling
case "$1" in
    prepare)
        prepare
        ;;
    run)
        run
        ;;
    stop)
        stop
        ;;
    *)
        echo "Usage: ./run_script.sh [command]"
        echo ""
        echo "Commands:"
        echo "  prepare   - Runs ingest.py to load documents and create Chroma DB"
        echo "  run       - Runs the Streamlit application (app.py)"
        echo "  stop      - Stops the running Streamlit application"
        echo ""
        echo "Example:"
        echo "  ./run_script.sh prepare"
        echo "  ./run_script.sh run"
        echo "  ./run_script.sh stop"
        exit 1
        ;;
esac