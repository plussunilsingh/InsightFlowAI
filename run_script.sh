#!/bin/bash

PORT=8005

# Function to clear the port and stop the app
stop_app() {
    echo "🛑 Cleaning up and stopping any app on port $PORT..."
    # Find process ID LISTENING on the exact port (ignore connected browsers)
    PID=$(lsof -t -i tcp:$PORT -s tcp:LISTEN)
    if [ ! -z "$PID" ]; then
        # Force kill all PIDs found
        echo "$PID" | xargs kill -9 2>/dev/null
        echo "✅ Process(es) on Port $PORT successfully terminated."
        # Give the OS a second to clear the socket
        sleep 1
    else
        echo "✅ Port $PORT is already free. No app is currently running."
    fi
}

# Function to bind to the port and start the app interactively
start_app() {
    echo "🔍 Preparing to bind to port $PORT..."
    # Always clean the port before binding to prevent overlap
    stop_app
    
    # Automatically clean up the port when you press Ctrl+C or close the terminal
    trap stop_app EXIT SIGINT SIGTERM
    
    echo "🚀 Starting Streamlit strictly on port $PORT..."
    echo "👀 Running in FOREGROUND so you can see it. Press Ctrl+C to stop."
    # Run the app normally in the foreground so it auto-opens your browser!
    streamlit run app.py --server.port $PORT
}

# Function to check if the port is currently bound
status_app() {
    PID=$(lsof -t -i tcp:$PORT -s tcp:LISTEN)
    if [ ! -z "$PID" ]; then
        PID_SINGLE_LINE=$(echo $PID | tr '\n' ' ')
        echo "🟢 App is currently RUNNING and bound to port $PORT (PID: $PID_SINGLE_LINE)."
    else
        echo "🔴 App is STOPPED. Nothing is running on port $PORT."
    fi
}

# Process the command line argument
case "$1" in
    start)
        start_app
        ;;
    stop)
        stop_app
        ;;
    status)
        status_app
        ;;
    restart)
        stop_app
        echo "⏳ Restarting..."
        sleep 1
        start_app
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
esac