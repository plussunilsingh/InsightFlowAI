#!/bin/bash

PORT=8005

# Function to clear the port and stop the app
stop_app() {
    echo "🛑 Cleaning up and stopping any app on port $PORT..."
    # Find process ID on the exact port
    PID=$(lsof -t -i tcp:$PORT)
    if [ ! -z "$PID" ]; then
        # Force kill all PIDs found (xargs handles multiple lines cleanly)
        echo "$PID" | xargs kill -9 2>/dev/null
        echo "✅ Process(es) on Port $PORT successfully terminated."
        # Give the OS a second to clear the socket
        sleep 1
    else
        echo "✅ Port $PORT is already free. No app is currently running."
    fi
}

# Function to bind to the port and start the app
start_app() {
    echo "🔍 Preparing to bind to port $PORT..."
    # Always clean the port before binding to prevent overlap
    stop_app
    
    echo "🚀 Starting Streamlit strictly on port $PORT..."
    # Run the app in the background. </dev/null is CRITICAL to fully detach from the terminal!
    nohup streamlit run app.py --server.port $PORT > streamlit.log 2>&1 < /dev/null &
    
    # Wait a couple of seconds to verify it successfully bound to the port
    sleep 3
    PID=$(lsof -t -i tcp:$PORT)
    if [ ! -z "$PID" ]; then
        # Format the PID output to be on a single line in case of multiple processes
        PID_SINGLE_LINE=$(echo $PID | tr '\n' ' ')
        echo "✅ App successfully started and bound to port $PORT (PID: $PID_SINGLE_LINE)!"
        echo "📝 Output is being logged to 'streamlit.log'. You can safely close this terminal."
    else
        echo "❌ Failed to bind to port $PORT. Check streamlit.log for details."
    fi
}

# Function to check if the port is currently bound
status_app() {
    PID=$(lsof -t -i tcp:$PORT)
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