import subprocess
import time

def get_session_names():
    # Get the names of running tmux sessions
    output = subprocess.run(['tmux', 'list-sessions'], capture_output=True, text=True)
    session_lines = output.stdout.strip().split('\n')
    return [line.split(':')[0].strip() for line in session_lines]

while True:
    # Get a list of running tmux session names
    sessions = get_session_names()

    # Loop through the list of sessions
    for session in sessions:
        # Connect to the tmux session for 1 second
        capture = subprocess.run(['tmux', 'capture-pane', '-t', session, '-p'], capture_output=True, text=True)
        first_line = capture.stdout.strip().split('\n')[0]

        # Show the output of the session
        print(f"Session '{session}':")
        print(first_line)
        print()

    # Wait for a second before checking again
    time.sleep(1)
