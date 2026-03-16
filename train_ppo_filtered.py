#!/usr/bin/env python3
"""
Wrapper script to run PPO training with filtered ROS2 warnings.
This filters out the annoying "Moved backwards in time" warnings.
"""

import subprocess
import sys
import re

# Patterns to filter out (suppress these warnings)
FILTER_PATTERNS = [
    r"Moved backwards in time",
    r"\[robot_state_publisher.*re-publishing joint transforms",
    r"WARN.*robot_state_publisher.*Moved backwards",
]

# Compile regex patterns
compiled_patterns = [re.compile(pattern) for pattern in FILTER_PATTERNS]

def should_filter_line(line):
    """Check if line should be filtered out."""
    for pattern in compiled_patterns:
        if pattern.search(line):
            return True
    return False

def main():
    """Run train_ppo.py and filter output."""
    print("="*60)
    print("Starting PPO Training (with filtered warnings)")
    print("="*60)
    print("")
    
    # Pass any sys arguments down to the root process
    cmd = ['python3', 'train_ppo.py'] + sys.argv[1:]
    
    # Start training process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    try:
        # Read and filter output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                # Only print if not filtered
                if not should_filter_line(line):
                    print(line, end='', flush=True)
        
        # Wait for process to complete
        process.wait()
        return process.returncode
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        process.terminate()
        process.wait()
        return 1

if __name__ == '__main__':
    sys.exit(main())
