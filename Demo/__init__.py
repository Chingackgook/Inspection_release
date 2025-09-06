def run_python_module(module_name: str, args: list = None):
    """
    Run a Python module in subprocess synchronously with real-time output
    
    Args:
        module_name: Module name to run (e.g., 'http.server', 'json.tool')
        args: Additional arguments for the module
    
    Returns:
        CompletedProcess object with returncode
    """
    import subprocess
    import sys
    if args is None:
        args = []
    # Build command: python -m module_name [args...]
    cmd = [sys.executable, '-m', module_name] + args
    try:
        # Run the command with real-time output (no capture_output)
        result = subprocess.run(cmd, text=True, timeout=30)
        return result
    except subprocess.TimeoutExpired:
        print(f"Timeout: {' '.join(cmd)}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None