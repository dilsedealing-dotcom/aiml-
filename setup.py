import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    directories = ['data', 'onnx_models', 'ea']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")

def run_tests():
    """Run system tests"""
    print("Running system tests...")
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ All tests passed")
            return True
        else:
            print(f"✗ Tests failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return False

def main():
    print("MT5 AI Trading System Setup")
    print("=" * 40)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Install requirements
    if not install_requirements():
        print("Setup failed at requirements installation")
        return
    
    # Step 3: Run tests
    if run_tests():
        print("\n" + "=" * 40)
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Ensure MetaTrader 5 is installed and running")
        print("2. Run 'python main.py' to start the trading system")
        print("3. Copy the EA file to your MT5 Experts folder")
        print("4. Configure the EA with your WebSocket URL")
        print("\nFor cloud deployment:")
        print("- Run 'python cloud_deploy.py' for Google Cloud setup")
        print("- Run 'python flask_server.py' to start the WebSocket server")
    else:
        print("\n" + "=" * 40)
        print("⚠️ Setup completed with warnings")
        print("Some tests failed, but core functionality should work")
        print("Check the test output above for details")

if __name__ == "__main__":
    main()