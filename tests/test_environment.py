#!/usr/bin/env python3
"""
Quick environment test script
Run this to verify your Python setup is working
"""

print("="*60)
print("🧪 TESTING PYTHON ENVIRONMENT")
print("="*60)

# Test 1: Python version
print("\n1️⃣ Testing Python version...")
import sys
print(f"   ✅ Python {sys.version}")
print(f"   ✅ Executable: {sys.executable}")

# Test 2: PyQt5
print("\n2️⃣ Testing PyQt5...")
try:
    from PyQt5 import QtWidgets, QtCore
    print("   ✅ PyQt5 imported successfully")
    print(f"   ✅ Qt version: {QtCore.QT_VERSION_STR}")
except ImportError as e:
    print(f"   ❌ ERROR: {e}")
    print("   Run: pip install PyQt5")
    sys.exit(1)

# Test 3: Matplotlib
print("\n3️⃣ Testing Matplotlib...")
try:
    import matplotlib
    print("   ✅ Matplotlib imported successfully")
    print(f"   ✅ Version: {matplotlib.__version__}")
except ImportError as e:
    print(f"   ❌ ERROR: {e}")
    print("   Run: pip install matplotlib")
    sys.exit(1)

# Test 4: NumPy
print("\n4️⃣ Testing NumPy...")
try:
    import numpy
    print("   ✅ NumPy imported successfully")
    print(f"   ✅ Version: {numpy.__version__}")
except ImportError as e:
    print(f"   ❌ ERROR: {e}")
    print("   Run: pip install numpy")
    sys.exit(1)

# Test 5: Platform check
print("\n5️⃣ Testing Platform...")
import platform
os_type = platform.system()
print(f"   ℹ️  OS: {os_type}")
if os_type != "Windows":
    print("   ⚠️  WARNING: AC shared memory only works on Windows!")
    print("   ⚠️  You won't be able to connect to Assetto Corsa on Mac/Linux")
else:
    print("   ✅ Windows detected - shared memory will work!")

# Test 6: Simple Qt Application
print("\n6️⃣ Testing Qt Application...")
try:
    app = QtWidgets.QApplication(sys.argv)
    print("   ✅ Qt Application created successfully")
    print("   ✅ (Not showing window, just testing)")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    sys.exit(1)

# Final result
print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nYour environment is ready. You can now run:")
print("   python main.py")
print("\n")
