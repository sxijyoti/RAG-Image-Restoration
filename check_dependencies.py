"""
Check and install dependencies for RAG-Image-Restoration.

Core requirements:
- numpy
- Pillow (PIL)
- torch
- open_clip_torch (for Phase 2: DA-CLIP encoder)
- faiss-cpu or faiss-gpu (for Phase 3: Retrieval)

Run: python check_dependencies.py
"""

import sys
import subprocess


def check_import(package: str, import_name: str = None) -> bool:
    """Check if a package is installed."""
    if import_name is None:
        import_name = package
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def print_status(package: str, installed: bool):
    """Print status of a package."""
    status = "✓ INSTALLED" if installed else "✗ MISSING"
    print(f"  {package:.<40} {status}")


def main():
    print("\n" + "="*60)
    print("RAG-Image-Restoration Dependency Check")
    print("="*60 + "\n")
    
    # Core dependencies
    print("CORE DEPENDENCIES:")
    core_deps = {
        "numpy": "numpy",
        "Pillow": "PIL",
        "torch": "torch",
    }
    
    core_ok = True
    for package, import_name in core_deps.items():
        installed = check_import(package, import_name)
        print_status(package, installed)
        if not installed:
            core_ok = False
    
    # Optional Phase 2 dependencies
    print("\nPHASE 2: DA-CLIP Encoder (Optional)")
    phase2_deps = {
        "open-clip-torch": "open_clip",
    }
    
    phase2_ok = True
    for package, import_name in phase2_deps.items():
        installed = check_import(package, import_name)
        print_status(package, installed)
        if not installed:
            phase2_ok = False
    
    # Optional Phase 3 dependencies
    print("\nPHASE 3: FAISS Retrieval (Optional)")
    phase3_deps = {
        "faiss-cpu": "faiss",
    }
    
    phase3_ok = True
    for package, import_name in phase3_deps.items():
        installed = check_import(package, import_name)
        print_status(package, installed)
        if not installed:
            phase3_ok = False
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if core_ok:
        print("✓ Core dependencies: SATISFIED")
        print("  You can run Phase 1 (Patch Extraction)")
    else:
        print("✗ Core dependencies: MISSING")
        print("  Install with: pip install numpy Pillow torch")
    
    if phase2_ok:
        print("✓ Phase 2 (DA-CLIP): SATISFIED")
    else:
        print("✗ Phase 2 (DA-CLIP): Missing dependencies")
        print("  Install with: pip install open-clip-torch")
    
    if phase3_ok:
        print("✓ Phase 3 (FAISS): SATISFIED")
    else:
        print("✗ Phase 3 (FAISS): Missing dependencies")
        print("  Install CPU version: pip install faiss-cpu")
        print("  Or GPU version: pip install faiss-gpu")
    
    print("\n" + "="*60 + "\n")
    
    # Suggest auto-install
    if not (core_ok and phase2_ok and phase3_ok):
        missing = []
        if not core_ok:
            missing.append("numpy Pillow torch")
        if not phase2_ok:
            missing.append("open-clip-torch")
        if not phase3_ok:
            missing.append("faiss-cpu")
        
        print("To install missing packages, run:")
        print(f"  pip install {' '.join(missing)}\n")


if __name__ == "__main__":
    main()
