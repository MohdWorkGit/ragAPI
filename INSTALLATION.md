# Installation Guide - Enhanced Writer Extraction

This guide provides detailed instructions for installing all dependencies needed for the enhanced journalist/writer name extraction system on Linux.

## Table of Contents

- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
- [Manual Installation](#manual-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Platform-Specific Notes](#platform-specific-notes)

---

## Quick Start

### Option 1: Automated Installation (Recommended)

```bash
# Make scripts executable (if not already)
chmod +x install_dependencies.sh

# Run the installation script
./install_dependencies.sh
```

This script will:
- Check system requirements
- Install Python dependencies
- Download spaCy language models
- Create necessary directories
- Verify the installation

### Option 2: Quick Install (Minimal Interaction)

```bash
chmod +x quick_install.sh
./quick_install.sh
```

Use this for automated/CI environments or if you want minimal prompts.

---

## Detailed Installation

### Step 1: Install System Dependencies (If Needed)

If you don't have Python 3.8+ and development tools installed:

```bash
chmod +x install_system_deps.sh
./install_system_deps.sh
```

**Supported Distributions:**
- Ubuntu/Debian
- CentOS/RHEL/Fedora
- Arch Linux
- Alpine Linux

**Manual Installation (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-dev build-essential git
```

**Manual Installation (CentOS/RHEL):**
```bash
sudo yum install -y python3 python3-pip python3-devel gcc make git
```

### Step 2: Install Python Dependencies

```bash
pip3 install -r requirements.txt --user
```

**Dependencies installed:**
- `pymupdf` (≥1.23.0) - PDF block extraction
- `spacy` (≥3.7.0) - NER and NLP
- `transformers` (≥4.35.0) - Advanced NLP models
- `torch` (≥2.1.0) - Deep learning framework

### Step 3: Download spaCy Language Models

**English model (required):**
```bash
python3 -m spacy download en_core_web_sm
```

**Multilingual model (optional, for Arabic support):**
```bash
python3 -m spacy download xx_ent_wiki_sm
```

### Step 4: Create Required Directories

```bash
mkdir -p writers_db docs
```

---

## Manual Installation

If you prefer to install each component manually:

### 1. Check Python Version

```bash
python3 --version  # Should be 3.8 or higher
```

### 2. Upgrade pip

```bash
python3 -m pip install --upgrade pip
```

### 3. Install Core Dependencies

```bash
# PyMuPDF for PDF processing
pip3 install pymupdf --user

# spaCy for NER
pip3 install spacy --user

# Optional: Transformers and PyTorch
pip3 install transformers torch --user
```

### 4. Download spaCy Models

```bash
python3 -m spacy download en_core_web_sm
```

### 5. Verify Installation

```bash
python3 -c "import pymupdf; print('PyMuPDF:', pymupdf.__version__)"
python3 -c "import spacy; print('spaCy:', spacy.__version__)"
python3 -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('Model loaded')"
```

---

## Verification

### Run the Test Suite

```bash
python3 test_enhanced_extraction.py
```

### Test with a PDF

```bash
python3 test_enhanced_extraction.py /path/to/your/sample.pdf
```

### Quick Verification

```python
from writer_manager import WriterManager

wm = WriterManager()
text = "By John Smith\n\nThis is a test article."
writers = wm.extract_writer_names(text)
print(f"Found: {writers}")
```

Expected output: `Found: ['John Smith']`

---

## Troubleshooting

### Issue: "PyMuPDF not available"

**Solution:**
```bash
pip3 install pymupdf --user
```

If build fails, install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc make
```

### Issue: "spaCy model not found"

**Solution:**
```bash
python3 -m spacy download en_core_web_sm
```

If download fails, try with sudo:
```bash
sudo python3 -m spacy download en_core_web_sm
```

### Issue: "Permission denied" when installing packages

**Solution 1 - User install (recommended):**
```bash
pip3 install --user -r requirements.txt
```

**Solution 2 - Virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Solution 3 - System-wide (not recommended):**
```bash
sudo pip3 install -r requirements.txt
```

### Issue: "ModuleNotFoundError" after installation

**Solution:**
Check if user site-packages is in PATH:
```bash
python3 -m site --user-site
```

Add to `~/.bashrc` or `~/.profile`:
```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.X/site-packages:$PYTHONPATH"
```

Then reload:
```bash
source ~/.bashrc
```

### Issue: PyTorch installation too slow or fails

PyTorch is optional for basic functionality. You can:

**Option 1 - Install CPU-only version (faster):**
```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu --user
```

**Option 2 - Skip PyTorch:**
Comment out `torch` in `requirements.txt` and install without it.

### Issue: Installation script hangs

- Check your internet connection
- Try running with verbose output: `bash -x install_dependencies.sh`
- Install dependencies one at a time manually

---

## Platform-Specific Notes

### Ubuntu/Debian

```bash
# Full installation
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-dev build-essential
./install_dependencies.sh
```

### CentOS/RHEL 8+

```bash
# Enable PowerTools for development packages
sudo yum install -y dnf-plugins-core
sudo yum config-manager --set-enabled powertools

# Install dependencies
./install_system_deps.sh
./install_dependencies.sh
```

### Fedora

```bash
sudo dnf install python3 python3-pip python3-devel gcc make
./install_dependencies.sh
```

### Arch Linux

```bash
sudo pacman -S python python-pip base-devel
./install_dependencies.sh
```

### Alpine Linux

```bash
sudo apk add python3 python3-dev py3-pip build-base
./install_dependencies.sh
```

### WSL (Windows Subsystem for Linux)

Works the same as native Linux. Follow Ubuntu/Debian instructions:
```bash
./install_system_deps.sh
./install_dependencies.sh
```

### Docker

Create a Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application files
COPY . .

# Create directories
RUN mkdir -p writers_db docs

CMD ["python", "test_enhanced_extraction.py"]
```

Build and run:
```bash
docker build -t writer-extraction .
docker run -v $(pwd)/docs:/app/docs writer-extraction
```

---

## Verification Checklist

After installation, verify each component:

- [ ] Python 3.8+ installed: `python3 --version`
- [ ] pip3 installed: `pip3 --version`
- [ ] PyMuPDF installed: `python3 -c "import pymupdf"`
- [ ] spaCy installed: `python3 -c "import spacy"`
- [ ] spaCy model downloaded: `python3 -c "import spacy; spacy.load('en_core_web_sm')"`
- [ ] Directories created: `ls -d writers_db docs`
- [ ] Test script runs: `python3 test_enhanced_extraction.py`

---

## Environment Variables (Optional)

Set these in `~/.bashrc` or `~/.profile`:

```bash
# Add user bin to PATH
export PATH="$HOME/.local/bin:$PATH"

# Python user site-packages
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"

# Increase PyTorch memory limit (if using GPU)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## Minimal System Requirements

- **OS:** Linux (any modern distribution)
- **Python:** 3.8 or higher
- **RAM:** 2GB minimum, 4GB recommended
- **Disk:** 2GB free space for dependencies
- **CPU:** Any modern CPU (GPU optional)

---

## Next Steps

After successful installation:

1. **Read the documentation:** `ENHANCED_EXTRACTION_GUIDE.md`
2. **Run tests:** `python3 test_enhanced_extraction.py`
3. **Try with your PDFs:** Place PDFs in `docs/` folder
4. **Use in your code:** See examples in `test_enhanced_extraction.py`

---

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Run the verification script: `./install_dependencies.sh` (verification step)
3. Check logs for specific error messages
4. Review `ENHANCED_EXTRACTION_GUIDE.md` for usage help

---

## Uninstallation

To remove installed packages:

```bash
pip3 uninstall pymupdf spacy transformers torch -y
```

To remove spaCy models:
```bash
python3 -m spacy download en_core_web_sm --uninstall
```

---

## Additional Resources

- **PyMuPDF Documentation:** https://pymupdf.readthedocs.io/
- **spaCy Documentation:** https://spacy.io/usage
- **Project Documentation:** `ENHANCED_EXTRACTION_GUIDE.md`
- **Writer Feature README:** `WRITER_FEATURE_README.md`
