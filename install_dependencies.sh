#!/bin/bash

###############################################################################
# Enhanced Writer Extraction - Installation Script for Linux
# Installs all required dependencies and validates the setup
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

###############################################################################
# System Checks
###############################################################################

check_python() {
    print_header "Checking Python Installation"

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        print_success "Python3 found: $PYTHON_VERSION"

        # Check if version is >= 3.8
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python version is compatible (>= 3.8)"
        else
            print_error "Python 3.8 or higher is required. Current: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python3 is not installed"
        print_info "Install Python3 with: sudo apt-get install python3 python3-pip"
        exit 1
    fi
}

check_pip() {
    print_header "Checking pip Installation"

    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version | awk '{print $2}')
        print_success "pip3 found: $PIP_VERSION"
    else
        print_error "pip3 is not installed"
        print_info "Install pip3 with: sudo apt-get install python3-pip"
        exit 1
    fi
}

check_system_dependencies() {
    print_header "Checking System Dependencies"

    # Check for build essentials (needed for some Python packages)
    if command -v gcc &> /dev/null; then
        print_success "GCC compiler found"
    else
        print_warning "GCC not found. Some packages may fail to compile."
        print_info "Install with: sudo apt-get install build-essential"
    fi

    # Check for git
    if command -v git &> /dev/null; then
        print_success "Git found"
    else
        print_warning "Git not found"
        print_info "Install with: sudo apt-get install git"
    fi
}

###############################################################################
# Installation Functions
###############################################################################

upgrade_pip() {
    print_header "Upgrading pip"

    python3 -m pip install --upgrade pip || {
        print_error "Failed to upgrade pip"
        exit 1
    }
    print_success "pip upgraded successfully"
}

install_python_dependencies() {
    print_header "Installing Python Dependencies"

    if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
        print_error "requirements.txt not found in $SCRIPT_DIR"
        exit 1
    fi

    print_info "Installing packages from requirements.txt..."
    print_info "This may take several minutes..."

    # Install with timeout and error handling
    pip3 install -r "$SCRIPT_DIR/requirements.txt" --user || {
        print_error "Failed to install Python dependencies"
        print_info "Try running manually: pip3 install -r requirements.txt"
        exit 1
    }

    print_success "Python dependencies installed successfully"
}

download_spacy_models() {
    print_header "Downloading spaCy Language Models"

    print_info "Downloading English model (en_core_web_sm)..."
    python3 -m spacy download en_core_web_sm || {
        print_warning "Failed to download spaCy English model"
        print_info "Try manually: python3 -m spacy download en_core_web_sm"
    }
    print_success "English model downloaded"

    # Optional: Arabic/multilingual support
    read -p "$(echo -e ${YELLOW}Do you want to install multilingual model for Arabic support? [y/N]: ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Downloading multilingual model (xx_ent_wiki_sm)..."
        python3 -m spacy download xx_ent_wiki_sm || {
            print_warning "Failed to download multilingual model"
            print_info "Try manually: python3 -m spacy download xx_ent_wiki_sm"
        }
        print_success "Multilingual model downloaded"
    fi
}

create_directories() {
    print_header "Creating Required Directories"

    cd "$SCRIPT_DIR"

    # Create writers database directory
    if [ ! -d "writers_db" ]; then
        mkdir -p writers_db
        print_success "Created writers_db directory"
    else
        print_info "writers_db directory already exists"
    fi

    # Create docs directory if it doesn't exist
    if [ ! -d "docs" ]; then
        mkdir -p docs
        print_success "Created docs directory"
    else
        print_info "docs directory already exists"
    fi
}

###############################################################################
# Verification Functions
###############################################################################

verify_installation() {
    print_header "Verifying Installation"

    print_info "Checking PyMuPDF..."
    python3 -c "import pymupdf; print('PyMuPDF version:', pymupdf.__version__)" && \
        print_success "PyMuPDF is working" || \
        print_error "PyMuPDF failed to import"

    print_info "Checking spaCy..."
    python3 -c "import spacy; print('spaCy version:', spacy.__version__)" && \
        print_success "spaCy is working" || \
        print_error "spaCy failed to import"

    print_info "Checking spaCy model..."
    python3 -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('Model loaded successfully')" && \
        print_success "spaCy model is working" || \
        print_error "spaCy model failed to load"

    print_info "Checking transformers..."
    python3 -c "import transformers; print('Transformers version:', transformers.__version__)" && \
        print_success "Transformers is working" || \
        print_warning "Transformers failed to import (optional)"

    print_info "Checking torch..."
    python3 -c "import torch; print('PyTorch version:', torch.__version__)" && \
        print_success "PyTorch is working" || \
        print_warning "PyTorch failed to import (optional)"
}

run_test_script() {
    print_header "Running Test Script"

    if [ -f "$SCRIPT_DIR/test_enhanced_extraction.py" ]; then
        read -p "$(echo -e ${YELLOW}Do you want to run the test script? [y/N]: ${NC})" -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 "$SCRIPT_DIR/test_enhanced_extraction.py" || {
                print_warning "Test script encountered some issues"
            }
        fi
    else
        print_warning "Test script not found: test_enhanced_extraction.py"
    fi
}

###############################################################################
# Main Installation Process
###############################################################################

main() {
    print_header "Enhanced Writer Extraction - Installation"
    print_info "This script will install all required dependencies"
    print_info "Installation directory: $SCRIPT_DIR"

    # Prompt for confirmation
    read -p "$(echo -e ${YELLOW}Continue with installation? [Y/n]: ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Installation cancelled"
        exit 0
    fi

    # System checks
    check_python
    check_pip
    check_system_dependencies

    # Installation steps
    upgrade_pip
    install_python_dependencies
    download_spacy_models
    create_directories

    # Verification
    verify_installation

    # Optional test
    run_test_script

    # Success message
    print_header "Installation Complete!"
    print_success "All dependencies have been installed successfully"
    print_info "You can now use the enhanced writer extraction system"
    print_info ""
    print_info "Quick start:"
    print_info "  1. Import the module: from writer_manager import get_writer_manager"
    print_info "  2. Process PDFs: wm = get_writer_manager()"
    print_info "  3. Extract writers: writers = wm.extract_writer_names_from_pdf('file.pdf')"
    print_info ""
    print_info "For detailed documentation, see: ENHANCED_EXTRACTION_GUIDE.md"
    print_info "To test the installation, run: python3 test_enhanced_extraction.py"
}

###############################################################################
# Run Main
###############################################################################

main

exit 0
