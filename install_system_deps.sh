#!/bin/bash

###############################################################################
# System Dependencies Installation Script
# Installs system-level packages needed for Python dependencies
# Requires sudo privileges
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
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

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}System Dependencies Installation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_warning "This script requires sudo privileges"
    print_info "You may be prompted for your password"
    echo ""
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    print_info "Detected OS: $OS"
else
    print_error "Cannot detect operating system"
    exit 1
fi

# Install based on OS
if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    print_info "Installing dependencies for Debian/Ubuntu..."

    sudo apt-get update
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        libffi-dev \
        libssl-dev \
        libjpeg-dev \
        zlib1g-dev

    print_success "System dependencies installed"

elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Fedora"* ]]; then
    print_info "Installing dependencies for CentOS/RHEL/Fedora..."

    sudo yum update -y
    sudo yum install -y \
        python3 \
        python3-pip \
        python3-devel \
        gcc \
        gcc-c++ \
        make \
        git \
        libffi-devel \
        openssl-devel \
        libjpeg-devel \
        zlib-devel

    print_success "System dependencies installed"

elif [[ "$OS" == *"Arch"* ]]; then
    print_info "Installing dependencies for Arch Linux..."

    sudo pacman -Syu --noconfirm \
        python \
        python-pip \
        base-devel \
        git \
        libffi \
        openssl \
        libjpeg-turbo \
        zlib

    print_success "System dependencies installed"

elif [[ "$OS" == *"Alpine"* ]]; then
    print_info "Installing dependencies for Alpine Linux..."

    sudo apk update
    sudo apk add \
        python3 \
        python3-dev \
        py3-pip \
        build-base \
        git \
        libffi-dev \
        openssl-dev \
        jpeg-dev \
        zlib-dev

    print_success "System dependencies installed"

else
    print_warning "Unsupported OS: $OS"
    print_info "Please install the following manually:"
    echo "  - Python 3.8+"
    echo "  - pip3"
    echo "  - build-essential / gcc"
    echo "  - git"
    echo "  - Development headers for: libffi, openssl, jpeg, zlib"
    exit 1
fi

echo ""
print_success "System dependencies installation complete!"
print_info "You can now run: ./install_dependencies.sh"

exit 0
