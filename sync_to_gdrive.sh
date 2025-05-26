#!/bin/bash

# Sync Jupyter Notebooks to Google Drive for Google Colab
# This script wraps the Python sync script for easier usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/sync_to_gdrive.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to check if Python script exists
check_python_script() {
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        print_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if required Python packages are installed
    python3 -c "import jupytext, google.auth, googleapiclient" 2>/dev/null
    if [[ $? -ne 0 ]]; then
        print_warning "Required Python packages not found"
        print_info "Installing required packages..."
        pip3 install jupytext google-api-python-client google-auth-httplib2 google-auth-oauthlib
        if [[ $? -ne 0 ]]; then
            print_error "Failed to install required packages"
            exit 1
        fi
        print_success "Dependencies installed successfully"
    else
        print_success "All dependencies are installed"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Sync Jupyter Notebooks to Google Drive for Google Colab

Usage:
    $0 setup                          # Setup Google Drive credentials and folder
    $0 setup --force-refresh          # Force refresh of credentials
    $0 config                         # Configure Google Drive folder
    $0 status                         # Show current configuration
    $0 reset                          # Reset stored credentials
    $0 sync --all                     # Sync all notebooks
    $0 sync --folder <path>           # Sync specific folder
    $0 sync --file <path>             # Sync specific file
    $0 sync --all --dry-run           # Show what would be synced
    $0 sync --all --force             # Force reconversion of all files
    
Examples:
    $0 setup
    $0 setup --force-refresh
    $0 status
    $0 config
    $0 reset
    $0 sync --all
    $0 sync --folder notebooks/01_vector
    $0 sync --file notebooks/01_vector/01_introduccion_datos_vectoriales.py
    $0 sync --all --dry-run
    
Options:
    --dry-run     Show what would be synced without actually doing it
    --force       Force reconversion of .ipynb files even if they're up to date
    --help        Show this help message

Commands:
    setup         Initial setup of credentials and folder configuration
    config        Reconfigure the target Google Drive folder
    status        Show current configuration and connection status
    reset         Reset/remove stored credentials
    sync          Synchronize notebooks to Google Drive

Options:
    --force-refresh   Force refresh of credentials during setup
EOF
}

# Main script logic
main() {
    # Check if no arguments provided
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 0
    fi
    
    # Handle help flag
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        show_usage
        exit 0
    fi
    
    check_python_script
    check_dependencies
    
    # Pass all arguments to the Python script
    print_info "Running sync script..."
    python3 "$PYTHON_SCRIPT" "$@"
    
    if [[ $? -eq 0 ]]; then
        print_success "Script completed successfully"
    else
        print_error "Script failed"
        exit 1
    fi
}

# Run main function
main "$@" 