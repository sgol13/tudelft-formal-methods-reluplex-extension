#!/bin/bash

# MLP Verification Script for Reluplex
# This script provides convenient ways to verify properties of MLP neural networks

TIMEOUT=1h
NETWORK_PATH="./nnet/model_mlp.nnet"
OUTPUT_DIR="logs"
RESULTS_FILE="${OUTPUT_DIR}/mlp_verification_summary.txt"

# Create logs directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

echo "MLP Neural Network Verification with Reluplex"
echo "=============================================="
echo ""

# Check if the network file exists
if [ ! -f "${NETWORK_PATH}" ]; then
    echo "Error: Network file not found at ${NETWORK_PATH}"
    echo "Please ensure your model_mlp.nnet file is in the nnet directory"
    exit 1
fi

# Check if the verification binary exists
if [ ! -f "./check_properties/bin/mlp_verification.elf" ]; then
    echo "Error: MLP verification binary not found"
    echo "Please run 'make' in the check_properties directory first"
    exit 1
fi

echo "Network: ${NETWORK_PATH}"
echo "Timeout: ${TIMEOUT}"
echo ""

# Function to run verification with different properties
run_verification() {
    local property_type="$1"
    local property_params="$2"
    local input_bounds="$3"
    local description="$4"
    
    echo "Running verification: ${description}"
    echo "Property: ${property_type} ${property_params}"
    echo "Input bounds: ${input_bounds}"
    echo "----------------------------------------"
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local stats_file="${OUTPUT_DIR}/mlp_verification_stats_${property_type}_${timestamp}.txt"
    
    timeout --foreground --signal=SIGQUIT ${TIMEOUT} \
        ./check_properties/bin/mlp_verification.elf \
        "${NETWORK_PATH}" \
        "${RESULTS_FILE}" \
        "${input_bounds}" \
        "${property_type}" \
        "${property_params}" \
        2>&1 | tee "${stats_file}"
    
    echo ""
    echo "Results saved to: ${stats_file}"
    echo "Summary saved to: ${RESULTS_FILE}"
    echo ""
}

# Default input bounds for MNIST-1D style data
DEFAULT_BOUNDS="all:[-2,2]"

echo "Available verification options:"
echo "1. General satisfiability check"
echo "2. Classification verification (class 0 is maximum)"
echo "3. Classification verification (class 5 is maximum)"
echo "4. Output bounds verification"
echo "5. Custom verification"
echo ""

read -p "Select option (1-5) or press Enter for option 1: " choice

case ${choice:-1} in
    1)
        run_verification "" "" "${DEFAULT_BOUNDS}" "General satisfiability check"
        ;;
    2)
        run_verification "classification" "0" "${DEFAULT_BOUNDS}" "Verify class 0 has maximum output"
        ;;
    3)
        run_verification "classification" "5" "${DEFAULT_BOUNDS}" "Verify class 5 has maximum output"
        ;;
    4)
        run_verification "output_bounds" "0:>0.5" "${DEFAULT_BOUNDS}" "Verify output 0 > 0.5"
        ;;
    5)
        echo "Custom verification options:"
        echo ""
        read -p "Input bounds (e.g., 'all:[-1,1]' or '0:[-0.5,0.5]'): " custom_bounds
        read -p "Property type (classification/robustness/output_bounds): " custom_property
        read -p "Property parameters (e.g., '0' for class 0, '0:>0.5' for output bounds): " custom_params
        
        run_verification "${custom_property}" "${custom_params}" "${custom_bounds:-${DEFAULT_BOUNDS}}" "Custom verification"
        ;;
    *)
        echo "Invalid option. Running default verification..."
        run_verification "" "" "${DEFAULT_BOUNDS}" "General satisfiability check"
        ;;
esac

echo "Verification completed!"
echo "Check the logs directory for detailed results."
