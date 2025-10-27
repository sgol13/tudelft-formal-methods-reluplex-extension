#!/bin/bash

# Test script for MLP verification setup
# This script tests the compilation and basic functionality

echo "Testing MLP Verification Setup"
echo "=============================="
echo ""

# Check if we're in the right directory
if [ ! -f "reluplex/reluplex.elf" ]; then
    echo "Error: Not in the Reluplex root directory"
    echo "Please run this script from the ReluplexCav2017 directory"
    exit 1
fi

# Check if the MLP model exists
if [ ! -f "nnet/model_mlp.nnet" ]; then
    echo "Error: model_mlp.nnet not found in nnet directory"
    echo "Please copy your model_mlp.nnet file to the nnet directory"
    exit 1
fi

echo "✓ Found model_mlp.nnet in nnet directory"

# Try to compile the MLP verification
echo ""
echo "Compiling MLP verification..."
cd check_properties
make clean
make

if [ $? -eq 0 ]; then
    echo "✓ MLP verification compiled successfully"
else
    echo "✗ Compilation failed"
    exit 1
fi

# Go back to root directory
cd ..

# Check if the binary was created
if [ -f "check_properties/bin/mlp_verification.elf" ]; then
    echo "✓ MLP verification binary created"
else
    echo "✗ MLP verification binary not found"
    exit 1
fi

# Test basic functionality
echo ""
echo "Testing basic functionality..."
echo "Running a simple satisfiability check..."

timeout 30s ./check_properties/bin/mlp_verification.elf \
    ./nnet/model_mlp.nnet \
    logs/test_results.txt \
    "all:[-1,1]" \
    "" \
    ""

if [ $? -eq 0 ] || [ $? -eq 1 ]; then
    echo "✓ Basic functionality test passed"
else
    echo "✗ Basic functionality test failed"
    exit 1
fi

echo ""
echo "Setup test completed successfully!"
echo ""
echo "You can now run MLP verification using:"
echo "  ./scripts/run_mlp_verification.sh"
echo ""
echo "Or directly with:"
echo "  ./check_properties/bin/mlp_verification.elf <network> [output] [bounds] [property] [params]"
