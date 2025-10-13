# MLP Neural Network Verification with Reluplex

This directory contains a custom verification script for Multi-Layer Perceptron (MLP) neural networks using the Reluplex SMT solver.

## Overview

The MLP verification system allows you to verify properties of your trained MLP models, including:
- **Classification verification**: Verify that a specific class has the maximum output
- **Output bounds verification**: Verify that outputs satisfy certain bounds
- **Local robustness**: Verify robustness around specific input points
- **General satisfiability**: Check if the network can produce certain outputs

## Files Created

### Core Verification Code
- `check_properties/mlp_verification/main.cpp` - Main verification logic
- `check_properties/mlp_verification/Makefile` - Build configuration

### Scripts
- `scripts/run_mlp_verification.sh` - Interactive verification script
- `test_mlp_verification.sh` - Setup and compilation test script

### Documentation
- `MLP_VERIFICATION_README.md` - This file

## Setup Instructions

### 1. Prerequisites
- Reluplex must be compiled (run `make` in the `reluplex` directory)
- GLPK must be installed and compiled
- Your MLP model must be in `.nnet` format in the `nnet` directory

### 2. Compilation
```bash
# From the ReluplexCav2017 root directory
cd check_properties
make
```

This will compile all properties including the new MLP verification.

### 3. Test Setup
```bash
# From the ReluplexCav2017 root directory
./test_mlp_verification.sh
```

## Usage

### Interactive Mode (Recommended)
```bash
./scripts/run_mlp_verification.sh
```

This will present you with options for different types of verification.

### Direct Command Line Usage
```bash
./check_properties/bin/mlp_verification.elf <network> [output_file] [input_bounds] [property_type] [property_params]
```

#### Parameters:
- `network`: Path to the .nnet file (e.g., `./nnet/model_mlp.nnet`)
- `output_file`: Optional output file for results
- `input_bounds`: Input bounds specification (e.g., `"all:[-1,1]"`)
- `property_type`: Type of property to verify
- `property_params`: Parameters for the property

## Property Types

### 1. Classification Verification
Verifies that a specific class has the maximum output.

```bash
./check_properties/bin/mlp_verification.elf ./nnet/model_mlp.nnet results.txt "all:[-1,1]" classification 0
```

This verifies that class 0 has the maximum output for inputs in the range [-1,1].

### 2. Output Bounds Verification
Verifies that specific outputs satisfy certain bounds.

```bash
./check_properties/bin/mlp_verification.elf ./nnet/model_mlp.nnet results.txt "all:[-1,1]" output_bounds "0:>0.5"
```

This verifies that output 0 is greater than 0.5.

### 3. General Satisfiability
Checks if the network can produce certain outputs without specific constraints.

```bash
./check_properties/bin/mlp_verification.elf ./nnet/model_mlp.nnet results.txt "all:[-1,1]"
```

## Input Bounds Specification

Input bounds can be specified in several ways:

- `"all:[-1,1]"` - All inputs bounded between -1 and 1
- `"0:[-0.5,0.5]"` - Only input 0 bounded between -0.5 and 0.5
- `"0:[-0.5,0.5],1:[-1,1]"` - Multiple inputs with different bounds

## Output Interpretation

### SAT Result
- **SAT**: The property is satisfiable (counterexample found)
- **UNSAT**: The property is unsatisfiable (property holds)
- **TIMEOUT**: Verification timed out
- **ERROR**: An error occurred during verification

### Counterexamples
When SAT is returned, the script will show:
- Input values that satisfy the property
- Corresponding output values

## Example Workflows

### 1. Verify MNIST-1D Classification
```bash
# Verify that class 0 is correctly classified for inputs in [-2,2]
./check_properties/bin/mlp_verification.elf ./nnet/model_mlp.nnet logs/class0_verification.txt "all:[-2,2]" classification 0
```

### 2. Check Output Bounds
```bash
# Verify that output 0 is always positive for inputs in [-1,1]
./check_properties/bin/mlp_verification.elf ./nnet/model_mlp.nnet logs/output_bounds.txt "all:[-1,1]" output_bounds "0:>0"
```

### 3. General Network Analysis
```bash
# Check general satisfiability
./check_properties/bin/mlp_verification.elf ./nnet/model_mlp.nnet logs/general_check.txt "all:[-1,1]"
```

## Network Format Requirements

Your MLP model must be in the `.nnet` format with:
- No input normalization (indicated by `-` in the .nnet file)
- ReLU activations between layers
- Proper layer size specification

## Troubleshooting

### Compilation Issues
- Ensure GLPK is properly installed and compiled
- Check that all dependencies are available
- Run `make clean` and `make` again

### Runtime Issues
- Verify the .nnet file format is correct
- Check that input bounds are reasonable
- Ensure sufficient memory for large networks

### Performance Issues
- Use smaller input bounds for faster verification
- Consider timeout settings for long-running verifications
- Monitor memory usage for large networks

## Integration with Existing Reluplex

The MLP verification script integrates seamlessly with the existing Reluplex infrastructure:
- Uses the same Reluplex core engine
- Follows the same property verification patterns
- Compatible with existing logging and output systems
- Can be extended with additional property types

## Extending the Verification

To add new property types:

1. Add the property logic in `main.cpp`
2. Update the command-line argument parsing
3. Add corresponding test cases
4. Update this documentation

## Results and Logging

Results are saved to:
- Summary files: `logs/mlp_verification_summary.txt`
- Statistics files: `logs/mlp_verification_stats_*.txt`
- Custom output files (if specified)

The logging format is compatible with existing Reluplex analysis tools.
