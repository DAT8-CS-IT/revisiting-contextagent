#!/bin/bash

# =============================================================================
# Script: format_results.sh
# Purpose: Convert CSV predictions into terminal tables and/or LaTeX format
# Usage: ./format_results.sh [OPTIONS]
# =============================================================================

set -euo pipefail

# ============================================================================
# Configuration & Defaults
# ============================================================================

RESULTS_DIR="Reliable-Action-Recommendations-in-LLM-Based-Adaptive-Systems-/ContextAgent/results/cab/scores"
OUTPUT_FORMAT="terminal"  # terminal, latex, both
DATASET="cab"
METHODS="icl"             # icl, sft, or "both"
OUTPUT_FILE=""

# ============================================================================
# Functions
# ============================================================================

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -f, --format FORMAT     Output format: terminal, latex, both (default: terminal)
    -m, --methods METHODS   Methods to include: icl, sft, both (default: icl)
    -d, --dataset DATASET   Dataset name (default: cab)
    -o, --output FILE       Output LaTeX file (default: results_table.tex)
    -r, --results-dir DIR   Results directory (default: $RESULTS_DIR)

EXAMPLES:
    # Print ICL results to terminal
    $0 -f terminal -m icl

    # Generate LaTeX table for all methods
    $0 -f latex -m both -o comparison_table.tex

    # Print both terminal and LaTeX
    $0 -f both -m both -o results.tex
EOF
    exit 0
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -m|--methods)
                METHODS="$2"
                shift 2
                ;;
            -d|--dataset)
                DATASET="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -r|--results-dir)
                RESULTS_DIR="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
}

# Find CSV files matching criteria
find_csv_files() {
    local method=$1
    local dir=$2
    
    if [ "$method" == "icl" ]; then
        find "$dir" -name "metrics_icl_*.csv" -type f
    elif [ "$method" == "sft" ]; then
        find "$dir" -name "metrics_sft_*.csv" -type f
    else
        find "$dir" -name "metrics_*.csv" -type f
    fi | sort
}

# Trim whitespace from strings
trim() {
    local var="$1"
    var="${var#"${var%%[![:space:]]*}"}"   # remove leading whitespace
    var="${var%"${var##*[![:space:]]}"}"   # remove trailing whitespace
    printf '%s' "$var"
}

# Extract key metrics from CSV and format for display
format_csv_row() {
    local csv_file=$1
    local format=$2
    
    # Parse filename to get model info
    local filename=$(basename "$csv_file")
    local method=$(echo "$filename" | cut -d'_' -f2)
    local model=$(echo "$filename" | cut -d'_' -f3- | sed 's/_fs_10_custom//g' | sed 's/\.csv$//')
    
    # Extract Overall column (column 5, index 4) from each metric row and trim whitespace
    local proactive_idx=$(trim "$(grep "^Proactive_Index_Accuracy" "$csv_file" | cut -d',' -f5)")
    local miss_needed=$(trim "$(grep "^Miss_Needed" "$csv_file" | cut -d',' -f5)")
    local false_detection=$(trim "$(grep "^False_Detection" "$csv_file" | cut -d',' -f5)")
    local proactive_score=$(trim "$(grep "^Proactive_Score_Accuracy" "$csv_file" | cut -d',' -f5)")
    local rmse=$(trim "$(grep "^RMSE" "$csv_file" | cut -d',' -f5)")
    local precision=$(trim "$(grep "^Precision_Tool_Names" "$csv_file" | cut -d',' -f5)")
    local recall=$(trim "$(grep "^Recall_Tool_Names" "$csv_file" | cut -d',' -f5)")
    local f1=$(trim "$(grep "^F1_Tool_Names" "$csv_file" | cut -d',' -f5)")
    local arg_acc=$(trim "$(grep "^Argument_Accuracy" "$csv_file" | cut -d',' -f5)")
    
    # Validate that all metrics were retrieved
    if [ -z "$proactive_idx" ] || [ -z "$f1" ]; then
        echo "Warning: Could not parse $csv_file" >&2
        return 1
    fi
    
    if [ "$format" == "terminal" ]; then
        printf "%-30s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f\n" \
            "$model" \
            "$proactive_idx" \
            "$miss_needed" \
            "$false_detection" \
            "$proactive_score" \
            "$rmse" \
            "$precision" \
            "$recall" \
            "$f1" \
            "$arg_acc" 2>/dev/null
    elif [ "$format" == "latex" ]; then
        printf "%s & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f \\\\\n" \
            "$model" \
            "$proactive_idx" \
            "$miss_needed" \
            "$false_detection" \
            "$proactive_score" \
            "$rmse" \
            "$precision" \
            "$recall" \
            "$f1" \
            "$arg_acc" 2>/dev/null
    fi
}

# Print terminal table
print_terminal_table() {
    local method=$1
    
    echo ""
    echo "============================================================================"
    echo "Results for: $method (Dataset: $DATASET)"
    echo "============================================================================"
    echo ""
    printf "%-30s | %10s | %12s | %12s | %12s | %8s | %10s | %10s | %8s | %8s\n" \
        "Model" "Idx Acc" "Miss Needed" "False Detect" "Score Acc" "RMSE" "Precision" "Recall" "F1" "Arg Acc"
    echo "------------------------------+------------+--------------+--------------+--------------+----------+------------+------------+----------+----------"
    
    local files=$(find_csv_files "$method" "$RESULTS_DIR")
    if [ -z "$files" ]; then
        echo "No CSV files found for method: $method"
        return 1
    fi
    
    while IFS= read -r csv_file; do
        format_csv_row "$csv_file" "terminal" || true
    done <<< "$files"
    
    echo ""
}

# Generate LaTeX table
generate_latex_table() {
    local output_file=${1:-results_table.tex}
    
    cat > "$output_file" <<'LATEX_HEADER'
% Auto-generated LaTeX table
\documentclass[11pt]{article}
\usepackage{booktabs}
\usepackage{array}

\begin{document}

LATEX_HEADER

    # Determine which methods to include
    local methods_list=""
    if [ "$METHODS" == "both" ]; then
        methods_list=("icl" "sft")
    else
        methods_list=("$METHODS")
    fi
    
    for method in "${methods_list[@]}"; do
        cat >> "$output_file" <<EOF

\section*{${method^^} Results (Dataset: $DATASET)}

\begin{table}[htbp]
\centering
\small
\begin{tabular}{l | c c c c c c c c c}
\toprule
Model & Idx Acc & Miss Need & False Det & Score Acc & RMSE & Prec & Recall & F1 & Arg Acc \\
\midrule
EOF
        
        local files=$(find_csv_files "$method" "$RESULTS_DIR")
        if [ -z "$files" ]; then
            echo "No CSV files found for method: $method" >&2
            continue
        fi
        
        while IFS= read -r csv_file; do
            format_csv_row "$csv_file" "latex" || true
        done <<< "$files" >> "$output_file"
        
        cat >> "$output_file" <<'EOF'
\bottomrule
\end{tabular}
\end{table}

EOF
    done
    
    cat >> "$output_file" <<'LATEX_FOOTER'

\end{document}
LATEX_FOOTER

    echo "[Generated] LaTeX table saved to: $(pwd)/$output_file"
}

# ============================================================================
# Main
# ============================================================================

main() {
    parse_args "$@"
    
    # Validate directories
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "Error: Results directory not found: $RESULTS_DIR"
        exit 1
    fi
    
    # Validate output format
    if [[ ! "$OUTPUT_FORMAT" =~ ^(terminal|latex|both)$ ]]; then
        echo "Error: Invalid format. Use: terminal, latex, or both"
        exit 1
    fi
    
    # Validate methods
    if [[ ! "$METHODS" =~ ^(icl|sft|both)$ ]]; then
        echo "Error: Invalid methods. Use: icl, sft, or both"
        exit 1
    fi
    
    # Set default output file if needed
    if [ -z "$OUTPUT_FILE" ]; then
        OUTPUT_FILE="results_table_${DATASET}_$(date +%Y%m%d_%H%M%S).tex"
    fi
    
    # Determine which methods to process
    local methods_to_process=""
    if [ "$METHODS" == "both" ]; then
        methods_to_process=("icl" "sft")
    else
        methods_to_process=("$METHODS")
    fi
    
    # Generate output
    if [[ "$OUTPUT_FORMAT" =~ ^(terminal|both)$ ]]; then
        for method in "${methods_to_process[@]}"; do
            print_terminal_table "$method"
        done
    fi
    
    if [[ "$OUTPUT_FORMAT" =~ ^(latex|both)$ ]]; then
        generate_latex_table "$OUTPUT_FILE"
    fi
}

# ============================================================================
# Run
# ============================================================================

main "$@"
