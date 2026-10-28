# ML-Based Predictive Parser for Adaptive Syntax Analysis

## Overview

This project implements a machine-learning-based predictive parser that improves production-rule selection when traditional LL(1) parsers encounter grammar conflicts.

The system combines grammar analysis, synthetic dataset generation, supervised machine learning, guarded parsing, prediction caching, and syntax-error recovery.

## Objectives

- Implement traditional LL(1) and SLR(1) parsers as baselines.
- Generate labelled parsing datasets from grammar derivations.
- Train and compare machine learning models for production-rule prediction.
- Improve parsing decisions using additional lookahead information.
- Evaluate syntax-error detection, error localisation, recovery, and adaptation to grammar changes.

## Datasets

Two synthetic datasets were generated using different grammars.

| Feature | G1 | G2 |
|---|---|---|
| Grammar type | Conflict-free LL(1) | Grammar with LL(1) conflicts |
| Production rules | 25 | 31 |
| LL(1) conflicts | 0 | 2 |
| Generated programs | 2,500 | 2,500 |
| Training programs | 2,000 | 2,000 |
| Testing programs | 500 | 500 |

**Why synthetic data?**

Synthetic data allows us to generate programs with known correct parsing decisions. An independent oracle is used to verify the ground-truth labels, enabling controlled and repeatable evaluation.

Each dataset contains parsing features and the corresponding correct grammar production as its label.

## Machine Learning Algorithms

The following supervised learning algorithms were implemented and compared:

- **Decision Tree:** Predicts production rules using parsing-context features.
- **Random Forest:** Combines multiple decision trees to make predictions.
- **Multi-Layer Perceptron (MLP):** A neural-network-based classifier for production-rule prediction.

### Input Features

The models use features such as:

- Current non-terminal
- First, second, and third lookahead tokens
- Previously consumed token
- Parent non-terminal
- Brace nesting depth
- Parenthesis nesting depth

Feature sets F1, F2, and F3 were compared to study the effect of additional context.

## System Architecture

1. **Grammar analysis:** Computes FIRST and FOLLOW sets and constructs LL(1) parsing tables.
2. **Lexical analysis:** Converts source input into tokens.
3. **Dataset generation:** Generates programs and records correct parsing decisions.
4. **Model training:** Trains the classifiers using labelled parsing data.
5. **ML-based parsing:** Predicts suitable production rules using the learned model.
6. **Guarded parsing:** Restricts predictions to grammar-permitted production rules.
7. **Error handling:** Detects syntax errors and supports panic-mode recovery.
8. **Evaluation:** Measures model performance, adaptation, error localisation, and runtime.

## Evaluation Metrics

The project evaluates:

- **Decision accuracy:** Percentage of correct individual parsing decisions.
- **Program accuracy:** Percentage of programs parsed with every decision correct.
- **Conflict accuracy:** Accuracy on ambiguous LL(1) decision points.
- **Oracle agreement:** Agreement with the independent reference recognizer.
- **False accepts and false rejects:** Measures incorrect acceptance or rejection of programs.
- **Exact error position:** Measures whether a syntax error is reported at the expected token position.
- **Runtime:** Measures parsing time and computational efficiency.

Precision, recall, and F1-score are not reported as separate experimental results in the current evaluation.

## Results

The experiments show that additional lookahead features improve production-rule prediction on G2.

| Method on G2 | Decision accuracy | Program accuracy |
|---|---:|---:|
| Classical LL(1) table | 89.5% | 45.4% |
| Decision Tree, F1 | 95.1% | 49.2% |
| Decision Tree, F2 | Strong performance | Strong performance |
| Decision Tree, F3 | Strong performance | Strong performance |

The experiments also investigate grammar adaptation, syntax-error localisation, error recovery, and prediction caching.

**Note:** Results were obtained using small grammars and synthetic, noise-free data. They do not guarantee equivalent performance on real-world programming languages.

## Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- LL(1) and SLR(1) parsing techniques

## Project Structure

```text
ML-Based-Predictive-Parser/
├── code/
│   ├── grammar.py
│   ├── lexer.py
│   ├── ll1_parser.py
│   ├── slr_parser.py
│   ├── generator.py
│   ├── train_part2.py
│   ├── ml_parser.py
│   ├── part3_exp1.py
│   ├── part3_exp2.py
│   ├── run_all_demo_FIXED.py
│   ├── grammars/
│   ├── data/
│   ├── models/
│   └── figures/
├── report/
│   ├── main.tex
│   └── figures/
├── demo_results.txt
└── README.md
```

## Installation and Execution

### 1. Install Python

Install a compatible Python version and ensure Python is available from your terminal.

### 2. Install dependencies

```bash
python -m pip install scikit-learn numpy pandas matplotlib
```

### 3. Run the project

From the repository root directory, run:

```bash
python code/run_all_demo_FIXED.py
```

Alternatively, run individual experiment scripts:

```bash
python code/train_part2.py
python code/part3_exp1.py
python code/part3_exp2.py
```

Run the scripts from the repository root and check their output for any required working-directory or dependency adjustments.

## Limitations and Future Work

- Evaluate the parser on real-world source code.
- Extend the system to larger and more complex grammars.
- Investigate additional contextual features and parsing strategies.
- Improve runtime performance and model efficiency.
- Explore precision, recall, and F1-score as additional evaluation metrics.

## Conclusion

This project demonstrates how supervised machine learning can assist predictive parsing by using additional context to resolve grammar conflicts. It also explores guarded parsing, error recovery, and adaptation to grammar changes.

The findings provide a foundation for further research into adaptive syntax analysis.

## Repository

Source code, experiments, datasets, and report files are available in this repository.

