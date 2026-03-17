# CS170 1-NN Feature Selection Problem 

## Introduction
In this project, I worked on programming forward selection and backward elimination with the 1-Nearest-Neighbor algorithm to classify instances into two or more categories. Feel free to run this repository and experiment with the different data files.

## Cloning the Repository

### 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/<your-user-name>/NN-feature-selection.git
```

### 2. Navigate into the Project Directory

```bash
cd NN-feature-selection
```

### 3. Run the Program

There are **no dependencies required**. You can run the program directly with Python:

```bash
python src/main.py
```

## Command Line Options

The program supports the following CLI flags (in any combination):

### `-s`

Runs the **simple sanity check dataset**.

```bash
python src/main.py -s
```

For both forwards and backwards, you should get `[9, 12]` and an accuracy of ~95 as your output.

### `-d`

Enables **verbose output**.

```bash
python src/main.py -d
```
