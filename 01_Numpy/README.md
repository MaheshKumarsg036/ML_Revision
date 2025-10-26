# Numpy Revision Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Array Creation](#array-creation)
3. [Array Operations](#array-operations)
4. [Indexing and Slicing](#indexing-and-slicing)
5. [Array Manipulation](#array-manipulation)
6. [Mathematical Functions](#mathematical-functions)
7. [Broadcasting](#broadcasting)
8. [Linear Algebra](#linear-algebra)
9. [Random Module](#random-module)

## Introduction
NumPy is the fundamental package for scientific computing in Python. It provides:
- Powerful N-dimensional array object
- Sophisticated broadcasting functions
- Tools for integrating C/C++ and Fortran code
- Useful linear algebra, Fourier transform, and random number capabilities

## Array Creation
See `numpy_basics.py` for examples of:
- Creating arrays from lists
- Using `np.zeros()`, `np.ones()`, `np.empty()`
- Using `np.arange()`, `np.linspace()`
- Identity matrices with `np.eye()`
- Random arrays

## Array Operations
- Element-wise operations
- Arithmetic operations
- Comparison operations
- Logical operations

## Indexing and Slicing
- Basic indexing
- Boolean indexing
- Fancy indexing

## Array Manipulation
- Reshaping: `reshape()`, `flatten()`, `ravel()`
- Stacking: `hstack()`, `vstack()`, `concatenate()`
- Splitting: `hsplit()`, `vsplit()`, `split()`

## Mathematical Functions
- Trigonometric functions
- Exponential and logarithmic functions
- Statistical functions (mean, median, std, var)
- Aggregation functions (sum, min, max)

## Broadcasting
Broadcasting allows NumPy to work with arrays of different shapes during arithmetic operations.

## Linear Algebra
- Matrix multiplication
- Determinants
- Eigenvalues and eigenvectors
- Matrix inverse
- Solving linear equations

## Random Module
- Random number generation
- Random sampling
- Random distributions
