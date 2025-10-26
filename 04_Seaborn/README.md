# Seaborn Revision Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Styling](#styling)
3. [Distribution Plots](#distribution-plots)
4. [Categorical Plots](#categorical-plots)
5. [Relational Plots](#relational-plots)
6. [Matrix Plots](#matrix-plots)
7. [Regression Plots](#regression-plots)
8. [Multi-plot Grids](#multi-plot-grids)

## Introduction
Seaborn is a Python data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.

## Styling
- `set_style()`: Set the aesthetic style
- `set_palette()`: Set the color palette
- `set_context()`: Set the plotting context
- Available styles: darkgrid, whitegrid, dark, white, ticks

## Distribution Plots
- **Histogram**: `histplot()`
- **KDE Plot**: `kdeplot()`
- **Distribution Plot**: `displot()`
- **Rug Plot**: `rugplot()`
- **Box Plot**: `boxplot()`
- **Violin Plot**: `violinplot()`

## Categorical Plots
- **Strip Plot**: `stripplot()`
- **Swarm Plot**: `swarmplot()`
- **Box Plot**: `boxplot()`
- **Violin Plot**: `violinplot()`
- **Bar Plot**: `barplot()`
- **Count Plot**: `countplot()`
- **Point Plot**: `pointplot()`

## Relational Plots
- **Scatter Plot**: `scatterplot()`
- **Line Plot**: `lineplot()`
- **Relational Plot**: `relplot()`

## Matrix Plots
- **Heatmap**: `heatmap()`
- **Cluster Map**: `clustermap()`
- Correlation matrices

## Regression Plots
- **Regression Plot**: `regplot()`
- **LM Plot**: `lmplot()`

## Multi-plot Grids
- **FacetGrid**: Create grids of plots
- **PairGrid**: Pairwise relationships
- **Pair Plot**: `pairplot()`
- **Joint Plot**: `jointplot()`
