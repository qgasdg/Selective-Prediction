"""
Exploratory Data Analysis plots.
Run standalone:  python -m src.eda --data data/diabetic_data.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

from src.preprocess import load_and_clean


def run_eda(raw_df: pd.DataFrame, out_dir: str = 'results/eda'):
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Target distribution ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = raw_df['target'].value_counts().sort_index()
    ax.bar(['재입원 없음 (0)', '30일 내 재입원 (1)'],
           counts.values, color=['#4c9be8', '#e87c4c'])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f'{v:,}\n({v/len(raw_df):.1%})',
                ha='center', fontsize=10)
    ax.set_title('Target Distribution')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── 2. Age distribution by target ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    age_ct = (raw_df.groupby(['age', 'target'])
                    .size().unstack(fill_value=0)
                    .reindex(age_order))
    age_ct.plot(kind='bar', ax=ax, color=['#4c9be8', '#e87c4c'],
                width=0.7)
    ax.set_title('Age Group vs Target')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    ax.legend(['재입원 없음', '30일 내 재입원'])
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/age_vs_target.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── 3. Numeric feature distributions ────────────────────────────────────
    num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_medications',
                'number_inpatient', 'number_emergency', 'number_outpatient',
                'num_procedures', 'number_diagnoses']
    num_cols = [c for c in num_cols if c in raw_df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flatten(), num_cols):
        for t, color, label in [(0, '#4c9be8', '재입원 없음'),
                                  (1, '#e87c4c', '30일 내 재입원')]:
            ax.hist(raw_df.loc[raw_df['target'] == t, col],
                    bins=30, alpha=0.6, density=True,
                    color=color, label=label)
        ax.set_title(col)
        ax.legend(fontsize=7)
    plt.suptitle('Numeric Feature Distributions by Target', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/numeric_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── 4. Correlation heatmap (numeric columns only) ────────────────────────
    num_df = raw_df[num_cols + ['target']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num_df, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True)
    ax.set_title('Correlation Matrix (Numeric Features + Target)')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── 5. Readmission rate by number_inpatient (key feature) ─────────────
    if 'number_inpatient' in raw_df.columns:
        capped = raw_df['number_inpatient'].clip(upper=10)
        rate = raw_df.groupby(capped)['target'].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        rate.plot(kind='bar', ax=ax, color='#4c9be8')
        ax.set_title('30일 재입원율 by 이전 입원 횟수 (number_inpatient)')
        ax.set_xlabel('이전 입원 횟수 (≥10 capped)')
        ax.set_ylabel('재입원율')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.tight_layout()
        plt.savefig(f'{out_dir}/readmission_by_inpatient.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"EDA plots saved to {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/diabetic_data.csv')
    parser.add_argument('--out',  default='results/eda')
    args = parser.parse_args()

    df = load_and_clean(args.data)
    run_eda(df, args.out)
