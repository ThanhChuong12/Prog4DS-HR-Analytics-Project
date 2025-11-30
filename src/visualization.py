import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


# VISUALIZATION BASE CLASS
class HRVisualizer:
    def __init__(self):
        self._setup_style()

    def _setup_style(self) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("notebook", font_scale = 1.15)
        # Core brand colors
        self.color_primary = "#2E86AB"
        self.color_secondary = "#C73E1D"
        self.categorical_palette = sns.color_palette("tab20")


# NUMERIC DISTRIBUTION
class NumericPlots:
    def __init__(self, viz: HRVisualizer):
        self.viz = viz

    def plot_distribution(self, arr: np.ndarray, col: str) -> Tuple[Optional[plt.Figure], Optional[List]]:
        arr = arr.astype(float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            print(f"[WARN] Column {col} is empty after removing NaN.")
            return None, None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        sns.histplot(arr, kde = True, color = self.viz.color_primary, ax = axes[0])
        axes[0].axvline(arr.mean(), color = "red", linestyle = "--", label = f"Mean: {arr.mean():.2f}")
        axes[0].axvline(np.median(arr), color = "green", linestyle = "-", label = f"Median: {np.median(arr):.2f}")
        axes[0].set_title(f"Distribution of {col}")
        axes[0].set_xlabel(col)
        axes[0].legend()

        # Boxplot
        axes[1].boxplot(arr, vert = False, patch_artist = True,
                        boxprops = dict(facecolor = self.viz.color_primary, alpha = 0.6))
        axes[1].set_title(f"Outlier Detection: {col}")
        axes[1].set_xlabel(col)
        plt.tight_layout()
        return fig, axes

# ORDINAL DISTRIBUTION
class OrdinalPlots:
    def __init__(self, viz: HRVisualizer):
        self.viz = viz

    def plot_ordinal(self, arr: np.ndarray, col: str, order: List[str]) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        unique, counts = np.unique(arr, return_counts = True)
        count_dict = dict(zip(unique, counts))
        ordered_vals = [v for v in order if v in count_dict]
        freqs = [count_dict[v] for v in ordered_vals]
        if len(ordered_vals) == 0:
            print(f"[WARN] Column {col} has no valid ordinal values matching provided order.")
            return None, None
        fig, ax = plt.subplots(figsize = (10, 6))
        x_pos = np.arange(len(ordered_vals))
        ax.bar(x_pos, freqs, color=self.viz.color_secondary)
        ax.set_title(f"Ordinal Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ordered_vals, rotation = 45, ha = "right")
        offset = max(freqs) * 0.02
        for i, v in enumerate(freqs):
            ax.text(i, v + offset, str(v), ha = "center", fontsize = 10, fontweight = "bold")
        plt.tight_layout()
        return fig, ax
    
# CATEGORICAL DISTRIBUTION
class CategoricalPlots:
    def __init__(self, viz: HRVisualizer):
        self.viz = viz
        
    def _get_palette(self, n: int) -> List:
        return sns.color_palette("tab20", n)

    def plot_frequency(self, arr: np.ndarray, col: str, top_n: int = 10, custom_order: List[str] = None, ax: Optional[plt.Axes] = None)-> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        unique, counts = np.unique(arr, return_counts = True)
        count_dict = dict(zip(unique, counts))

        if custom_order:
            values = [v for v in custom_order if v in count_dict]
            freqs = [count_dict[v] for v in values]
            remaining = [v for v in unique if v not in values]
            if remaining:
                values.extend(remaining)
                freqs.extend([count_dict[v] for v in remaining])
            values = np.array(values)
            freqs = np.array(freqs)
        else:
            # Sort by quantity descending
            idx = np.argsort(counts)[::-1]
            values = unique[idx][:top_n]
            freqs = counts[idx][:top_n]
         
        created_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_ax = True
        else:
            fig = ax.figure
        x_pos = np.arange(len(values))
        ax.bar(x_pos, freqs, color=self.viz.color_primary)
        ax.set_title(f"Frequency of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(values, rotation=45, ha="right")
        offset = max(freqs) * 0.02
        for i, v in enumerate(freqs):
            ax.text(i, v + offset, str(v), ha = "center", fontsize = 10, fontweight = "bold")
        plt.tight_layout()
        return fig, ax

# FEATURE vs TARGET
class BivariatePlots:
    def __init__(self, viz: HRVisualizer):
        self.viz = viz

    def plot_feature_vs_target(self, feature: np.ndarray, target: np.ndarray, feature_col: str, ax: Optional[plt.Axes] = None) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        target = np.round(target.astype(float)).astype(int)
        unique_vals = np.unique(feature)

        if len(unique_vals) > 20:
            print(f"[SKIP] Feature {feature_col} has high cardinality ({len(unique_vals)}).")
            return None, None
        categories, rates = [], []
        for val in unique_vals:
            mask = (feature == val)
            if mask.sum() == 0:
                continue
            rate = target[mask].mean() * 100
            categories.append(str(val))
            rates.append(rate)
        idx = np.argsort(rates)[::-1]
        cat_sorted = np.array(categories)[idx]
        rate_sorted = np.array(rates)[idx]
        created_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize = (12, 6))
            created_ax = True
        else:
            fig = ax.figure
        x_pos = np.arange(len(cat_sorted))
        ax.bar(x_pos, rate_sorted, color=self.viz.color_secondary, alpha = 0.85)
        ax.axhline(y=target.mean() * 100, color = "black", linestyle="--", label = "Global Avg")
        ax.set_title(f"Job Change Probability by {feature_col}")
        ax.set_ylabel("Probability (%)")
        ax.set_xlabel(feature_col)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cat_sorted, rotation=45, ha = "right")
        ax.legend()
        for i, rate in enumerate(rate_sorted):
            ax.text(i, rate + 0.5, f"{rate:.1f}%", ha = "center", fontsize = 10)
        if created_ax:
            plt.tight_layout()
        return fig, ax
    
    def plot_brain_drain(self, results_dict: dict, train_threshold: float, ax: Optional[plt.Axes] = None):
        """
        Plot the CDI x Training Hours interaction graph to test the Brain Drain hypothesis.
        """
        cdi_groups = results_dict['cdi_group']
        churn_rates = results_dict['churn_rate']
        train_groups = results_dict['train_group']
        
        created_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
            created_ax = True
        else:
            fig = ax.figure

        palette = [self.viz.color_primary, self.viz.color_secondary] 
        
        sns.barplot(
            x=cdi_groups,
            y=churn_rates,
            hue=train_groups,
            palette=palette,
            edgecolor='black',
            alpha=0.9,
            ax=ax
        )
        
        # Styling
        ax.set_title(f"Brain Drain Hypothesis: Training Impact by City Tier\n(High Training > {train_threshold:.0f} hrs)", fontsize=14)
        ax.set_ylabel("Probability of Job Change (%)", fontsize=12)
        ax.set_xlabel("City Development Index (CDI)", fontsize=12)
        plt.ylim(0, max(churn_rates) * 1.15)
        ax.legend(title="Training Intensity")

        # Annotation (Highlight the difference in the Low CDI group)
        if len(churn_rates) >= 2:
            low_cdi_diff = churn_rates[1] - churn_rates[0] # High - Low
            if low_cdi_diff > 0:
                ax.annotate(
                    f"+{low_cdi_diff:.1f}% Risk\n(Brain Drain)",
                    xy=(-0.2, churn_rates[1]),
                    xytext=(-0.2, churn_rates[1] + 5),
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    color='red', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.9)
                )

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)
        if created_ax:
            plt.tight_layout()
        return fig, ax
    
class HREDA:
    def __init__(self):
        self.viz = HRVisualizer()
        self.numeric = NumericPlots(self.viz)
        self.ordinal = OrdinalPlots(self.viz)
        self.categorical = CategoricalPlots(self.viz)
        self.bivariate = BivariatePlots(self.viz)

    def numeric_section(self, data: Dict[str, np.ndarray], numeric_cols: List[str]):
        for col in numeric_cols:
            if col not in data:
                print(f"[WARN] Column '{col}' not found in data dictionary.")
                continue
            fig, ax = self.numeric.plot_distribution(data[col], col)
            if fig: plt.show()
                
    def ordinal_section(self, data: Dict[str, np.ndarray], ordinal_cols: Dict[str, List[str]]):
        for col, order in ordinal_cols.items():
            if col not in data:
                print(f"[WARN] Column '{col}' not found in data.")
                continue

            fig, ax = self.ordinal.plot_ordinal(data[col], col, order)
            if fig: plt.show()

    def categorical_section(self, data: Dict[str, np.ndarray], categorical_cols: List[str]):
        for col in categorical_cols:
            if col not in data:
                print(f"[WARN] Column '{col}' not found in data dictionary.") # <--- Thêm log để debug
                continue
            fig, ax = self.categorical.plot_frequency(data[col], col)
            if fig: plt.show()

    def bivariate_section(self, data: Dict[str, np.ndarray], categorical_cols: List[str], target_col: str = "target"):
        if target_col not in data:
            print("[WARN] Target column not found. Bivariate analysis skipped.")
            return
        target = data[target_col]
        for col in categorical_cols:
            fig, ax = self.bivariate.plot_feature_vs_target(data[col], target, col)
            if fig: plt.show()

def plot_churn_by_experience_group(experience_labels, target_arr):
    """
    Plot churn rate by experience group (Bar Chart).
    """

    groups = [
        "0-1 Year (Junior)",
        "1-5 Years (Mid)",
        "5-10 Years (Senior)",
        "10-20 Years (Expert)",
        ">20 Years (Veteran)",
    ]

    rates = []
    for grp in groups:
        mask = experience_labels == grp
        rate = target_arr[mask].mean() * 100 if mask.sum() else np.nan
        rates.append(rate)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(groups, rates, edgecolor="black", alpha=0.85)

    plt.title("Churn Rate by Experience Level")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, rate in zip(bars, rates):
        if not np.isnan(rate):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.8,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.show()


def plot_heatmap_experience_vs_size(matrix, row_labels, col_labels):
    """
    Heatmap showing churn interaction between company size and experience.
    """

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.5,
        cbar_kws={"label": "Churn Rate (%)"},
    )

    plt.title("Risk Heatmap: Churn Rate by Experience & Company Size")
    plt.xlabel("Experience Level")
    plt.ylabel("Company Size")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.show()

