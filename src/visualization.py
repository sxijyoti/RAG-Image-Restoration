"""
Phase 4: Visualization Module for DA-CLIP Retrieval

Visualize retrieval results for debugging and validation.
"""

import numpy as np
from typing import List, Union, Optional, Tuple
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class RetrievalVisualizer:
    """Visualize query patch and retrieved results."""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 6), debug: bool = True):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size (width, height)
            debug: Print debug information
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib not available. Install with: pip install matplotlib"
            )
        
        self.figsize = figsize
        self.debug = debug
    
    def visualize_retrieval(
        self,
        query_patch: np.ndarray,
        retrieved_patches: List[np.ndarray],
        distances: Optional[np.ndarray] = None,
        titles: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Visualize query patch and top-k retrieved patches.
        
        Args:
            query_patch: Query patch array (H, W, 3)
            retrieved_patches: List of retrieved patch arrays
            distances: Optional distances/similarities for each retrieved patch
            titles: Optional titles for each retrieved patch
            save_path: Path to save figure
            show: Whether to display the figure
            
        Returns:
            matplotlib Figure object
        """
        if not isinstance(retrieved_patches, list):
            retrieved_patches = [retrieved_patches]
        
        k = len(retrieved_patches)
        
        # Create figure
        fig = plt.figure(figsize=self.figsize, dpi=100)
        gs = GridSpec(2, k + 1, figure=fig, hspace=0.3, wspace=0.3)
        
        # Query patch
        ax_query = fig.add_subplot(gs[0, 0])
        self._plot_patch(ax_query, query_patch, "Query Patch", is_query=True)
        
        # Retrieved patches (top row)
        for i, patch in enumerate(retrieved_patches):
            ax = fig.add_subplot(gs[0, i + 1])
            
            title = f"Result {i+1}"
            if titles:
                title = titles[i]
            elif distances is not None:
                title = f"Result {i+1}\n(dist: {distances[i]:.4f})"
            
            self._plot_patch(ax, patch, title)
        
        # Add difference maps (bottom row)
        ax_query_diff = fig.add_subplot(gs[1, 0])
        ax_query_diff.text(
            0.5, 0.5, "Diff Maps\n(below)",
            ha='center', va='center',
            fontsize=10,
            transform=ax_query_diff.transAxes
        )
        ax_query_diff.set_xticks([])
        ax_query_diff.set_yticks([])
        ax_query_diff.spines['left'].set_visible(False)
        ax_query_diff.spines['bottom'].set_visible(False)
        ax_query_diff.spines['right'].set_visible(False)
        ax_query_diff.spines['top'].set_visible(False)
        
        # Difference maps
        for i, patch in enumerate(retrieved_patches):
            ax = fig.add_subplot(gs[1, i + 1])
            diff = self._compute_difference(query_patch, patch)
            
            im = ax.imshow(diff, cmap='hot')
            ax.set_title(f"Diff {i+1}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('L2 dist', fontsize=8)
        
        fig.suptitle(
            f"DA-CLIP Retrieval Results (Top-{k})",
            fontsize=14,
            fontweight='bold'
        )
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            if self.debug:
                print(f" Figure saved: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig
    
    def _plot_patch(
        self,
        ax,
        patch: np.ndarray,
        title: str,
        is_query: bool = False
    ):
        """Plot a single patch."""
        # Ensure uint8
        if patch.dtype == np.float32 or patch.dtype == np.float64:
            if patch.max() <= 1.0:
                patch = (patch * 255).astype(np.uint8)
            else:
                patch = np.clip(patch, 0, 255).astype(np.uint8)
        
        ax.imshow(patch)
        ax.set_title(title, fontsize=10, fontweight='bold' if is_query else 'normal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border for query patch
        if is_query:
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
    
    def _compute_difference(
        self,
        patch1: np.ndarray,
        patch2: np.ndarray
    ) -> np.ndarray:
        """Compute L2 difference map between two patches."""
        # Ensure float32
        p1 = patch1.astype(np.float32) if patch1.dtype != np.float32 else patch1
        p2 = patch2.astype(np.float32) if patch2.dtype != np.float32 else patch2
        
        # Normalize to [0, 1] if needed
        if p1.max() > 1.0:
            p1 = p1 / 255.0
        if p2.max() > 1.0:
            p2 = p2 / 255.0
        
        # Compute per-pixel L2 distance
        diff = np.sqrt(np.sum((p1 - p2) ** 2, axis=2))
        
        return diff
    
    def visualize_grid(
        self,
        patches_list: List[List[np.ndarray]],
        query_patches: Optional[List[np.ndarray]] = None,
        labels: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Visualize multiple retrieval results in a grid.
        
        Args:
            patches_list: List of patch lists (one list per query)
            query_patches: Optional list of query patches
            labels: Optional labels for each column
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        num_queries = len(patches_list)
        max_k = max(len(p) for p in patches_list)
        
        # Determine number of rows (including query row)
        num_rows = max_k + (1 if query_patches else 0)
        num_cols = num_queries
        
        fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=(self.figsize[0], max(4, num_rows * 2)),
            dpi=100
        )
        
        # Ensure axes is 2D
        if num_rows == 1 or num_cols == 1:
            axes = axes.reshape(num_rows, num_cols)
        
        # Plot query patches
        row = 0
        if query_patches:
            for col, query in enumerate(query_patches):
                ax = axes[row, col]
                self._plot_patch(ax, query, "Query", is_query=True)
            row += 1
        
        # Plot retrieved patches
        for col, patches in enumerate(patches_list):
            for i, patch in enumerate(patches):
                ax = axes[row + i, col]
                self._plot_patch(ax, patch, f"Result {i+1}")
            
            # Hide unused cells
            for i in range(len(patches), max_k):
                if row + i < num_rows:
                    axes[row + i, col].axis('off')
        
        fig.suptitle(
            f"DA-CLIP Batch Retrieval (Top-{max_k})",
            fontsize=14,
            fontweight='bold'
        )
        
        # Save and show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            if self.debug:
                print(f" Grid figure saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_distance_distribution(
        self,
        distances_list: List[np.ndarray],
        labels: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot distribution of retrieval distances.
        
        Args:
            distances_list: List of distance arrays
            labels: Labels for each distribution
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if labels is None:
            labels = [f"Query {i+1}" for i in range(len(distances_list))]
        
        for distances, label in zip(distances_list, labels):
            ax.hist(distances, bins=50, alpha=0.6, label=label)
        
        ax.set_xlabel("Distance", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Retrieval Distance Distribution", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            if self.debug:
                print(f" Distance plot saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
