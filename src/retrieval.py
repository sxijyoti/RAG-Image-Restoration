"""
Phase 4: Retrieval Module for DA-CLIP Image Restoration

This module implements FAISS-based retrieval for finding similar clean patches
given a degraded patch. It bridges the DA-CLIP encoder with the FAISS index.

Key Features:
- Load FAISS index and patch mappings
- Query index with degradation-aware embeddings
- Map FAISS indices back to original patches
- Support for top-k retrieval with distance metrics
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import faiss

# Try to import PIL for image loading
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class FAISSIndexLoader:
    """Load and validate FAISS index and patch mappings."""
    
    def __init__(self, debug: bool = True):
        """
        Initialize FAISS index loader.
        
        Args:
            debug: Print loading information
        """
        self.debug = debug
        self.index = None
        self.patch_map = None
        self.index_path = None
        self.patch_map_path = None
        self.index_info = {}
    
    def load_index(self, index_path: Union[str, Path]) -> faiss.Index:
        """
        Load FAISS index from disk.
        
        Args:
            index_path: Path to FAISS index file
            
        Returns:
            FAISS Index object
            
        Raises:
            FileNotFoundError: If index file not found
            RuntimeError: If index loading fails
        """
        index_path = Path(index_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        try:
            self.index = faiss.read_index(str(index_path))
            self.index_path = index_path
            
            if self.debug:
                ntotal = self.index.ntotal
                print(f" FAISS Index loaded")
                print(f"   Path: {index_path}")
                print(f"   Size: {ntotal:,} vectors")
                print(f"   Dimension: {self.index.d}")
                print(f"   Index type: {type(self.index).__name__}")
            
            self.index_info['ntotal'] = self.index.ntotal
            self.index_info['dimension'] = self.index.d
            self.index_info['type'] = type(self.index).__name__
            
            return self.index
        
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")
    
    def load_patch_map(self, patch_map_path: Union[str, Path]) -> Dict:
        """
        Load patch mapping (index → image + coordinates).
        
        Args:
            patch_map_path: Path to patch_map.json
            
        Returns:
            Dictionary mapping index → {image_name, x, y}
            
        Raises:
            FileNotFoundError: If patch_map not found
            json.JSONDecodeError: If patch_map is invalid JSON
        """
        patch_map_path = Path(patch_map_path)
        
        if not patch_map_path.exists():
            raise FileNotFoundError(f"Patch map not found: {patch_map_path}")
        
        try:
            with open(patch_map_path, 'r') as f:
                self.patch_map = json.load(f)
            
            self.patch_map_path = patch_map_path
            
            if self.debug:
                print(f" Patch map loaded")
                print(f"   Path: {patch_map_path}")
                print(f"   Entries: {len(self.patch_map):,}")
                
                # Show a sample entry
                sample_key = next(iter(self.patch_map))
                sample_entry = self.patch_map[sample_key]
                print(f"   Sample: {sample_key} → {sample_entry}")
            
            # Validate patch_map structure
            for key, value in list(self.patch_map.items())[:3]:
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid patch_map format: entry {key} is not a dict")
                if 'image' not in value or 'x' not in value or 'y' not in value:
                    raise ValueError(
                        f"Invalid patch_map format: entry {key} missing required keys "
                        "(image, x, y)"
                    )
            
            return self.patch_map
        
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse patch_map JSON: {e.msg}",
                e.doc,
                e.pos
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load patch_map: {e}")


class PatchRetriever:
    """
    Retrieve similar clean patches from FAISS index given a query embedding.
    """
    
    def __init__(
        self,
        index: faiss.Index,
        patch_map: Dict,
        normalize_query: bool = True,
        debug: bool = True
    ):
        """
        Initialize patch retriever.
        
        Args:
            index: FAISS Index object
            patch_map: Dictionary mapping index → patch metadata
            normalize_query: Whether to normalize query embeddings (must match index)
            debug: Print retrieval information
        """
        self.index = index
        self.patch_map = patch_map
        self.normalize_query = normalize_query
        self.debug = debug
        self.embedding_dim = index.d
        
        if debug:
            print(f" PatchRetriever initialized")
            print(f"   Index size: {index.ntotal:,}")
            print(f"   Embedding dim: {self.embedding_dim}")
            print(f"   Normalize query: {normalize_query}")
    
    def search(
        self,
        embedding: Union[np.ndarray, list],
        k: int = 5,
        return_metadata: bool = True,
        debug: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
        """
        Search FAISS index for top-k similar patches.
        
        Args:
            embedding: Query embedding (embedding_dim,) or (1, embedding_dim)
            k: Number of top results to return
            return_metadata: Whether to include patch metadata
            debug: Print search details
            
        Returns:
            (indices, distances, metadata) or (indices, distances, None)
            - indices: Array of shape (1, k) with FAISS indices
            - distances: Array of shape (1, k) with distances/similarities
            - metadata: List of k dicts with {index, image, x, y, distance}
            
        Raises:
            ValueError: If embedding has wrong shape
            IndexError: If index not loaded
        """
        if self.index is None:
            raise IndexError("FAISS index not loaded")
        
        # Validate and reshape embedding
        embedding = np.array(embedding, dtype=np.float32)
        
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        if embedding.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {embedding.shape[0]}")
        
        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, "
                f"got {embedding.shape[1]}"
            )
        
        # Normalize if required
        if self.normalize_query:
            embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
        
        # Search
        distances, indices = self.index.search(embedding, k)
        
        if debug or self.debug:
            print(f" Search Results:")
            print(f"   Query embedding shape: {embedding.shape}")
            print(f"   Top-k: {k}")
            print(f"   Indices: {indices[0]}")
            print(f"   Distances: {distances[0]}")
        
        # Prepare metadata
        metadata = None
        if return_metadata:
            metadata = self._prepare_metadata(indices[0], distances[0])
        
        return indices, distances, metadata
    
    def _prepare_metadata(
        self,
        indices: np.ndarray,
        distances: np.ndarray
    ) -> List[Dict]:
        """
        Convert FAISS indices to patch metadata.
        
        Args:
            indices: Array of FAISS indices
            distances: Array of corresponding distances
            
        Returns:
            List of dicts with {index, image, x, y, distance}
        """
        metadata = []
        
        for idx, dist in zip(indices, distances):
            idx_str = str(int(idx))
            
            if idx_str not in self.patch_map:
                # Try without string conversion
                try:
                    patch_info = self.patch_map.get(idx, {})
                except (KeyError, TypeError):
                    patch_info = {}
            else:
                patch_info = self.patch_map[idx_str]
            
            entry = {
                'faiss_index': int(idx),
                'image': patch_info.get('image', 'unknown'),
                'x': patch_info.get('x', -1),
                'y': patch_info.get('y', -1),
                'distance': float(dist)
            }
            metadata.append(entry)
        
        return metadata
    
    def batch_search(
        self,
        embeddings: np.ndarray,
        k: int = 5,
        batch_size: int = 32,
        return_metadata: bool = True,
        debug: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[List[Dict]]]]:
        """
        Search for multiple embeddings (batch mode).
        
        Args:
            embeddings: Array of shape (batch_size, embedding_dim)
            k: Number of top results per query
            batch_size: Processing batch size (for memory efficiency)
            return_metadata: Whether to include metadata
            debug: Print debug info
            
        Returns:
            (all_indices, all_distances, all_metadata)
        """
        embeddings = np.array(embeddings, dtype=np.float32)
        
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got {embeddings.ndim}D")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        all_indices = []
        all_distances = []
        all_metadata = []
        
        num_queries = embeddings.shape[0]
        
        for i in range(0, num_queries, batch_size):
            batch = embeddings[i:i+batch_size]
            
            # Search batch
            distances, indices = self.index.search(batch, k)
            all_indices.append(indices)
            all_distances.append(distances)
            
            # Get metadata
            if return_metadata:
                batch_metadata = []
                for idx_array, dist_array in zip(indices, distances):
                    meta = self._prepare_metadata(idx_array, dist_array)
                    batch_metadata.append(meta)
                all_metadata.append(batch_metadata)
        
        # Concatenate
        all_indices = np.vstack(all_indices) if all_indices else np.array([])
        all_distances = np.vstack(all_distances) if all_distances else np.array([])
        all_metadata = [item for sublist in all_metadata for item in sublist] if all_metadata else None
        
        if debug:
            print(f"Batch search complete: {num_queries} queries processed")
        
        return all_indices, all_distances, all_metadata


class PatchLoader:
    """
    Load actual image patches from disk given patch metadata.
    """
    
    def __init__(self, dataset_root: Union[str, Path] = None, debug: bool = True):
        """
        Initialize patch loader.
        
        Args:
            dataset_root: Root directory containing images
            debug: Print loading information
        """
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.debug = debug
        self._image_cache = {}
        self._cache_size = 0
        self.max_cache_size = 500  # MB
        
        if self.debug and self.dataset_root:
            print(f" PatchLoader initialized")
            print(f"   Dataset root: {self.dataset_root}")
    
    def set_dataset_root(self, dataset_root: Union[str, Path]):
        """Set the dataset root directory."""
        self.dataset_root = Path(dataset_root)
        if self.debug:
            print(f" Dataset root set to: {self.dataset_root}")
    
    def load_patch(
        self,
        image_name: str,
        x: int,
        y: int,
        patch_size: int = 64,
        debug: bool = False
    ) -> Optional[np.ndarray]:
        """
        Load a single patch from an image.
        
        Args:
            image_name: Filename of image (relative to dataset_root)
            x: X coordinate of patch (top-left)
            y: Y coordinate of patch (top-left)
            patch_size: Size of patch to extract (default 64)
            debug: Print debug info
            
        Returns:
            Patch as numpy array (patch_size, patch_size, 3) or None if error
        """
        if not self.dataset_root:
            if debug or self.debug:
                print("  Dataset root not set")
            return None
        
        if not PIL_AVAILABLE:
            if debug or self.debug:
                print("  PIL not available")
            return None
        
        try:
            # Load image (with caching)
            image_path = self.dataset_root / image_name
            
            if not image_path.exists():
                if debug or self.debug:
                    print(f"  Image not found: {image_path}")
                return None
            
            # Load from cache or disk
            if str(image_path) in self._image_cache:
                image = self._image_cache[str(image_path)]
            else:
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
                
                # Cache image
                self._image_cache[str(image_path)] = image
            
            # Extract patch
            patch = image[y:y+patch_size, x:x+patch_size, :].copy()
            
            # Validate patch size
            if patch.shape != (patch_size, patch_size, 3):
                if debug or self.debug:
                    print(
                        f"  Patch incomplete: got {patch.shape}, "
                        f"expected ({patch_size}, {patch_size}, 3)"
                    )
                # Pad if necessary
                h, w = patch.shape[:2]
                if h < patch_size or w < patch_size:
                    padded = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                    padded[:h, :w] = patch
                    patch = padded
            
            return patch
        
        except Exception as e:
            if debug or self.debug:
                print(f" Error loading patch: {e}")
            return None
    
    def load_patches_from_metadata(
        self,
        metadata_list: List[Dict],
        patch_size: int = 64,
        debug: bool = False
    ) -> List[np.ndarray]:
        """
        Load multiple patches from metadata list.
        
        Args:
            metadata_list: List of dicts with {image, x, y, ...}
            patch_size: Size of patches to extract
            debug: Print debug info
            
        Returns:
            List of loaded patches (may be shorter if some failed)
        """
        patches = []
        
        for i, meta in enumerate(metadata_list):
            patch = self.load_patch(
                meta['image'],
                meta['x'],
                meta['y'],
                patch_size=patch_size,
                debug=debug
            )
            
            if patch is not None:
                patches.append(patch)
            elif debug or self.debug:
                print(f"Skipped patch {i}: {meta}")
        
        return patches
    
    def clear_cache(self):
        """Clear image cache."""
        self._image_cache.clear()
        self._cache_size = 0
        if self.debug:
            print(" Image cache cleared")
