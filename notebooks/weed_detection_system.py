#!/usr/bin/env python3
"""
Multi-Spectral Weed Detection System for Rice Fields
Based on research-backed vegetation indices and machine learning approaches
"""

import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print("Warning: GDAL not available, will try alternative methods")
    gdal = None

try:
    from skimage import io, filters, morphology, measure
    from skimage.util import img_as_float
    from scipy import ndimage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"Warning: Some libraries not available: {e}")


class VegetationIndices:
    """Calculate research-backed vegetation indices for weed detection"""
    
    @staticmethod
    def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """
        Normalized Difference Vegetation Index
        NDVI = (NIR - Red) / (NIR + Red)
        Range: -1 to 1, higher values indicate more vegetation
        """
        denominator = nir + red
        denominator = np.where(denominator == 0, 0.0001, denominator)
        return (nir - red) / denominator
    
    @staticmethod
    def ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
        """
        Normalized Difference Red Edge Index
        NDRE = (NIR - RedEdge) / (NIR + RedEdge)
        Sensitive to chlorophyll content and early stress detection
        """
        denominator = nir + red_edge
        denominator = np.where(denominator == 0, 0.0001, denominator)
        return (nir - red_edge) / denominator
    
    @staticmethod
    def gndvi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
        """
        Green Normalized Difference Vegetation Index
        GNDVI = (NIR - Green) / (NIR + Green)
        More sensitive to chlorophyll than NDVI
        """
        denominator = nir + green
        denominator = np.where(denominator == 0, 0.0001, denominator)
        return (nir - green) / denominator
    
    @staticmethod
    def wsri(red_edge: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Weed-Sensitive Ratio Index (WSRI)
        WSRI = (RedEdge - Red) / (RedEdge - Blue)
        Developed specifically for weed discrimination
        Research: Xia et al. 2022, Frontiers in Plant Science
        """
        denominator = red_edge - blue
        denominator = np.where(np.abs(denominator) < 0.0001, 0.0001, denominator)
        return (red_edge - red) / denominator
    
    @staticmethod
    def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
        """
        Soil Adjusted Vegetation Index
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        L = 0.5 for intermediate vegetation density
        """
        denominator = nir + red + L
        denominator = np.where(denominator == 0, 0.0001, denominator)
        return ((nir - red) / denominator) * (1 + L)
    
    @staticmethod
    def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Enhanced Vegetation Index
        EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
        """
        denominator = nir + 6*red - 7.5*blue + 1
        denominator = np.where(denominator == 0, 0.0001, denominator)
        return 2.5 * ((nir - red) / denominator)


class MultiSpectralWeedDetector:
    """
    Weed detection system using multi-spectral imagery
    Implements research-backed preprocessing and detection pipeline
    """
    
    def __init__(self, field_folders: List[str], 
                 base_path: str = r"E:\Visual Studio Code\Projects\Capstone_Weed_Detection\ORTHOMOSAIC-2023"):
        """
        Initialize weed detector
        
        Args:
            field_folders: List of folder names containing band images (e.g., ['1', '2', '3'])
            base_path: Base directory path (default set to your ORTHOMOSAIC-2023 folder)
        """
        self.field_folders = field_folders
        self.base_path = Path(base_path)
        self.band_names = ['red.tif', 'green.tif', 'blue.tif', 'nir.tif', 'red edge.tif']
        self.results = {}

        if not self.base_path.exists():
            print(f"Warning: base path does not exist: {self.base_path!s}. Verify the path or provide correct base_path.")
    
    def load_band(self, field_folder: str, band_name: str) -> np.ndarray:
        """Load a single band image and normalize to 0-1 range"""
        file_path = self.base_path / field_folder / band_name
        
        # Try GDAL first for proper GeoTIFF handling
        if gdal is not None:
            try:
                dataset = gdal.Open(str(file_path))
                if dataset is not None:
                    band = dataset.GetRasterBand(1)
                    data = band.ReadAsArray()
                    # Normalize to 0-1
                    data = data.astype(np.float32)
                    if data.max() > 1.0:
                        data = data / data.max()
                    return data
            except Exception as e:
                print(f"GDAL failed for {file_path}, trying alternative: {e}")
        
        # Fallback to skimage
        try:
            from PIL import Image
            img = Image.open(file_path)
            data = np.array(img, dtype=np.float32)
            if data.max() > 1.0:
                data = data / data.max()
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_field_bands(self, field_folder: str) -> Dict[str, np.ndarray]:
        """Load all bands for a field"""
        bands = {}
        band_mapping = {
            'red.tif': 'red',
            'green.tif': 'green',
            'blue.tif': 'blue',
            'nir.tif': 'nir',
            'red edge.tif': 'red_edge'
        }
        
        print(f"Loading bands for field {field_folder}...")
        for band_file, band_key in band_mapping.items():
            band_data = self.load_band(field_folder, band_file)
            if band_data is not None:
                bands[band_key] = band_data
                print(f"  ✓ {band_key}: {band_data.shape}")
            else:
                print(f"  ✗ {band_key}: Failed to load")
        
        return bands
    
    def preprocess_bands(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocessing pipeline based on research recommendations:
        1. Radiometric normalization (already done in loading)
        2. Denoising
        3. Band alignment check
        """
        processed = {}
        
        for band_name, band_data in bands.items():
            # Apply Gaussian denoising (research recommendation)
            denoised = filters.gaussian(band_data, sigma=1.0, preserve_range=True)
            processed[band_name] = denoised
        
        return processed
    
    def calculate_all_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate all vegetation indices"""
        indices = {}
        vi = VegetationIndices()
        
        if all(k in bands for k in ['nir', 'red']):
            indices['ndvi'] = vi.ndvi(bands['nir'], bands['red'])
            indices['savi'] = vi.savi(bands['nir'], bands['red'])
        
        if all(k in bands for k in ['nir', 'red_edge']):
            indices['ndre'] = vi.ndre(bands['nir'], bands['red_edge'])
        
        if all(k in bands for k in ['nir', 'green']):
            indices['gndvi'] = vi.gndvi(bands['nir'], bands['green'])
        
        if all(k in bands for k in ['red_edge', 'red', 'blue']):
            indices['wsri'] = vi.wsri(bands['red_edge'], bands['red'], bands['blue'])
        
        if all(k in bands for k in ['nir', 'red', 'blue']):
            indices['evi'] = vi.evi(bands['nir'], bands['red'], bands['blue'])
        
        return indices
    
    def create_composite_weed_score(self, indices: Dict[str, np.ndarray], 
                                    bands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create composite weed detection score using ensemble of indices
        Research shows combining multiple indices improves accuracy
        """
        scores = []
        weights = []
        
        # WSRI: highest weight (research: 81-92% accuracy for weed discrimination)
        if 'wsri' in indices:
            wsri_normalized = (indices['wsri'] - np.nanmin(indices['wsri'])) / \
                            (np.nanmax(indices['wsri']) - np.nanmin(indices['wsri']) + 1e-10)
            scores.append(wsri_normalized)
            weights.append(0.35)  # Highest weight
        
        # NDVI: strong baseline indicator
        if 'ndvi' in indices:
            # Weeds typically have different NDVI than rice
            ndvi_score = np.abs(indices['ndvi'] - 0.6)  # 0.6 is typical rice NDVI
            ndvi_normalized = ndvi_score / (np.nanmax(ndvi_score) + 1e-10)
            scores.append(ndvi_normalized)
            weights.append(0.25)
        
        # NDRE: sensitive to chlorophyll differences
        if 'ndre' in indices:
            ndre_normalized = (indices['ndre'] - np.nanmin(indices['ndre'])) / \
                            (np.nanmax(indices['ndre']) - np.nanmin(indices['ndre']) + 1e-10)
            scores.append(ndre_normalized)
            weights.append(0.20)
        
        # GNDVI: chlorophyll sensitivity
        if 'gndvi' in indices:
            gndvi_normalized = (indices['gndvi'] - np.nanmin(indices['gndvi'])) / \
                             (np.nanmax(indices['gndvi']) - np.nanmin(indices['gndvi']) + 1e-10)
            scores.append(gndvi_normalized)
            weights.append(0.10)
        
        # Red edge band: research shows strong separability
        if 'red_edge' in bands:
            re_normalized = bands['red_edge'] / (np.nanmax(bands['red_edge']) + 1e-10)
            scores.append(re_normalized)
            weights.append(0.10)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted combination
        composite_score = np.zeros_like(scores[0])
        for score, weight in zip(scores, weights):
            composite_score += weight * score
        
        return composite_score
    
    def adaptive_threshold(self, score_map: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply adaptive thresholding based on Otsu's method
        Research recommendation: data-driven threshold selection
        """
        # Remove NaN and infinite values
        valid_scores = score_map[np.isfinite(score_map)]
        
        if len(valid_scores) == 0:
            return np.zeros_like(score_map, dtype=bool), 0.0
        
        # Otsu's threshold
        try:
            threshold = filters.threshold_otsu(valid_scores)
        except:
            # Fallback to percentile-based threshold
            threshold = np.percentile(valid_scores, 75)
        
        # Apply threshold
        weed_mask = score_map > threshold
        
        return weed_mask, threshold
    
    def morphological_cleanup(self, weed_mask: np.ndarray, 
                            min_size: int = 5) -> np.ndarray:
        """
        Morphological post-processing to remove noise
        Research recommendation: remove small patches
        """
        # Remove small objects
        cleaned = morphology.remove_small_objects(weed_mask, min_size=min_size)
        
        # Fill small holes
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size)
        
        # Binary closing to connect nearby regions
        cleaned = morphology.binary_closing(cleaned, morphology.disk(2))
        
        return cleaned
    
    def calculate_weed_intensity(self, score_map: np.ndarray, 
                                 weed_mask: np.ndarray,
                                 indices: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate weed intensity/depth based on multiple factors
        Research: intensity correlates with biomass, vigor, and spectral response
        
        Intensity calculation combines:
        1. Composite score magnitude
        2. NDVI deviation (vigor indicator)
        3. Red edge response (biomass indicator)
        """
        intensity_map = np.zeros_like(score_map)
        
        # Base intensity from composite score
        intensity_map = score_map.copy()
        
        # Enhance with NDVI deviation (vigor)
        if 'ndvi' in indices:
            ndvi_deviation = np.abs(indices['ndvi'] - 0.6)
            ndvi_factor = ndvi_deviation / (np.nanmax(ndvi_deviation) + 1e-10)
            intensity_map = 0.6 * intensity_map + 0.4 * ndvi_factor
        
        # Apply weed mask
        intensity_map = intensity_map * weed_mask
        
        # Normalize to 0-1 range
        if intensity_map.max() > 0:
            intensity_map = intensity_map / intensity_map.max()
        
        return intensity_map
    
    def extract_weed_locations(self, weed_mask: np.ndarray, 
                               intensity_map: np.ndarray,
                               min_intensity: float = 0.1) -> List[List[float]]:
        """
        Extract weed locations with intensities
        Output format: [[row, col, intensity], ...]
        """
        weed_locations = []
        
        # Find weed pixels
        weed_coords = np.argwhere(weed_mask)
        
        for coord in weed_coords:
            row, col = coord
            intensity = intensity_map[row, col]
            
            # Filter by minimum intensity
            if intensity >= min_intensity:
                weed_locations.append([int(row), int(col), float(intensity)])
        
        return weed_locations
    
    def detect_weeds_in_field(self, field_folder: str) -> Dict:
        """
        Complete weed detection pipeline for one field
        """
        print(f"\n{'='*60}")
        print(f"Processing Field: {field_folder}")
        print(f"{'='*60}")
        
        # Load bands
        bands = self.load_field_bands(field_folder)
        if len(bands) < 4:
            print(f"Error: Insufficient bands loaded for field {field_folder}")
            return None
        
        # Preprocess
        print("Preprocessing bands...")
        bands = self.preprocess_bands(bands)
        
        # Calculate vegetation indices
        print("Calculating vegetation indices...")
        indices = self.calculate_all_indices(bands)
        print(f"  Calculated indices: {list(indices.keys())}")
        
        # Create composite weed score
        print("Creating composite weed detection score...")
        composite_score = self.create_composite_weed_score(indices, bands)
        
        # Adaptive thresholding
        print("Applying adaptive thresholding...")
        weed_mask, threshold = self.adaptive_threshold(composite_score)
        print(f"  Threshold value: {threshold:.4f}")
        print(f"  Initial weed pixels: {weed_mask.sum()}")
        
        # Morphological cleanup
        print("Performing morphological cleanup...")
        weed_mask_cleaned = self.morphological_cleanup(weed_mask, min_size=10)
        print(f"  Cleaned weed pixels: {weed_mask_cleaned.sum()}")
        
        # Calculate intensity
        print("Calculating weed intensities...")
        intensity_map = self.calculate_weed_intensity(composite_score, weed_mask_cleaned, indices)
        
        # Extract locations
        print("Extracting weed locations...")
        weed_locations = self.extract_weed_locations(weed_mask_cleaned, intensity_map, min_intensity=0.15)
        print(f"  Total weed detections: {len(weed_locations)}")
        
        # Calculate statistics
        if len(weed_locations) > 0:
            intensities = [loc[2] for loc in weed_locations]
            stats = {
                'total_weeds': len(weed_locations),
                'mean_intensity': float(np.mean(intensities)),
                'max_intensity': float(np.max(intensities)),
                'min_intensity': float(np.min(intensities)),
                'std_intensity': float(np.std(intensities)),
                'threshold_used': float(threshold)
            }
        else:
            stats = {
                'total_weeds': 0,
                'mean_intensity': 0.0,
                'max_intensity': 0.0,
                'min_intensity': 0.0,
                'std_intensity': 0.0,
                'threshold_used': float(threshold)
            }
        
        result = {
            'field': field_folder,
            'weed_locations': weed_locations,
            'statistics': stats,
            'image_shape': bands['red'].shape
        }
        
        return result
    
    def process_all_fields(self) -> Dict:
        """Process all fields and compile results"""
        print("\n" + "="*60)
        print("MULTI-SPECTRAL WEED DETECTION SYSTEM")
        print("="*60)
        print(f"Fields to process: {self.field_folders}")
        print(f"Base path: {self.base_path}")
        
        all_results = {}
        
        for field_folder in self.field_folders:
            result = self.detect_weeds_in_field(field_folder)
            if result is not None:
                all_results[field_folder] = result
        
        self.results = all_results
        return all_results
    
    def save_results(self, output_dir: str = "/home/sandbox/weed_detection_results"):
        """Save results in multiple formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Saving Results")
        print(f"{'='*60}")
        
        # Save JSON with all details
        json_path = output_path / "weed_detection_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved JSON results: {json_path}")
        
        # Save individual field matrices
        for field, result in self.results.items():
            matrix_path = output_path / f"field_{field}_weed_matrix.txt"
            locations = result['weed_locations']
            
            with open(matrix_path, 'w') as f:
                f.write(f"# Weed Detection Results for Field {field}\n")
                f.write(f"# Format: [row, col, intensity]\n")
                f.write(f"# Total detections: {len(locations)}\n")
                f.write(f"# Statistics:\n")
                for key, value in result['statistics'].items():
                    f.write(f"#   {key}: {value}\n")
                f.write("\n")
                
                for loc in locations:
                    f.write(f"[{loc[0]}, {loc[1]}, {loc[2]:.4f}]\n")
            
            print(f"✓ Saved field {field} matrix: {matrix_path}")
        
        # Save summary report
        summary_path = output_path / "detection_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("WEED DETECTION SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            for field, result in self.results.items():
                f.write(f"Field {field}:\n")
                f.write(f"  Image dimensions: {result['image_shape']}\n")
                f.write(f"  Total weed detections: {result['statistics']['total_weeds']}\n")
                f.write(f"  Mean intensity: {result['statistics']['mean_intensity']:.4f}\n")
                f.write(f"  Max intensity: {result['statistics']['max_intensity']:.4f}\n")
                f.write(f"  Std intensity: {result['statistics']['std_intensity']:.4f}\n")
                f.write(f"  Detection threshold: {result['statistics']['threshold_used']:.4f}\n")
                f.write("\n")
        
        print(f"✓ Saved summary report: {summary_path}")
        
        return output_path


def main():
    """Main execution function"""
    
    # Configuration
    field_folders = ['1', '2', '3', '4', '5', '6']
    base_path = r"E:\Visual Studio Code\Projects\Capstone_Weed_Detection\ORTHOMOSAIC-2023"
    
    # Initialize detector
    detector = MultiSpectralWeedDetector(field_folders, base_path)
    
    # Process all fields
    results = detector.process_all_fields()
    
    # Save results
    output_dir = detector.save_results()
    
    # Print final summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total fields processed: {len(results)}")
    print(f"Results saved to: {output_dir}")
    print("\nOutput format: [[row, col, intensity], ...]")
    print("  - row, col: pixel coordinates of weed location")
    print("  - intensity: weed vigor/depth (0.0 to 1.0)")
    print("\nFiles generated:")
    print("  - weed_detection_results.json (complete results)")
    print("  - field_X_weed_matrix.txt (per-field matrices)")
    print("  - detection_summary.txt (statistics summary)")
    

if __name__ == "__main__":
    main()
