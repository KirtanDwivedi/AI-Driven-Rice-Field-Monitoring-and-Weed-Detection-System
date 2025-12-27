import numpy as np
import os
from pathlib import Path
from skimage import filters, morphology, exposure
try:
    from PIL import Image
except ImportError:
    import Image

class RiceWeedDetection:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.band_files = {
            'red': 'red.tif', 'green': 'green.tif', 'blue': 'blue.tif',
            'nir': 'nir.tif', 'red_edge': 'red edge.tif'
        }

    def load_band(self, folder, band_name):
        path = self.base_path / str(folder) / self.band_files[band_name]
        try:
            img = Image.open(path)
            data = np.array(img).astype(np.float32)
            # Normalize 0-1 (handles both 8-bit and 16-bit tifs)
            return data / (np.max(data) + 1e-6)
        except Exception:
            return None

    def detect_weeds(self, folder):
        print(f"Processing Folder {folder}...")
        bands = {name: self.load_band(folder, name) for name in self.band_files}
        
        if any(b is None for b in bands.values()):
            print(f"Skipping Folder {folder}: Missing bands.")
            return None

        # 1. VEGETATION MASK (GNDVI) - To remove soil noise
        gndvi = (bands['nir'] - bands['green']) / (bands['nir'] + bands['green'] + 1e-6)
        veg_mask = gndvi > filters.threshold_otsu(gndvi)

        # 2. WEED DISCRIMINATION (WSRI) - Best for rice vs weeds
        # WSRI = (RedEdge - Red) / (RedEdge - Blue)
        wsri = (bands['red_edge'] - bands['red']) / (bands['red_edge'] - bands['blue'] + 1e-6)
        
        # 3. INTENSITY CALCULATION (Weighted Vigor Score)
        # Using NDRE for chlorophyll 'depth' + WSRI for weed likelihood
        ndre = (bands['nir'] - bands['red_edge']) / (bands['nir'] + bands['red_edge'] + 1e-6)
        raw_intensity = (wsri * 0.7) + (ndre * 0.3)
        
        # Apply mask and normalize
        weed_intensity_map = np.where(veg_mask, raw_intensity, 0)
        # Threshold to keep only high-likelihood weed pixels
        thresh = np.percentile(weed_intensity_map[veg_mask], 85) 
        weed_mask = (weed_intensity_map > thresh) & veg_mask
        
        # Clean noise
        weed_mask = morphology.binary_opening(weed_mask, morphology.disk(1))
        
        # 4. GENERATE MATRIX [[row, col, intensity]]
        indices = np.argwhere(weed_mask)
        matrix = []
        for r, c in indices:
            val = weed_intensity_map[r, c]
            # Rescale to 0.0-1.0 range
            norm_val = round(float((val - thresh) / (np.max(weed_intensity_map) - thresh + 1e-6)), 3)
            if norm_val > 0.1: # Only include confident detections
                matrix.append([int(r), int(c), norm_val])
        
        return matrix

    def run_all(self):
        for i in range(1, 7): # Processes folders 1, 2, 3, 4, 5, 6
            result_matrix = self.detect_weeds(i)
            if result_matrix:
                output_file = f"field_{i}_result.txt"
                with open(output_file, "w") as f:
                    f.write(str(result_matrix))
                print(f"Success: {output_file} created with {len(result_matrix)} points.")

if __name__ == "__main__":
    # Update this path to your folder
    PATH = r"E:\Visual Studio Code\Projects\Capstone_Weed_Detection\ORTHOMOSAIC-2023"
    RiceWeedDetection(PATH).run_all()