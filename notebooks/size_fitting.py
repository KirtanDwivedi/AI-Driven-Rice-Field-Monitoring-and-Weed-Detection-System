import os
import rasterio
from rasterio.windows import Window

LEFT, TOP = 1000, 1000
WIDTH, HEIGHT = 100, 100

INPUT_FOLDER = r"E:\Visual Studio Code\Projects\Capstone_Weed_Detection\ORTHOMOSAIC-2023\1"
OUTPUT_FOLDER = r"E:\Visual Studio Code\Projects\Capstone_Weed_Detection\ORTHOMOSAIC-2023\1\Fitted_Samples"

bands = ['red.tif', 'green.tif', 'blue.tif', 'nir.tif', 'red edge.tif']

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Cropping area: ({LEFT}, {TOP}) size {WIDTH}x{HEIGHT}")

for band in bands:
    in_path = os.path.join(INPUT_FOLDER, band)
    out_path = os.path.join(OUTPUT_FOLDER, band)

    with rasterio.open(in_path) as src:
        window = Window(LEFT, TOP, WIDTH, HEIGHT)
        data = src.read(window=window)

        profile = src.profile
        profile.update(
            height=HEIGHT,
            width=WIDTH,
            transform=rasterio.windows.transform(window, src.transform)
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

    print(f"✓ Saved {band}")

print("\nDONE ✅ Files written to Fitted_Samples")
