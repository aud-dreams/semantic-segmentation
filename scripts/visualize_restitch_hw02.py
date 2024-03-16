import pyprojroot
import sys
root = pyprojroot.here()
sys.path.append(str(root))
from src.preprocessing.subtile_esd_hw02 import restitch
from pathlib import Path
import numpy as np

def main():
    import pyprojroot
    root = pyprojroot.here()
    import matplotlib.pyplot as plt
    sys.path.append(root)
    print(f'Added {root} to path.')
    
    # stitched_sentinel2 = restitch(Path(root/"data/processed/Train/subtiles"), "sentinel2", "Tile1", (0,4), (0,4))
    stitched_sentinel2, metadata = restitch(Path(root/"data/processed/Train/subtiles"), "sentinel2", "Tile1", (0,4), (0,4))
    plt.imshow(np.dstack([stitched_sentinel2[0,3,:,:], stitched_sentinel2[0,2,:,:], stitched_sentinel2[0,1,:,:]]))
    plt.savefig(Path(root / 'plots' / "restitch_scatterplot.png"))

if __name__ == "__main__":
    main()