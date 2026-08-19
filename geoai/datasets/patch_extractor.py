import numpy as np
from typing import Tuple

class PatchExtractor:
    """Standardized patch extraction API supporting custom patch sizes, strides, overlaps, and padding."""
    
    def __init__(self, patch_size: int = 15, stride: int = 1, overlap: float = 0.0, padding_mode: str = "reflect"):
        self.patch_size = patch_size
        # Overlap overrides stride if overlap > 0
        if overlap > 0.0 and overlap < 1.0:
            self.stride = max(1, int(patch_size * (1.0 - overlap)))
        else:
            self.stride = stride
        self.overlap = overlap
        self.padding_mode = padding_mode

    def extract(self, t1: np.ndarray, t2: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract patches from T1, T2, and labels.
        Returns:
            t1_patches: (N, C, P_h, P_w)
            t2_patches: (N, C, P_h, P_w)
            label_patches: (N, P_h, P_w)
        """
        C, H, W = t1.shape
        P = self.patch_size
        S = self.stride
        
        # Calculate padding needed to cover edge pixels
        pad_h = (S - (H - P) % S) % S
        pad_w = (S - (W - P) % S) % S
        
        # Apply padding if necessary
        if pad_h > 0 or pad_w > 0:
            t1_padded = np.pad(t1, ((0, 0), (0, pad_h), (0, pad_w)), mode=self.padding_mode)
            t2_padded = np.pad(t2, ((0, 0), (0, pad_h), (0, pad_w)), mode=self.padding_mode)
            label_padded = np.pad(label, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        else:
            t1_padded = t1
            t2_padded = t2
            label_padded = label
            
        H_pad, W_pad = t1_padded.shape[1], t1_padded.shape[2]
        
        t1_list = []
        t2_list = []
        lbl_list = []
        
        for y in range(0, H_pad - P + 1, S):
            for x in range(0, W_pad - P + 1, S):
                t1_list.append(t1_padded[:, y:y+P, x:x+P])
                t2_list.append(t2_padded[:, y:y+P, x:x+P])
                lbl_list.append(label_padded[y:y+P, x:x+P])
                
        return (np.array(t1_list, dtype=np.float32), 
                np.array(t2_list, dtype=np.float32), 
                np.array(lbl_list, dtype=np.int64))
