# Images for README

This folder should contain visual assets for the main README.md file. Below are suggestions for images to extract from the Project_Report.pdf:

## Recommended Images to Extract

### 1. **example_mask_overlay.png**
- **Source:** Figure 1 from Project_Report.pdf (page 2)
- **Description:** Example of aerial image with corresponding segmentation mask overlapped at 50% opacity
- **Shows:** The 24 semantic classes with color coding
- **Used in:** README header section

### 2. **multiscale_patches.png**
- **Source:** Figure 2 from Project_Report.pdf (page 3)
- **Description:** Visual comparison of original image and reconstructed version using augmented patches
- **Shows:** Multi-scale stitch level approach with the last 5 patches showing different scales
- **Used in:** Multi-Scale Stitch Levels section

### 3. **prediction_results.png**
- **Source:** Figure 5 or Figure 6 from Project_Report.pdf (pages 8-9)
- **Description:** Complete visualization grid showing:
  - Original image
  - True mask
  - Predicted mask
  - Confidence heatmap
  - Entropy map
  - Predicted mask overlay
- **Shows:** Model predictions with confidence and uncertainty visualization
- **Used in:** Results > Visualization Examples section

### 4. **patch_size_comparison.png** (Optional)
- **Source:** Figure 3 from Project_Report.pdf (page 6)
- **Description:** Validation history graphs comparing different patch sizes (500, 1000, 2000)
- **Shows:** Loss, Accuracy, IoU, and Dice score curves
- **Used in:** Could add to Methodology > Patch-Based Processing

### 5. **training_history.png** (Optional)
- **Source:** Figure 7 or Figure 8 from Project_Report.pdf (page 9)
- **Description:** Training metrics over epochs (Loss, Accuracy, IoU, Dice)
- **Shows:** Model convergence and learning behavior
- **Used in:** Could add to Results section

### 6. **class_distribution.png** (Optional)
- **Source:** Could be created from dataset statistics
- **Description:** Bar chart showing frequency of each of the 24 semantic classes
- **Shows:** Class imbalance in the dataset
- **Used in:** Dataset section

## How to Extract Images from PDF

### Method 1: Using pdfimages (Linux/Mac)
```bash
# Install poppler-utils if needed
sudo apt-get install poppler-utils  # Ubuntu/Debian
brew install poppler  # Mac

# Extract all images
cd Final_Project/images
pdfimages -png ../Project_Report.pdf report_img
```

### Method 2: Using PDF viewers
1. Open Project_Report.pdf
2. Take screenshots of the desired figures
3. Crop and save as PNG files
4. Name them according to the list above

### Method 3: Using Python
```python
import fitz  # PyMuPDF
doc = fitz.open("Project_Report.pdf")
for page_num in range(len(doc)):
    page = doc[page_num]
    images = page.get_images()
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        with open(f"page{page_num}_img{img_index}.png", "wb") as f:
            f.write(image_bytes)
```

## Alternative: README without Images

The README is fully functional without images - all referenced images will simply not display but won't break the markdown. The textual content is comprehensive enough to understand the project.

## Image Specifications

- **Format:** PNG (preferred) or JPG
- **Resolution:** High enough to be readable (recommended: width ≥ 800px for main images)
- **File size:** Optimize for web (typically < 500KB per image)

## License Note

All images extracted from the Project_Report.pdf are part of the academic work by Mirko Morello and are covered under the same MIT License as the repository.
