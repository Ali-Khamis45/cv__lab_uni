# tutorial_1.0

## Purpose and reference notebooks

Use this guide when creating future Computer Vision task notebooks. Match the teaching style, cell structure, code level, and naming habits of Labs 1, 2, 3, and 4. The goal is to make each task look like a continuation of these labs and make every operation easy to follow.

This version is based on reading every markdown and code cell in the following notebooks, together with their saved text outputs and error messages:

| Reference | Cells reviewed | What it teaches | What to copy |
| --- | --- | --- | --- |
| [Lab_1.ipynb](Lab_1.ipynb) | 92: 49 markdown, 43 code | Python variables, mutability, types, conditions, loops, functions; NumPy creation, dtypes, indexing, slicing, arithmetic, and Boolean masks | Introduce a concept, give a small example, inspect its result |
| [Lab2_CV.ipynb](Lab2_CV.ipynb) | 63: 24 markdown, 39 code | Reading color and gray images; brightness, cropping, noise, drawing, splitting and recombining channels; OpenCV reading and display | Treat images as arrays; explain slices; display intermediate results |
| [Lab_3_CV.ipynb](Lab_3_CV.ipynb) | 22: 4 markdown, 18 code | Manual nearest-neighbor enlargement, crop-and-zoom, block-average shrinking, and bilinear interpolation | Allocate an output array, fill it with loops, then inspect and display it |
| [lab4_CV.ipynb](lab4_CV.ipynb) | 9: 4 markdown, 5 code (including an empty final cell) | Contrast stretching, histogram equalization, and RGB histogram matching | Explain the concept, implement the calculation manually, compare with scikit-image, and show images beside their histograms |

All 186 cells across Labs 1-4 were reviewed. Use the closest lab as the style reference: Labs 1-3 favor small demonstrations, while Lab 4 groups each algorithm into a longer section with a shared plotting helper and manual/library comparison.

The existing `Tasks_in_Lab/task120230129_1.ipynb` is empty, so it provides a filename example but no code or structure to imitate. These observations are a style guide, not an instructor-issued grading rubric. An actual assignment's requirements take priority.

## 1. Understand the course's approach

The progression is:

1. Understand Python variables, loops, and functions.
2. Understand NumPy arrays, dimensions, indexing, and arithmetic.
3. Treat an image as an array and manipulate its pixels directly.
4. Build image-processing algorithms from those operations.
5. Show the output so the effect of each step is visible.
6. Use intensity distributions and cumulative distributions to adjust contrast and match a reference image, then compare manual and library results.

Use simple procedural code. Small functions are appropriate when a calculation becomes reusable, as in Lab 3. The reference notebooks do not build classes, application frameworks, configuration systems, or elaborate utility layers.

For a task about implementing an algorithm, keep that algorithm visible. Follow Lab 3's use of loops, slicing, arithmetic, and array allocation instead of replacing the exercise with a ready-made resize call. Basic NumPy operations are part of the approach: the shrinking example explicitly uses `.mean()`, and Lab 4's manual implementations use `np.histogram`, `.cumsum()`, array indexing, and `np.argmin`. Manual implementation does not require writing every operation as a pixel loop. Lab 4 explicitly follows its manual algorithms with scikit-image comparisons.

## 2. Match the notebook structure

### Headings

Use short markdown headings, usually in their own cells:

```markdown
# Image Manipulation
## Part 1
### Read image
### Image array properties
### Draw plus sign
```

The labs use `#` for major topics or libraries, `##` for parts or subtopics, `###` for individual operations, and sometimes `####` for variants such as 1D and 2D arrays. Use only the levels needed. A short task does not need multiple Parts.

Some original headings omit the space after `#`. Add the space in new work so markdown renders reliably. A heading can share a markdown cell with a short definition, as it sometimes does in Lab 1.

### Cell rhythm

Use this sequence as the default:

```text
Markdown: short operation heading
Markdown: brief explanation, if needed
Code:     perform the operation
Code:     inspect or display the result
Markdown: explain an important detail, if needed
```

The labs also put an operation and its display in the same cell. Both are valid. Separate the display when the operation is longer or when checking the result is a useful teaching step.

Keep cells focused on one concept. Many cells are just one expression or a few lines, but function definitions naturally take longer. Do not split a loop or function across cells, and do not combine the entire task into one large cell.

Lab 1 is explanation-heavy, Lab 2 alternates explanations and image operations, and Lab 3 is mostly code. Lab 4 uses numbered algorithm headings, occasional **Concept:** and **The Math:** explanations, and longer code cells divided by numbered comments such as `SETUP`, `MANUAL IMPLEMENTATION`, and `SKIMAGE IMPLEMENTATION`. Match the closest example instead of forcing the same amount of prose or the same cell length into every task.

### Imports and setup

Use the aliases from the labs:

```python
import numpy as np
import skimage as ski
```

For an OpenCV section in Google Colab:

```python
import cv2
from google.colab.patches import cv2_imshow
```

Put imports before their first use, normally in a small setup cell. Lab 2 sometimes combines an import with loading an image and repeats imports at the start of Part 2; a separate import cell is a useful default, not a strict rule observed everywhere.

Lab 4 uses direct imports from skimage and Matplotlib:

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, data, io
```

Its original setup also imports `color` and `urllib.request`, but neither is used in the notebook. Include imports needed for the new task. Use `data.camera()` and `io.imread(...)` with this import style; use `ski.data.camera()` and `ski.io.imread(...)` with the earlier labs' style.

The notebooks contain Colab metadata. Lab 2 also uses Colab-specific imports and `/content/` paths. Adapt paths and display code if working locally. Lab 1 includes `!pip install numpy`; include installation cells only when the environment needs them.

## 3. Match the writing and naming style

Write short, plain explanations describing the operation or the reason behind it. Suitable examples are:

- "To get a section of the image, use slicing."
- "First, we copy the image because we want to keep the original."
- "Each pixel is copied into a larger block in the new image."

Use bullets for definitions and backticks for code expressions such as `image[:,:,0]`. Explain important details such as shared arrays, row/column order, or intensity ranges. Avoid a long introduction before a simple operation.

The naming style is mixed. Follow the section you are extending:

| Use | Examples from the labs |
| --- | --- |
| Small demonstrations | `x`, `y`, `nuArray`, `zeroArray` |
| Images and intermediate results | `image`, `newImage`, `imCamera`, `BrightnessIncrease` |
| Sizes and positions | `height`, `width`, `midHeight`, `midWidth` |
| Loop indexes | `i` for rows, `j` for columns, `k` for channels |
| Functions and parameters | `nearest_neighbor_upscale`, `width_scale`, `height_scale`, `new_width` |
| Other result names | `scaledImage`, `resized_image`, `zoomed_image` |
| Lab 4 functions | `plot_img_and_hist`, `manual_contrast_stretch`, `manual_histogram_equalization`, `calculate_cdf`, `manual_match_histograms` |
| Lab 4 intermediate values and comparisons | `hist`, `pdf`, `cdf`, `mapping`, `src_img`, `ref_img`, `img_eq_manual`, `img_eq_skimage` |

Use similar short, understandable names. Keep existing names when extending a notebook, but do not introduce spelling errors just to imitate it.

Comments should explain a useful step, for example `# rows`, `# red line`, or `# Copy each pixel into a larger block`. Lab 1 also uses expected-output comments and f-strings:

```python
print(f"2D zero array\n{zeroArray}")
```

Lab 3 uses two-space indentation in the nearest-neighbor and shrinking examples and four spaces in bilinear interpolation. Lab 4 uses four spaces. Keep indentation consistent within each function or code block. The nearest-neighbor example below uses two spaces to match its reference section.

## 4. Use the same image-operation patterns

Each block below is an example to use where relevant, not a requirement to include every operation in every task.

### Read, display, and inspect

```python
image = ski.data.astronaut()
ski.io.imshow(image)
```

Then inspect the array in a separate cell:

```python
image.shape
```

Other normal inspection cells are `type(image)`, `image.dtype`, or the array variable itself. Lab 2 also reads external images using `ski.io.imread(...)` and `as_gray=True`. Use the task's supplied image when one is specified; built-in examples are useful for demonstrations.

For color images, the array dimensions are `(height, width, channels)`. Gray images have `(height, width)`, so code using `image.shape[2]` assumes a color image.

### Copy, then change pixels

```python
newImage = image.copy()
newImage[0:100] = (255,0,0)
ski.io.imshow(newImage)
```

For an RGB image, channel 0 is red, channel 1 is green, and channel 2 is blue. Index rows first, columns second:

```python
cropped_section = image[100:300, 100:300]
ski.io.imshow(cropped_section)
```

A basic slice may share memory with the original array. Use `.copy()` on the crop if you will modify it independently. Assignment such as `newImage = image` does not create a separate image.

### Keep one color channel

```python
red = image.copy()
red[:,:,1:3] = 0
ski.io.imshow(red)
```

This retains a three-channel image with only red values. `image[:,:,0]` by itself is a two-dimensional array of channel intensities.

Lab 2 makes separate red, green, and blue copies, zeroes the other channels, and adds the three results to reconstruct the original. That works because only one component contributes at each channel position.

### Create a canvas

For an 8-bit color output, Lab 3 uses:

```python
height = image.shape[0]
width = image.shape[1]
Canvas = np.zeros((height,width,3), dtype=np.uint8)
```

Do not make `uint8` a universal requirement. Lab 2 also uses floating-point gray canvases and random images. Choose the dtype and intensity range appropriate to the operation: 8-bit images use 0-255, while the random float images in Lab 2 use values from 0 to below 1. Keep fractional intermediate calculations until conversion is appropriate.

### Implement algorithms step by step

The nearest-neighbor section in Lab 3 follows this progression:

1. Load and display an image; inspect its shape.
2. Demonstrate horizontal enlargement with fixed dimensions and nested loops.
3. Generalize the operation into a function with width and height scale parameters.
4. Call the function, inspect the output shape, and display the result.
5. Apply the same function to a cropped region to demonstrate zooming.

Use this progression when it helps explain a reusable algorithm. It is not mandatory for a simple crop or channel edit. The lab's shrinking example stays as a loop without a generalized function, and bilinear interpolation uses a scalar example followed by an image function.

For shrinking, the lab loops through output rows, columns, and channels, takes a 2-by-2 input block, and assigns its mean to one output pixel. For bilinear interpolation, it maps each output position to the input, finds four neighboring pixels, and combines them using fractional weights. Keep these steps explicit when implementing similar tasks.

## 5. A complete example of the cell layout

This is a small Lab 3-style task you can adapt. Each numbered item is a separate notebook cell; the cell labels belong in this guide, not in the notebook source. This example assumes an RGB `uint8` image and positive integer scale factors.

**Cell 1 - code: imports**

```python
import numpy as np
import skimage as ski
```

**Cell 2 - markdown**

```markdown
# Nearest Neighbor Image Interpolation
```

**Cell 3 - code: read and display**

```python
image = ski.data.astronaut()
ski.io.imshow(image)
```

**Cell 4 - code: inspect**

```python
image.shape
```

**Cell 5 - markdown**

```markdown
## Increase image width

To double the width, copy each pixel into two adjacent columns.
```

**Cell 6 - code: fixed-size demonstration for this 512-by-512 image**

```python
scaledImageCanva = np.zeros((512,1024,3), dtype=np.uint8)
for i in range(image.shape[0]):
  for j in range(image.shape[1]):
    scaledImageCanva[i, j*2:(j*2)+2] = image[i,j]
```

**Cell 7 - code: display**

```python
ski.io.imshow(scaledImageCanva)
```

**Cell 8 - markdown**

```markdown
## Scale width and height

Use a function so we can choose the width and height scale factors.
```

**Cell 9 - code: generalize and call**

```python
def nearest_neighbor_upscale(image, width_scale, height_scale):
  height = image.shape[0]
  width = image.shape[1]
  channels = image.shape[2]
  scaledImageCanva = np.zeros((height*height_scale,width*width_scale,channels), dtype=np.uint8)
  for i in range(height):
    for j in range(width):
      scaledImageCanva[i*height_scale:(i+1)*height_scale, j*width_scale:(j+1)*width_scale] = image[i,j]
  return scaledImageCanva
#--------------------------------------------------------------------#
width_scale = 3
height_scale = 3
scaledImage = nearest_neighbor_upscale(image,width_scale,height_scale)
```

**Cell 10 - code: inspect**

```python
scaledImage.shape
```

The expected shape is `(1536, 1536, 3)`.

**Cell 11 - code: display the function result**

```python
ski.io.imshow(scaledImage)
```

**Cell 12 - markdown**

```markdown
## Zooming (NNeighbor)

Crop a section of the image, then enlarge it using the same function.
```

**Cell 13 - code: apply to a crop**

```python
cropped_section = image[0:200, 100:300]
zoomed_image = nearest_neighbor_upscale(cropped_section, 2, 2)
ski.io.imshow(zoomed_image)
```

The cropped region is `(200, 200, 3)` and the zoomed result is `(400, 400, 3)`.

For a Lab 2-style task, reuse the shorter sequence of heading, explanation, image operation, and display. Add a function only when the task benefits from it.

## 6. Follow Lab 4 for histogram tasks

### Section structure and plotting

Lab 4 starts with a topic overview and a list of its three algorithms, then defines `plot_img_and_hist(image, title, bins=256)` once. Each algorithm section follows this pattern:

```text
Markdown: numbered algorithm heading; concept or formula when useful
Code:     prepare the input and show its image and histogram
Code:     define and call the manual implementation; show the result
Code:     call the scikit-image implementation; show the result
```

In the original notebook, the setup, manual code, and library comparison for one algorithm share a single code cell, separated by numbered comment banners. Keeping them together matches Lab 4; splitting them at those boundaries also preserves the teaching sequence.

The plotting helper uses `plt.subplots(1, 2, figsize=(12, 4))`: the image is on the left and its histogram is on the right. Gray images use `cmap='gray'` with limits 0 and 255; RGB images get separate red, green, and blue histogram traces. It uses `.ravel()` to supply pixel values to the histogram, labels the plots, calls `plt.tight_layout()`, and finishes with `plt.show()`.

Reuse one plotting helper when comparing several outputs. Keep the intensity scale consistent so the pictures and histograms can be compared fairly. For comparable 8-bit histogram bins, explicitly use `range=(0, 256)` in the helper's `axes[1].hist(...)` calls; setting the x-axis limits alone does not fix the bin edges.

### Contrast stretching

The lab creates a low-contrast gray image from `data.camera()` by compressing its values to 100-150. It then:

1. Finds `img_min` and `img_max`.
2. Converts to float before calculating the new intensities.
3. Applies a linear mapping to the requested output range.
4. Converts the result to `uint8`, displays it, and prints the resulting minimum and maximum.
5. Compares with `exposure.rescale_intensity(img_low_contrast, out_range=(0, 255))`.

For a general output range, the corrected formula is:

```text
output = (input - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
```

The lab multiplies by `new_max` instead of `new_max - new_min`. Its demonstrated 0-255 case works, but use the corrected expression for a nonzero lower bound. A constant input has `old_max == old_min`; define a simple policy, such as filling the output with `new_min`, before dividing.

### Histogram equalization

The manual function in cell 6 shows the full calculation:

1. Count 256 intensity bins using `np.histogram(img.flatten(), 256, [0, 256])`.
2. Divide counts by the number of pixels to obtain `pdf`.
3. Use `pdf.cumsum()` to obtain `cdf`.
4. Multiply the CDF by 255 and cast it to `uint8` to create `map_to_s`.
5. Apply the lookup table with `map_to_s[img]`.

Keep those intermediate variables visible rather than compressing everything into one expression. This example assumes a nonempty, 8-bit gray image whose pixel values can index the 256-entry table.

The comparison in the notebook is:

```python
img_eq_skimage = exposure.equalize_hist(img_unequalized)
img_eq_skimage = (img_eq_skimage * 255).astype(np.uint8)
plot_img_and_hist(img_eq_skimage, "Skimage Histogram Equalization")
```

This cell depends on the earlier imports, helper, and `img_unequalized` setup. The explicit conversion puts the library result on the helper's 0-255 display scale. Describe the manual method as the lab's direct CDF mapping; it does not subtract the first nonzero CDF value.

### Histogram matching

Lab 4 uses two images: a source to modify and a reference whose intensity distribution is the target. Its manual implementation:

1. Defines `calculate_cdf(channel)` using a histogram, cumulative sum, and normalization.
2. Creates `matched = np.zeros_like(src)`.
3. Loops over the RGB channels, calculating source and reference CDFs separately.
4. Builds a 256-entry `uint8` lookup table.
5. For each source intensity, uses `np.argmin(np.abs(ref_cdf - src_cdf[src_pixel_val]))` to find a reference intensity with the closest CDF value.
6. Applies the table with `mapping[src_channel]` and stores the result in the corresponding output channel.

Then it compares with `exposure.match_histograms(src_img, ref_img, channel_axis=-1)`. Preserve the distinction between source and reference in names and plot titles. The manual lookup method assumes nonempty 8-bit RGB arrays; different image dimensions are acceptable because it compares distributions, not corresponding pixels.

Compare the resulting images and histograms without requiring exact pixel equality between manual and library methods. Use source and reference images whose appearance you have actually checked; the notebook's cinematic descriptions are labels, not proof of what a remote image contains.

## 7. Copy the style while correcting mistakes

Cell numbers below are one-based positions in the notebook, including markdown cells. They are not execution counts.

| Location | Detail to correct or account for |
| --- | --- |
| Lab 1, cells 3-8 | The value/reference explanation is simplified. Python assignment binds a name to an object; rebinding an integer differs from mutating a shared list. Do not explain assignment as always copying data. |
| Lab 1, cells 59-61 | `listArray = [...]` creates a Python list despite its print label. Use `np.array(...)` to create a NumPy array. |
| Lab 1, cells 63-64 | The heading says Int, but the example specifies `np.float16`. Make headings and dtype descriptions agree with the code. |
| Lab 2, cell 11 | Use the property `image.dtype`, not `image.dtype()`. |
| Lab 2, cells 18, 20, and 62 | Adding 50 directly to `uint8` can wrap values above 255. For brightness with saturation, convert to a wider type before addition, clip, and convert back. |
| Lab 2, cell 31 | `widht` should be `width`. The dimensions shown draw a rectangle, despite the square comment. Saved output does not prove the current source runs from a fresh kernel. |
| Lab 2, cells 58-60 | `cv2.imshow("img.png")` has a saved Colab error. The working sequence reads the image with `cv2.imread(...)` and then calls `cv2_imshow(img)`. |
| Lab 2, OpenCV section | OpenCV color images use BGR ordering, so the RGB channel meanings from the skimage section must be adjusted. |
| Lab 3, cell 9 | It displays the earlier `scaledImageCanva`; display `scaledImage` to show the function result inspected in cell 8. |
| Lab 3, cells 18-19 | Two different functions are named `bilinear_interpolation`; the second replaces the first. Use distinct names if both examples must remain callable. |
| Labs 2-3, saved outputs | There are deprecation warnings for `ski.io.imshow`. It is the reference display style; if the task environment no longer supports it, use a small Matplotlib display cell and keep the same notebook structure. |
| Lab 4, cells 3-4 | General contrast stretching needs the factor `new_max - new_min`, and constant images need handling before division by `img_max - img_min`. |
| Lab 4, cell 2 | The plotting helper assumes gray intensities on a 0-255 scale. Adapt its limits for float images in 0-1, and use fixed histogram bin ranges when comparing distributions. |
| Lab 4, cells 6 and 8 | The manual lookup tables assume integer intensities in 0-255. The equalization example is gray; the matching example expects color channels. Check those assumptions before reusing them. |

For ordinary brightness increase on an 8-bit image, a corrected small example is:

```python
BrightnessIncrease = image.astype(np.int32) + 50
BrightnessIncrease = np.clip(BrightnessIncrease, 0, 255).astype(np.uint8)
ski.io.imshow(BrightnessIncrease)
```

Casting a float image in the range 0-1 straight to `uint8` does not rescale it to 0-255. Check the range before copying a conversion pattern. Also state an algorithm's assumptions: the Lab 3 functions expect color images, nearest-neighbor block replication expects positive integer scales, and the fixed shrinking example assumes a 512-by-512 input.

Correct spelling in new headings and variables. Preserve the simple teaching style without reproducing typos, stale outputs, or incorrect labels.

## 8. Steps to follow for every new task

1. **Read the task requirements.** Identify the required images, operations, outputs, and whether the algorithm must be manual.
2. **Choose the closest lab example.** Use Lab 1 for array basics, Lab 2 for direct image manipulation, Lab 3 for loop-based resizing, and Lab 4 for histogram algorithms and manual/library comparisons.
3. **Create the notebook cells.** Use a short title, relevant imports, and small sections matching the requested operations.
4. **Load and inspect the input.** Show it and check its shape, dtype, and relevant intensity range.
5. **Implement one operation at a time.** Use slices and arithmetic for simple changes; allocate and fill an output array when the algorithm requires it.
6. **Explain the important detail.** Add a short note about the slice, copy, formula, or coordinate mapping the reader needs to understand.
7. **Show intermediate and final results.** Inspect dimensions when resizing; display the variable produced by the current step. For histogram tasks, show the image and its distribution using the shared plotting helper and compare manual/library outputs where requested.
8. **Generalize when useful.** Add a small function and a clear call when parameters or repeated use justify it.
9. **Restart the kernel and run all cells in order.** Fix missing variables, path problems, overwritten function names, and stale outputs. Save the resulting notebook.
10. **Use the required submission name and location.** `Tasks_in_Lab/` exists in this project, but its single empty file is not enough evidence to infer a mandatory student-ID naming convention.

### Final checklist

- [ ] The notebook solves the actual assignment requirements.
- [ ] Its structure resembles the relevant Lab 1, 2, 3, or 4 examples.
- [ ] Headings and explanations are short and useful.
- [ ] Imports use the familiar aliases and appear before use.
- [ ] Each cell has a clear purpose; algorithms are visible and easy to follow.
- [ ] Images are copied when independent edits are needed.
- [ ] Rows, columns, channels, dtypes, and value ranges are handled correctly.
- [ ] The current operation's result is inspected or displayed.
- [ ] Function assumptions match the input and parameters.
- [ ] Histogram tasks use suitable intensity ranges and clearly labeled source, reference, manual, and library results where applicable.
- [ ] A fresh top-to-bottom run succeeds and the outputs are saved.

### Instruction to reuse for future tasks

> Follow tutorial_1.0 in tutorial.md. Use Lab_1.ipynb, Lab2_CV.ipynb, Lab_3_CV.ipynb, and lab4_CV.ipynb as the style references. Match the closest lab's cell structure, explanations, naming, NumPy operations, and visible intermediate results. Use loops and output canvases for resizing algorithms; use explicit histogram, CDF, and lookup-table steps for histogram algorithms. For Lab 4-style tasks, use a shared image-and-histogram plotting helper and manual/library comparisons. Follow the assignment requirements, correct errors in the reference examples, and make sure the notebook runs from top to bottom.
