# Computer Vision Labs 1-4: Complete Study Guide

This guide explains the concepts, code, calculations, and results in all four course notebooks. Read it alongside the notebooks: [Lab 1](Lab_1.ipynb), [Lab 2](Lab2_CV.ipynb), [Lab 3](Lab_3_CV.ipynb), and [Lab 4](lab4_CV.ipynb).

[tutorial.md](tutorial.md) explains how to write future tasks in the labs' style. This document explains what the labs teach and why their code works. Examples below follow the notebooks, with corrections explicitly identified. Cell references count both markdown and code cells, starting at 1.

## Contents

- [Before starting: notebooks, arrays, and images](#before-starting-notebooks-arrays-and-images)
- [Lab 1: Python and NumPy](#lab-1-python-and-numpy)
- [Lab 2: Images as NumPy arrays](#lab-2-images-as-numpy-arrays)
- [Lab 3: Resizing and interpolation](#lab-3-resizing-and-interpolation)
- [Lab 4: Histograms and intensity transformations](#lab-4-histograms-and-intensity-transformations)
- [Connections and revision questions](#connections-and-revision-questions)
- [Notebook coverage map](#notebook-coverage-map)
- [Cell structure to copy for new tasks](#cell-structure-to-copy-for-new-tasks)

## Before starting: notebooks, arrays, and images

A notebook has two main cell types. Markdown cells explain ideas; code cells execute Python. Variables and imports stay in the running kernel, so later cells can use earlier definitions. A saved output can remain visible after its source has changed: always restart the kernel and run cells in order when checking correctness.

In a code cell, `print(value)` explicitly prints a value. A final expression such as `image.shape` is displayed automatically by the notebook. `#` begins a Python comment, while a heading such as `# NumPy` belongs in a markdown cell.

The labs use these libraries:

| Library | Role |
| --- | --- |
| NumPy, imported as `np` | Arrays, indexing, arithmetic, random numbers, histograms |
| scikit-image, imported as `ski` or via direct imports | Example images, reading/displaying images, reference processing functions |
| OpenCV, imported as `cv2` | Another image-processing library, used for reading and displaying in Lab 2 |
| Matplotlib, imported as `plt` | Image and histogram plots in Lab 4 |

An image is represented by a grid of pixel values:

| Image type | Example shape | Meaning |
| --- | --- | --- |
| Gray | `(512, 512)` | 512 rows and 512 columns; one intensity per pixel |
| RGB | `(512, 512, 3)` | 512 rows, 512 columns, three color components per pixel |

For an RGB image, `image[row, column]` is a three-value vector such as `[255, 0, 0]`, representing red. `image[row, column, channel]` is a single component. The first coordinate is vertical position, the second is horizontal position.

The common 8-bit type, `uint8`, stores integers from 0 through 255. For gray images, 0 is black and 255 is white. The float images used in some examples instead use 0.0 through 1.0. Shape describes layout; dtype describes storage; the actual values describe brightness and color. These are separate properties.

## Lab 1: Python and NumPy

### 1.1 Variables, assignment, and shared objects

The notebook introduces “value types” and “reference types” to explain why numbers and lists behave differently. A more accurate Python explanation is that **assignment binds a name to an object**.

```python
x = 10
y = x
y = 20
print(f"X: {x}, Y: {y}")
```

Initially, `x` and `y` refer to the integer object representing 10. `y = 20` rebinds `y`; it does not change the object referenced by `x`. The output is `X: 10, Y: 20`.

Compare that with:

```python
List_1 = [1, 2, 3]
List_2 = List_1
List_2.append(5)
print(List_1)
print(List_2)
```

Both names refer to the same list. `.append(5)` modifies that list, so both prints show `[1, 2, 3, 5]`. No independent list was created.

This explains the later use of `image.copy()`: a second name for an array is not a second array.

### 1.2 Mutable, immutable, and `id`

A mutable object can change after creation; an immutable object cannot.

| Type | Mutable? | Meaning |
| --- | --- | --- |
| `int`, `float`, `bool`, `str` | No | An apparent update normally rebinds the variable |
| `list`, `dict`, `set` | Yes | The object's contents can be modified |
| `tuple` | No | Its sequence of object references cannot be changed |
| NumPy array | Yes | Its stored elements can be changed |

`id(x)` identifies the object referenced by `x` during its lifetime. Its numeric value is not something to memorize or expect to reproduce.

With `x = 1; y = x; x = 2`, `x` now refers to a different object, while `y` still refers to 1. With a shared list, `x[0] = 100` changes the existing object, so both names still share its identity and see the new value.

**Correction:** rebinding to an integer does not necessarily allocate a brand-new object; Python may reuse existing immutable objects. The key difference is rebinding a name versus mutating an object. Also, a tuple can contain a mutable object, such as a list, whose internal contents can still change.

### 1.3 Python data types

The notebook demonstrates these common types:

| Type | Example | Purpose |
| --- | --- | --- |
| Integer | `5` | Whole numbers |
| Float | `5.3` | Approximate real-number calculations |
| Boolean | `True`, `False` | Conditions and logical values |
| String | `"Hello"` | Text |
| List | `[1, "hello"]` | Ordered collection; may mix types |
| Set | `{1, 2, 3}` | Collection of unique elements |
| Tuple | `(1, 2, 3)` | Ordered immutable sequence |
| Dictionary | `{'a': 1}` | Key-value lookup |
| None | `None` | Absence of a value |

Python has more types than this introductory list, and all these values are objects. A tuple is not restricted to pairs.

The constructor forms shown in the lab have specific default values:

```python
print(int())    # 0
print(float())  # 0.0
print(str())    # empty string
```

Therefore, `x = 5` and `x = int()` do not assign the same value. In the notebook, the constructor lines illustrate another way to create an object of that type, not an equivalent value assignment.

`type(x)` reports its type. For a NumPy array, use `.dtype` to inspect the element type instead.

### 1.4 Conditions

An `if` statement executes a block when its condition is true. `elif` tests another condition if the earlier one failed; `else` handles the remaining case.

```python
if False:
    print("Hello world 1")
elif True:
    print("Hello world 2")
else:
    print("Hello world 3")
```

Only `Hello world 2` is printed. Within one `if`/`elif`/`else` chain, the first matching branch runs. Separate `if` statements are independent. Indentation defines which lines belong to each block.

### 1.5 Loops

```python
for i in range(5):
    print(i, end="-")
```

`range(5)` supplies 0, 1, 2, 3, and 4. The upper bound is excluded. `end="-"` replaces the normal newline after each print, so the actual output is `0-1-2-3-4-`, including the final dash.

This becomes important in image processing: `range(image.shape[0])` visits every valid row index.

### 1.6 Functions

```python
def greet():
    print("Hello, world!")

greet()
```

`def` defines the function; `greet()` calls it. The body does not run merely because the definition was executed. This function prints a message and implicitly returns `None`.

In later labs, functions receive parameters such as an image and scale factors, then `return` a computed array. Printing and returning are different: returning allows another expression to use the result.

### 1.7 NumPy installation and import

The notebook includes a shell command in a code cell:

```text
!pip install numpy
```

The leading `!` is notebook/IPython syntax for running a shell command, not ordinary Python syntax. Installation makes a package available to the environment. Importing makes it available in the current Python session:

```python
import numpy as np
```

`np` is an alias, so `np.zeros(...)` calls NumPy's `zeros` function.

### 1.8 Creating arrays

**Regularly spaced values:**

```python
arangeArray = np.arange(5, 10, 1.001, dtype=np.float32)
```

The arguments mean start at 5, stop before 10, and step by approximately 1.001. The result is approximately `[5, 6.001, 7.002, 8.003, 9.004]`. Floating-point representation can introduce small rounding differences.

**Zero arrays:**

```python
oneD = np.zeros(3)
twoD = np.zeros((3, 4), dtype=np.float16)
threeD = np.zeros((3, 4, 3))
```

The shapes are `(3,)`, `(3, 4)`, and `(3, 4, 3)`. A shape tuple gives the size of each axis; the 3D example has 36 elements. Without a dtype argument, these zero arrays use floating-point storage. A zero array later serves as a blank image canvas.

**Random floating-point values:**

```python
randomArray = np.random.rand(3, 4)
```

This creates 12 random values in `[0, 1)`: zero is included, one is excluded. Re-execution normally produces different values.

**Random integers:**

```python
randomArrayInt = np.random.randint(0, 256, (3, 4))
```

The result has shape `(3, 4)` with values from 0 through 255. Here too, the upper bound is excluded. Those values fit in 8 bits, but the function call does not explicitly request `uint8` storage.

**From a list:**

```python
listArray = [1, 2, 3, 4, 5]
int_array = np.array(listArray, dtype=np.int32)
```

`listArray` is a Python list; `int_array` is a NumPy array. The lab's first “numpy array from list” print label is misleading until `np.array(...)` is actually used.

### 1.9 Array dtypes

| Dtype | Interpretation | Example use in the labs |
| --- | --- | --- |
| `np.int32` | Signed 32-bit integers | Integer array construction |
| `np.uint8` | Unsigned 8-bit integers, 0-255 | Image output canvases |
| `np.uint16` | Unsigned 16-bit integers, 0-65535 | Stores the lab's value 300 |
| `np.float16` | 16-bit floating point | Small float arrays |
| `np.float32` | 32-bit floating point | `arange` example |
| `np.float64` | 64-bit floating point | Float example |
| `np.complex64` | Complex numbers with two 32-bit float components | Values such as `3 + 4j` |

Signed integers allow negative values; unsigned integers do not. Floating-point types allow fractional values but have finite precision. In Python, `j` denotes the imaginary unit.

The notebook's “Int” example actually specifies `np.float16`, so its values are floats. Trust the dtype argument and `.dtype` output, not the heading or variable name.

An f-string such as `f"Type: {int_array.dtype}"` inserts an expression into text. `\n` adds a newline. These are display conveniences, not array operations.

### 1.10 Indexing and slicing

The lab uses:

```python
nuArray = np.array([
    [100, 120, 150],
    [200, 210, 250],
    [300, 320, 350]
])
```

Its shape is `(3, 3)`. Indices begin at zero:

| Expression | Result | Explanation |
| --- | --- | --- |
| `nuArray[1]` | `[200, 210, 250]` | Second row |
| `nuArray[2, 1]` | `320` | Third row, second column |
| `nuArray[1, 1:3]` | `[210, 250]` | Second row, columns 1 and 2 |
| `nuArray[1, :]` | `[200, 210, 250]` | Second row, all columns |
| `nuArray[:, 0]` | `[100, 200, 300]` | All rows, first column |

`start:stop` includes the start and excludes the stop. A bare `:` selects the entire axis. This same notation selects image regions and channels in Lab 2.

### 1.11 Arithmetic and Boolean masks

```python
newArr = nuArray + 20
newArr = nuArray * 20
```

A scalar operation is applied element by element. For example, the first row becomes `[120, 140, 170]` after addition, or `[2000, 2400, 3000]` after multiplication. These expressions create results; assigning them to `newArr` does not change `nuArray`. The second assignment replaces the first value of `newArr`.

```python
h = nuArray > 150
nuArray[h] = 0
```

The comparison produces a Boolean array:

```text
[[False, False, False],
 [ True,  True,  True],
 [ True,  True,  True]]
```

Assignment through this mask changes only positions marked `True`. The resulting array is:

```text
[[100, 120, 150],
 [  0,   0,   0],
 [  0,   0,   0]]
```

The value 150 remains because the condition is strictly greater than 150. This is the foundation for threshold-based pixel manipulation.

## Lab 2: Images as NumPy arrays

### 2.1 Reading color and gray images

The notebook uses:

```python
import numpy as np
import skimage as ski
```

`ski.io.imread(...)` reads image data from the supplied path or URL. The raw-string prefix `r"..."` prevents backslash escapes, which is particularly useful for Windows paths; it does not change how an ordinary HTTPS URL is downloaded.

The notebook's second read adds `as_gray=True` to request a gray image. For the RGB input in this example, conversion reduces the color components to a weighted intensity and produces a 2D floating-point image. Do not assume every image reader always returns the same dtype or value range.

For a built-in RGB example used elsewhere in the labs:

```python
image = ski.data.astronaut()
ski.io.imshow(image)
```

`imread` loads the array; `imshow` displays an array already in memory. The saved notebooks contain warnings about `ski.io.imshow` being deprecated. Lab 4 provides a Matplotlib display pattern if the environment does not support that function.

### 2.2 Image properties and conversion

| Expression | What it tells you or does |
| --- | --- |
| `type(image)` | The container type, normally `numpy.ndarray` |
| `image.shape` | Array dimensions |
| `image.dtype` | Element storage type; no parentheses |
| `image.flat` | Iterator over elements in flattened order |
| `image.flatten()` | A 1D copy of the elements |
| `image.ravel()` | A 1D array, sharing storage when possible |
| `image.reshape(...)` | Changes array layout without changing element count; does not resize the image by interpolation |
| `image.astype(np.uint8)` | Converts the values to the requested storage type |

**Correction:** the lab's prose lists `.dtype()`, but `.dtype` is a property. Also, casting does not automatically normalize values: converting 0.5 directly to an integer gives 0, not 128. A float image known to be in 0-1 can be scaled by 255 before conversion.

### 2.3 Brightness and overflow

The original brightness example is:

```python
BrightnessIncrease = image + 50
```

Its intention is to increase every component by 50. However, `uint8` cannot represent values above 255. In this operation, values wrap: 240 + 50 becomes 34 rather than a brighter value.

For saturated brightness increase on an 8-bit image, use:

```python
BrightnessIncrease = image.astype(np.int32) + 50
BrightnessIncrease = np.clip(BrightnessIncrease, 0, 255).astype(np.uint8)
ski.io.imshow(BrightnessIncrease)
```

The wider dtype holds intermediate values such as 290. `np.clip` limits them to 0-255; only then is the result converted back to `uint8`. Casting after the addition would be too late if overflow had already occurred.

### 2.4 Changing only one channel

The lab copies an image before increasing its red channel:

```python
BrightnessIncrease = image.copy()
BrightnessIncrease[:,:,0] += 50
```

The two colons select every row and column; channel 0 selects red for RGB data. This retains all three channels but still has the overflow issue above.

A corrected version is:

```python
BrightnessIncrease = image.copy()
red_values = image[:,:,0].astype(np.int32) + 50
BrightnessIncrease[:,:,0] = np.clip(red_values, 0, 255).astype(np.uint8)
```

Assigning into `BrightnessIncrease[:,:,0]` replaces one component of the existing 3D image. Assigning `BrightnessIncrease = image[:,:,0] + 50` instead replaces the variable with a 2D intensity array. A display function may render that array in grayscale or a colormap; it no longer contains an RGB triplet at each pixel.

### 2.5 Cropping

```python
row_crop = image[100:300]
column_crop = image[:,100:300]
region = image[100:300, 100:300]
```

The first keeps 200 rows and all columns. The second keeps all rows and 200 columns. The third keeps a 200-by-200 region, assuming the input is large enough.

Cropping selects existing pixels; it does not interpolate them or change their values. Basic NumPy slices can share memory with the source, so use `region = image[100:300, 100:300].copy()` before independently editing a crop.

### 2.6 Random noise images

```python
mat = np.random.rand(512, 512)
matRGB = np.random.rand(512, 512, 3)
```

`mat` gives one random intensity per pixel; `matRGB` gives three random components. Both use float values in `[0, 1)`. They look like noise because neighboring pixels have unrelated random values. These calls create noise images; they do not add noise to an existing photograph.

### 2.7 Drawing a rectangle on a canvas

The corrected version of the lab's example is:

```python
Canvas = np.zeros((512, 512))
x = 100
y = 200
width = 100
height = 200
Canvas[y:y+height, x:x+width] = 1
ski.io.imshow(Canvas)
```

The float canvas begins as black. Rows 200-399 and columns 100-199 become white. The assignment broadcasts the scalar value 1 to every selected pixel.

Although position is named `(x, y)`, array indexing is `[y, x]`: row first, column second. The original variable `widht` is misspelled, and a 100-by-200 region is a rectangle, not a square.

### 2.8 Drawing colored bands

Lab 2 loads another RGB image into `NatrureImage` and shows its saved shape `(350, 625, 3)`. The spelling is just a variable name.

```python
image = NatrureImage.copy()
image[0:100] = (255, 0, 0)
image[:,:200] = (0, 255, 0)
```

The first operation paints the top 100 rows red; the second paints the left 200 columns green. NumPy repeats each RGB triplet across the selected region. The overlapping top-left area becomes green because the second assignment runs last.

These operations overwrite pixels. They do not blend transparent lines with the original photograph.

### 2.9 Drawing a centered plus sign

```python
imCamera = ski.data.camera()
height = imCamera.shape[0]
width = imCamera.shape[1]
midHeight = int(height / 2)
midWidth = int(width / 2)
imCamera[midHeight-50:midHeight+50] = 0
imCamera[:,midWidth-50:midWidth+50] = 0
```

The camera image is gray. For its 512-by-512 shape, both midpoint indices are 256. The horizontal black band occupies rows 206-305; the vertical black band occupies columns 206-305. Their union forms a plus sign with 100-pixel-wide strokes spanning the image.

The notebook displays the image before and after each assignment so you can see the construction. Using dimensions rather than hard-coding 256 makes the midpoint calculation adapt to other sizes, although the fixed stroke width still assumes a sufficiently large image.

### 2.10 Splitting and recombining RGB

```python
red = NatrureImage.copy()
green = NatrureImage.copy()
blue = NatrureImage.copy()

red[:,:,1:3] = 0
green[:,:,0] = 0
green[:,:,2] = 0
blue[:,:,0:2] = 0

n = red + green + blue
ski.io.imshow(n)
```

Each result still has three channels. For a pixel originally `[40, 120, 200]`, the copies contain `[40, 0, 0]`, `[0, 120, 0]`, and `[0, 0, 200]`. Adding them restores `[40, 120, 200]`.

There is no overflow here because only one image contributes a nonzero value at each channel position. This does not imply that adding arbitrary RGB images is safe.

The notebook first allocates `n = np.zeros(...)` and then reassigns `n` to the sum. That allocation is unnecessary for this particular expression because the sum itself creates the result.

### 2.11 OpenCV and Google Colab

The final section introduces:

```python
import cv2
from google.colab.patches import cv2_imshow
```

The notebook uses `!curl -o img.png URL` to download an image into a local file. The `-o` option selects the output filename; the exclamation mark runs the command from a notebook cell.

After downloading in Colab:

```python
img = cv2.imread(r"/content/img.png")
cv2_imshow(img)
```

`/content/` is a Colab path. Use the actual local path outside Colab. If reading fails, `cv2.imread` can return `None`, so check the path before trying to display or modify the result.

The original `cv2.imshow("img.png")` cell is incorrect: ordinary OpenCV `imshow` expects a window name and image data, and Colab disables that desktop-window function. The notebook's working display uses `cv2_imshow(img)`.

OpenCV normally reads color images in **BGR** order; scikit-image uses **RGB** in these examples. Thus OpenCV channel 0 is blue, not red. Colab's OpenCV display helper handles that convention. When using Matplotlib to show an OpenCV color image, convert BGR to RGB first.

The OpenCV array is still a NumPy array. Its `img + 50` example has the same `uint8` overflow concern as the earlier brightness calculation.

## Lab 3: Resizing and interpolation

### 3.1 What interpolation means

Resizing changes how many pixels represent an image. A larger image needs additional pixel values; interpolation specifies how to choose them from the existing image. It does not recover new detail that was never recorded.

Lab 3 demonstrates three operations:

| Operation | Rule | Typical effect |
| --- | --- | --- |
| Nearest-neighbor enlargement | Repeat an existing pixel value | Sharp block boundaries; pixelated appearance |
| Block-average shrinking | Average a group of input pixels | Smaller image with local smoothing |
| Bilinear resizing | Weighted combination of four neighboring pixels | Smoother transitions |

The examples use RGB `uint8` images. Their access to `image.shape[2]` means they do not directly support a 2D gray image.

### 3.2 Fixed horizontal enlargement

`ski.data.astronaut()` gives the lab a `(512, 512, 3)` image. It first doubles only the width:

```python
scaledImageCanva = np.zeros((512, 1024, 3), dtype=np.uint8)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        scaledImageCanva[i, j*2:(j*2)+2] = image[i,j]
```

The outer loop visits rows; the inner loop visits columns. Input pixel `(i, j)` is copied to output columns `2*j` and `2*j + 1` in row `i`. Selecting no explicit channel index copies all three components.

For one row:

```text
Input:   A B C
Output:  A A B B C C
```

For `j = 3`, the destination slice is `6:8`, which contains columns 6 and 7. NumPy broadcasts the source RGB vector into both destination pixels. The output height is unchanged, so the image becomes wider without becoming taller.

### 3.3 General nearest-neighbor enlargement

The reusable form accepts positive integer width and height multipliers:

```python
def nearest_neighbor_upscale(image, width_scale, height_scale):
    height, width, channels = image.shape
    scaledImageCanva = np.zeros(
        (height*height_scale, width*width_scale, channels),
        dtype=np.uint8
    )
    for i in range(height):
        for j in range(width):
            scaledImageCanva[
                i*height_scale:(i+1)*height_scale,
                j*width_scale:(j+1)*width_scale
            ] = image[i,j]
    return scaledImageCanva
```

This is the lab's algorithm with the shape calculation split into readable lines. Each source pixel fills a rectangle with `height_scale` rows and `width_scale` columns.

```text
Input:          Both scales = 2:
A B             A A B B
C D             A A B B
                C C D D
                C C D D
```

For the astronaut image and both scales equal to 3, the output shape is `(1536, 1536, 3)`. The number of pixels increases by 9, while the channel count remains 3.

The parameter order is `(image, width_scale, height_scale)`, but the output shape is `(height, width, channels)`. Keep that distinction clear.

**Correction:** the notebook inspects `scaledImage.shape` and then displays the old `scaledImageCanva`. Display `scaledImage` to see the function's output. This block-replication function is for integer enlargement, not arbitrary fractional resizing.

### 3.4 Zooming into a region

```python
cropped_section = image[0:200, 100:300]
zoomed_image = nearest_neighbor_upscale(cropped_section, 2, 2)
ski.io.imshow(zoomed_image)
```

The crop selects the top 200 rows and columns 100-299, giving `(200, 200, 3)`. Doubling each spatial dimension produces `(400, 400, 3)`.

Zooming here means crop first, then enlarge the selected region. The crop discards the surroundings; enlargement repeats the retained pixels. The original variable spelling `croped_section` is corrected above.

### 3.5 Shrinking with block means

The notebook reduces a 512-by-512 image to 256-by-256:

```python
shrinked = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(shrinked.shape[0]):
    for j in range(shrinked.shape[1]):
        for k in range(shrinked.shape[2]):
            tempArray = image[i*2:(i*2)+2, j*2:(j*2)+2, k]
            shrinked[i,j,k] = tempArray.mean()
```

Unlike enlargement, the loops visit output positions. Each output component is the mean of a 2-by-2 input block in the same channel:

```text
Input channel block:
10  20
30  40

Output component = (10 + 20 + 30 + 40) / 4 = 25
```

The `k` loop handles red, green, and blue separately. `.mean()` produces a fractional result when needed; storing it in the `uint8` output discards the fractional part.

This is block averaging, not nearest-neighbor shrinking. Averaging smooths local variation before reducing the number of pixels, which can reduce aliasing: misleading patterns caused by sampling fine detail too sparsely. A 2-by-2 average does not eliminate every possible artifact.

The fixed output shape and block size assume the lab's input dimensions. For other sizes, define how to handle incomplete edge blocks. The standalone `512/2` cell illustrates the target size, but Python `/` returns `256.0`; array dimensions need integers.

### 3.6 Bilinear interpolation at a point

The first bilinear function receives four corner samples `(x, y, value)` and a point inside their rectangle. It sorts the samples, unpacks their coordinates, checks that they form a rectangle, and checks that the requested point lies inside it.

Let the corners be:

```text
(x1, y1): q11       (x2, y1): q21
(x1, y2): q12       (x2, y2): q22
```

Define horizontal and vertical fractions:

```text
u = (x - x1) / (x2 - x1)
v = (y - y1) / (y2 - y1)
```

The interpolated value is:

```text
q = q11*(1-u)*(1-v) + q21*u*(1-v) + q12*(1-u)*v + q22*u*v
```

The lab expresses the same calculation using coordinate distances divided by the rectangle area. The four weights sum to 1 for a nondegenerate rectangle. At a corner, that corner's weight becomes 1; at the center, every weight is 0.25.

For the lab's samples `[(1,1,1), (1,2,2), (2,1,3), (2,2,4)]`, the center `(1.5, 1.5)` gives `(1 + 2 + 3 + 4)/4 = 2.5`.

`sorted(points)` orders tuples by x, then y. Unpacking assigns the four samples to named variables; the underscored coordinate names are used to verify alignment. `raise ValueError(...)` stops execution when the supplied geometry is invalid.

The denominator must not be zero: the rectangle needs positive width and height. Adding `0.0` to the denominator, as in the notebook, is not a zero-division safeguard.

### 3.7 Bilinear interpolation for a whole image

The second function maps every output pixel back to a fractional input coordinate:

```python
def bilinear_resize(image, new_width, new_height):
    height, width, channels = image.shape
    resized_image = np.zeros(
        (new_height, new_width, channels), dtype=np.uint8
    )
    x_scale = width / new_width
    y_scale = height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x = j * x_scale
            y = i * y_scale
            x1 = int(x)
            x2 = min(x1 + 1, width - 1)
            y1 = int(y)
            y2 = min(y1 + 1, height - 1)
            x_weight = x - x1
            y_weight = y - y1

            a = image[y1, x1]
            b = image[y1, x2]
            c = image[y2, x1]
            d = image[y2, x2]
            resized_image[i,j] = (
                a * (1-x_weight) * (1-y_weight)
                + b * x_weight * (1-y_weight)
                + c * (1-x_weight) * y_weight
                + d * x_weight * y_weight
            )
    return resized_image
```

This preserves the notebook's calculation but uses a distinct function name. The original defines two unrelated functions named `bilinear_interpolation`, causing the image version to replace the point version.

The calculation works as follows:

1. `x_scale` and `y_scale` convert output positions into input positions.
2. `int(x)` and `int(y)` give the lower neighboring indices because these coordinates are nonnegative.
3. `min(..., width-1)` and `min(..., height-1)` prevent the upper neighbor from exceeding the image boundary.
4. Fractional parts give the interpolation weights.
5. The four RGB vectors are combined component by component; a separate channel loop is unnecessary.
6. Assignment into `uint8` converts the floating-point result to an integer.

For a fractional input position `(x, y) = (10.25, 20.5)`, the weights for `a`, `b`, `c`, and `d` are 0.375, 0.125, 0.375, and 0.125. If one channel's corner values are 0, 100, 200, and 240, the result is `117.5`, stored as 117 in this output array.

The lab's `new_width = 300` and `new_height = 200` produce shape `(200, 300, 3)`. Its coordinate rule is `x = j * width / new_width`, not an endpoint-aligned or half-pixel mapping. Different libraries can use different coordinate conventions, so correct implementations need not produce identical values.

At an enlarged image's right or bottom edge, the clamping may make two neighbors the same input pixel. Their weights then combine naturally. Dimensions must be positive integers. This simple bilinear implementation does not include an additional antialiasing filter for strong reductions.

## Lab 4: Histograms and intensity transformations

### 4.1 What a histogram represents

A histogram counts how many pixels have each intensity. For an 8-bit gray image, there are 256 possible intensities:

```text
h(k) = number of pixels whose intensity equals k
```

The horizontal axis is intensity; the vertical axis is pixel count. A narrow occupied intensity range often indicates low contrast. Concentration at low values indicates many dark pixels; concentration at high values indicates many bright pixels. These observations describe the distribution, not whether the photograph is good or bad.

A histogram does not retain pixel positions. Two images can have the same histogram but completely different spatial arrangements.

RGB images have a distribution for each channel. Lab 4 processes and plots the channels independently in its matching example.

### 4.2 Imports and the shared plotting helper

Lab 4 switches to direct imports:

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, data, io
```

`data.camera()` supplies the gray example; `io.imread(...)` loads source and reference images; `exposure` supplies the comparison algorithms. The notebook also imports `color` and `urllib.request`, but does not use them.

Its `plot_img_and_hist(image, title, bins=256)` helper creates an image plot and histogram side by side. This version includes one explicit improvement: fixed histogram bin limits for comparable 8-bit plots.

```python
def plot_img_and_hist(image, title, bins=256):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if len(image.shape) == 2:
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[1].hist(
            image.ravel(), bins=bins, range=(0, 256),
            color="black", alpha=0.7
        )
    else:
        axes[0].imshow(image)
        colors = ["red", "green", "blue"]
        for i, c in enumerate(colors):
            axes[1].hist(
                image[:,:,i].ravel(), bins=bins, range=(0, 256),
                color=c, alpha=0.5
            )
    axes[0].set_title(title)
    axes[0].axis("off")
    axes[1].set_title("Histogram")
    axes[1].set_xlim([0, 256])
    plt.tight_layout()
    plt.show()
```

`plt.subplots(1, 2)` creates one row with two plotting areas. `axes[0]` displays the image, and `axes[1]` displays the counts. `.ravel()` supplies a 1D sequence of pixel intensities; `enumerate(colors)` supplies both a channel index and its plotting color. `alpha` controls transparency so overlapping traces remain visible.

`vmin` and `vmax` fix the gray display scale. `axis("off")` hides image axes; `tight_layout()` adjusts spacing; `show()` displays the figure. `set_xlim` controls the visible horizontal range but, by itself, does not set histogram bin edges.

This helper assumes gray values on a 0-255 scale and ordinary RGB input. For a gray float image in 0-1, change the display and histogram ranges or explicitly convert its scale first.

### 4.3 Creating a low-contrast image

```python
img_original = data.camera()
img_low_contrast = (img_original.astype(float) / 255.0) * 50.0 + 100.0
img_low_contrast = img_low_contrast.astype(np.uint8)
```

Dividing by 255 maps the original range to 0-1. Multiplication by 50 compresses it to 0-50, and addition of 100 shifts it to 100-150.

Original black becomes 100, original white becomes 150, and a middle value becomes approximately 125. Dark and bright regions become closer in intensity, making the image look washed out. Converting to integers loses some distinctions between neighboring original values; later stretching cannot recover those exact discarded values.

### 4.4 Contrast stretching

Contrast stretching linearly maps the observed minimum and maximum into a chosen range:

```text
output = (input - old_min) / (old_max - old_min)
         * (new_max - new_min) + new_min
```

Each part has a role: subtraction moves the minimum to zero, division normalizes the original span, multiplication sets the new span, and the final addition sets the new minimum.

For input range 100-150 and output range 0-255:

| Input | Normalized value | Output before integer conversion |
| --- | --- | --- |
| 100 | 0 | 0 |
| 110 | 0.2 | 51 |
| 125 | 0.5 | 127.5 |
| 150 | 1 | 255 |

**Correction:** the notebook multiplies by `new_max` instead of `new_max - new_min`. Those agree only when `new_min` is zero, as in its demonstrated call.

Here is a corrected version for nonempty 8-bit input and `0 <= new_min <= new_max <= 255`. It also defines a policy for constant images:

```python
def manual_contrast_stretch(img, new_min, new_max):
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max == img_min:
        return np.full(img.shape, new_min, dtype=np.uint8)
    stretched = (
        (img.astype(float) - img_min) / (img_max - img_min)
        * (new_max - new_min) + new_min
    )
    return np.clip(stretched, new_min, new_max).astype(np.uint8)
```

A constant image has no intensity span to stretch, so ordinary division would fail. Filling with `new_min` is the explicit policy chosen here, not a rule from the original notebook.

The lab compares with:

```python
img_stretched_skimage = exposure.rescale_intensity(
    img_low_contrast, out_range=(0, 255)
)
```

It also prints minima and maxima to check the result. Stretching changes contrast by a single linear transformation; it does not use intensity frequencies to redistribute values.

### 4.5 PDF and CDF

The next algorithms use normalized counts and cumulative counts.

```text
pdf(k) = h(k) / N
cdf(k) = pdf(0) + pdf(1) + ... + pdf(k)
```

`N` is the total number of pixels. The lab calls the normalized histogram a probability distribution function (PDF); for discrete intensities, it is a probability mass function. Its entries sum to approximately 1.

The CDF is the fraction of pixels with intensity less than or equal to `k`. It never decreases and reaches approximately 1 by the highest occupied intensity.

For four possible levels and counts `[2, 1, 1, 0]`:

| Intensity | Count | Probability | Cumulative probability |
| --- | --- | --- | --- |
| 0 | 2 | 0.50 | 0.50 |
| 1 | 1 | 0.25 | 0.75 |
| 2 | 1 | 0.25 | 1.00 |
| 3 | 0 | 0.00 | 1.00 |

`cumsum()` is the NumPy operation that creates these running totals.

### 4.6 Manual histogram equalization

The notebook uses:

```python
def manual_histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    pdf = hist / len(img.flatten())
    cdf = pdf.cumsum()
    cdf_m = cdf * (len(pdf) - 1)
    map_to_s = cdf_m.astype("uint8")
    img_equalized = map_to_s[img]
    return img_equalized
```

Read it line by line:

1. `np.histogram` returns 256 counts and 257 bin edges. With 8-bit integer input and these limits, each count corresponds to one intensity.
2. `len(img.flatten())` counts pixels in this gray-image example; `img.size` would also give that count without making another flattened copy.
3. Dividing by the count gives probabilities.
4. `.cumsum()` converts them to cumulative probabilities.
5. `len(pdf) - 1` is 255, the maximum 8-bit output value.
6. Casting forms a 256-entry integer lookup table, discarding fractional parts.
7. `map_to_s[img]` replaces each original intensity with the table entry at that index.

The indexing in the last line is powerful: if `img` contains a pixel with value 100, the result at that position is `map_to_s[100]`. NumPy applies this lookup to the whole array while preserving its shape.

Using the four-level example above and a maximum output of 3 gives the table `[1, 2, 3, 3]` after truncation of `[1.5, 2.25, 3, 3]`. Input `[0, 0, 1, 2]` becomes `[1, 1, 2, 3]`.

This demonstrates two limitations: the output histogram need not be perfectly uniform, and the minimum output need not be zero. The lab uses direct CDF mapping without subtracting the first nonzero CDF value. With a constant nonempty input, this particular mapping sends the occupied intensity to 255.

Assume a nonempty 2D `uint8` image. Floating-point values cannot directly index this lookup table, and equalizing a combined RGB histogram is not the same as the gray procedure demonstrated here.

### 4.7 The library equalization comparison

```python
img_unequalized = data.camera()
img_eq_manual = manual_histogram_equalization(img_unequalized)
img_eq_skimage = exposure.equalize_hist(img_unequalized)
img_eq_skimage = (img_eq_skimage * 255).astype(np.uint8)
```

The notebook scales the library's 0-1 floating-point result into the 0-255 display range and shows the original, manual result, and library result using the same helper.

Equalization uses the image's intensity distribution to allocate output values. It is not merely adding brightness or linearly stretching endpoints, and it can also make noise or unwanted variation more visible.

### 4.8 Histogram matching: source and reference

Matching transforms a source image so its intensity distribution resembles that of a reference image. The reference specifies the target distribution; the two images do not need equal width and height.

The notebook reads two remote images with `io.imread(src_url)` and `io.imread(ref_url)`, then plots each. Its “cinematic” descriptions are illustrative labels: inspect the actual loaded images instead of assuming a URL's appearance or availability.

Matching does not copy the reference's objects, positions, or texture. It changes the source's intensities using a statistical mapping. Applied independently to RGB channels, it can change the source's color balance.

### 4.9 Calculating a channel CDF

```python
def calculate_cdf(channel):
    hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf.max())
    return cdf_normalized
```

`_` indicates that the returned histogram edges are unused. Since cumulative counts are nondecreasing, the maximum is the total number of counted pixels. Dividing by it turns cumulative counts into cumulative probabilities.

This is equivalent to normalizing counts before taking the cumulative sum, apart from floating-point rounding. It assumes a nonempty channel with values in the counted range; otherwise, the normalization denominator can be zero.

### 4.10 Building and applying the matching table

```python
def manual_match_histograms(src, ref):
    matched = np.zeros_like(src)
    for c in range(src.shape[2]):
        src_channel = src[:,:,c]
        ref_channel = ref[:,:,c]
        src_cdf = calculate_cdf(src_channel)
        ref_cdf = calculate_cdf(ref_channel)

        mapping = np.zeros(256, dtype=np.uint8)
        for src_pixel_val in range(256):
            closest_match = np.argmin(
                np.abs(ref_cdf - src_cdf[src_pixel_val])
            )
            mapping[src_pixel_val] = closest_match

        matched[:,:,c] = mapping[src_channel]
    return matched
```

`np.zeros_like(src)` preserves the source shape and dtype. The outer loop processes each channel. The inner loop considers each possible intensity, not each image pixel.

Suppose source intensity 100 has cumulative probability 0.60. The function searches the reference CDF for a value nearest 0.60. If reference intensity 140 has the closest value, then `mapping[100] = 140`: every source pixel with that channel intensity becomes 140.

`ref_cdf - src_cdf[src_pixel_val]` subtracts one probability from all 256 reference probabilities. `np.abs` measures unsigned differences; `np.argmin` returns the index of the smallest difference. If several entries tie, it returns the first minimum.

Finally, `mapping[src_channel]` applies the chosen values to every pixel in that channel. Repeating for all RGB channels completes the output.

This manual method assumes nonempty RGB `uint8` images. A gray image has no third axis, and a float image needs an appropriate representation before it can index an integer lookup table.

### 4.11 The library matching comparison

```python
img_matched_skimage = exposure.match_histograms(
    src_img, ref_img, channel_axis=-1
)
```

`channel_axis=-1` means the final array axis contains the channels. Compare the manual result with this library result using both image appearance and per-channel histograms.

Exact pixel equality is not the requirement: the manual method chooses a nearest CDF entry in a fixed 256-level table, while another implementation can interpolate the distribution mapping differently. Independent channel matching also does not ensure that relationships between RGB channels match those of the reference.

### 4.12 Comparing the three transformations

| Method | Information it uses | Main calculation | Goal |
| --- | --- | --- | --- |
| Stretching | Input minimum and maximum | Linear range mapping | Expand a narrow intensity span |
| Equalization | Input histogram/CDF | Map through the source CDF | Redistribute intensities using their frequency |
| Matching | Source and reference CDFs | Map source cumulative probabilities toward reference probabilities | Resemble a target distribution |

All three normally preserve image dimensions. They change values, whereas the geometric operations in Lab 3 change dimensions or the selected region.

## Connections and revision questions

### How the four labs build on one another

| Foundational idea | Later use |
| --- | --- |
| Shared objects and mutation | Copy an image before independent edits |
| NumPy shapes | Allocate canvases and distinguish gray from RGB |
| Slicing | Crop, draw, select channels, and extract shrinking blocks |
| Loops | Replicate pixels, interpolate output positions, and build mappings |
| Arithmetic and dtype conversion | Brightness, weighted averages, and contrast stretching |
| Functions | Reuse scaling, plotting, equalization, and matching |
| Array indexing | Apply histogram lookup tables to whole images |

### Questions with short answers

1. **Why does `y = x` sometimes appear safe for numbers but affect both variables for lists?** Rebinding one name leaves the other binding alone; mutating a shared object is visible through both names.
2. **What is `image[20, 40, 1]` in an RGB image?** The green component at row 20 and column 40.
3. **How many rows does `image[100:300]` select?** 200, provided the source has enough rows.
4. **Why is `.copy()` needed before some edits?** Assignment and basic slices can share the original storage.
5. **Why can adding brightness make bright pixels dark?** `uint8` addition can wrap past 255.
6. **Does reshaping perform image resizing?** No. It rearranges array dimensions while retaining element count; it does not estimate new pixel values.
7. **Why does nearest-neighbor enlargement look blocky?** A single input value fills multiple output positions.
8. **What does block-average shrinking preserve locally?** The average intensity of each selected block, subject to output conversion.
9. **Why are bilinear weights useful?** They make nearby neighbors contribute more and combine the four corner values smoothly.
10. **What does a CDF value of 0.8 at intensity 150 mean?** About 80% of the counted pixels have intensity at most 150.
11. **Does equalization guarantee equal counts in every bin?** No, especially with discrete levels and repeated input values.
12. **Why can different-sized images be histogram-matched?** Their normalized distributions are compared, not pixel coordinates.

### What to check when code gives an unexpected image

Check the input shape, dtype, and value range first. Then verify row/column order, RGB versus BGR, shared storage, and which variable is actually being displayed. For manual algorithms, inspect one small numerical example before applying the operation to a full image. Restart the notebook to uncover accidental dependence on old variables or overwritten functions.

## Notebook coverage map

This map includes all substantive sections of the four notebooks. Empty cells and repeated display-only cells introduce no additional algorithm, but their role is covered in the surrounding explanations.

| Notebook cells | Subject | Guide section |
| --- | --- | --- |
| Lab 1: 1-8 | Variables, reference behavior, mutability, object identity | 1.1-1.2 |
| Lab 1: 9-27 | Python data types and structures | 1.3 |
| Lab 1: 28-37 | Conditions, loops, functions | 1.4-1.6 |
| Lab 1: 38-61 | NumPy setup and array construction | 1.7-1.8 |
| Lab 1: 62-70 | Array dtypes | 1.9 |
| Lab 1: 71-85 | Indexing and slicing | 1.10 |
| Lab 1: 86-92 | Arithmetic and Boolean masks | 1.11 |
| Lab 2: 1-15 | Reading, displaying, inspecting, and casting | 2.1-2.2 |
| Lab 2: 16-24 | Brightness, channel edits, and crops | 2.3-2.5 |
| Lab 2: 25-31 | Noise and canvas drawing | 2.6-2.7 |
| Lab 2: 32-45 | Second image, colored bands, and plus sign | 2.8-2.9 |
| Lab 2: 46-53 | RGB splitting and recombination | 2.10 |
| Lab 2: 54-63 | OpenCV, downloading, Colab display, brightness | 2.11 |
| Lab 3: 1-9 | Nearest-neighbor enlargement | 3.1-3.3 |
| Lab 3: 10-11 | Crop and zoom | 3.4 |
| Lab 3: 12-16 | Block-average shrinking | 3.5 |
| Lab 3: 17-18 | Scalar bilinear interpolation | 3.6 |
| Lab 3: 19-22 | Image bilinear interpolation | 3.7 |
| Lab 4: 1-2 | Overview, imports, and plotting helper | 4.1-4.2 |
| Lab 4: 3-4 | Low-contrast input and contrast stretching | 4.3-4.4 |
| Lab 4: 5-6 | Histogram equalization | 4.5-4.7 |
| Lab 4: 7-9 | Histogram matching and comparison | 4.8-4.12 |

## Cell structure to copy for new tasks

This section shows how the lab notebooks are built **cell by cell**, so a new task notebook can copy the same layout. In the templates, each `[markdown]` / `[code]` label is one notebook cell. The labels are for this guide only; don't put them in the notebook.

### How each lab is laid out

**Lab 1 (concept → example):** explanation-heavy, with many small cells.

```text
[markdown]  # python                      ← big topic
[markdown]  ## Definitions                ← section
[markdown]  2–5 lines / bullet list explaining the idea
[code]      small example + print(f"...") + "# Output: ..." comments
[markdown]  ### Integers (int)            ← one sub-topic per heading
[markdown]  *   Stores whole numbers.
[code]      example
...
[markdown]  # numpy
[markdown]  ## Install numpy  →  [code] !pip install numpy
[markdown]  ## Import library →  [code] import numpy as np
[markdown]  ### Zero Array → #### 1D → [code] → #### 2D → [code] → #### N-D → [code]
[code]      type(zeroArray)               ← one-line inspection cell
```

**Lab 2 (operation → display → explanation):**

```text
[markdown]  # *Lab 2 will focus on how to read an image and manipulate it.*   ← goal line
[markdown]  # SKImage
[markdown]  ## Part 1
[markdown]  ### Read image
[code]      import skimage as ski
            image = ski.io.imread(r"https://...")
[code]      ski.io.imshow(image)          ← display in its own cell
[markdown]  To read gray image.           ← "To ..." lead-in sentence
[code]      read with as_gray=True
[code]      ski.io.imshow(x)
[markdown]  ### Image Manipulation
[markdown]  short explanation + "To increase brightness in the image."
[code]      operation + imshow in one cell
[markdown]  explanation of WHY (copy, channel indexing, ...)
...
[markdown]  ## Part 2
[code]      imports repeated at the start of the part
```

**Lab 3 (code first, build the algorithm up):** very little prose.

```text
[code]      import numpy as np / import skimage as ski      ← imports first
[markdown]  # Nearest Neighbor Image Interpolation
[code]      image = ski.data.astronaut(); ski.io.imshow(image)
[code]      image.shape                                      ← inspect
[code]      fixed-size demo: np.zeros(...) canvas + nested loops
[code]      ski.io.imshow(result)
[code]      def function(...) ... return
            #----------------------------------------#      ← separator
            parameters + call
[code]      result.shape
[code]      ski.io.imshow(result)
[markdown]  ## Zooming (NNeighbor)
[code]      crop + call the same function + imshow           ← apply to a new case
[markdown]  # Image Interpolation (Shrinking)
[code]      loop version
[code]      ski.io.imshow(...)
[markdown]  # BL interpolation
[code]      single-point example with a "# Test" and expected value
[code]      full-image function
[code]      load image → [code] call with new size + imshow
```

**Lab 4 (one long section per algorithm):** the most complete layout.

```text
[markdown]  # Lab: <Title>
            1-paragraph welcome/overview
            numbered list of the algorithms covered (bold names)
            "Let's start by importing our libraries and defining a helper..."
[code]      all imports + shared helper function (plot_img_and_hist)
[markdown]  ---
            ## 1. <Algorithm name>
            **Concept:** what it does and when it helps
            **The Math:** $$ LaTeX formula $$  + what the symbols mean
[code]      # ==========================================
            # 1. SETUP: <prepare input>
            # ==========================================
            ...input + plot
            # ==========================================
            # 2. MANUAL IMPLEMENTATION
            # ==========================================
            def manual_<name>(...): numbered "# 1. ..., # 2. ..." step comments
            call + plot + print checks
            # ==========================================
            # 3. SKIMAGE IMPLEMENTATION
            # ==========================================
            library call + plot
[markdown]  ---
            ## 2. <Next algorithm>
...
```

(Labs 1–3 sometimes write `#SKImage` or `##Part 1` with no space after `#`. Add the space in new notebooks so the heading renders.)

### Common rules (all labs)

| Rule | How it appears in the labs |
| --- | --- |
| Headings in their own markdown cell | `#` topic, `##` part/section, `###` operation, `####` variant |
| Short headings | "Read image", "Draw plus sign", "Zooming (NNeighbor)" |
| Imports before first use | Own cell at the top (Labs 3–4) or at the start of a Part (Lab 2) |
| One idea per code cell | Most cells are 1–10 lines; Lab 4 is the exception (one cell per algorithm) |
| Always show the result | `ski.io.imshow(...)`, `plot_img_and_hist(...)`, `print(f"...")`, or `.shape` |
| Inspect, don't assume | One-line cells such as `image.shape`, `type(image)`, `image` |
| `.copy()` before editing | `newImage = image.copy()` |
| Image canvases | `np.zeros((h, w, c), dtype=np.uint8)` filled by loops |
| Explain the "why" | Short markdown after a tricky cell (copy, channel index, overflow) |
| Function pattern | `def` → separator line `#----#` → parameters → call |
| Manual first, library second | Lab 4: `manual_...` function, then `exposure.<function>` to compare |

### Template A: small image task (Lab 2 / Lab 3 style)

Use this for pixel edits, drawing, cropping, channel work, resizing, and similar tasks.

````text
[markdown]
# <Task title>

[markdown]
*This task will focus on <one-sentence goal>.*

[code]
import numpy as np
import skimage as ski

[markdown]
## Part 1

[markdown]
### Read image

[code]
image = ski.data.astronaut()
ski.io.imshow(image)

[code]
image.shape

[markdown]
### <Operation 1>

[markdown]
To <do X>, we <short explanation>.

[code]
newImage = image.copy()
# <what this line does>
newImage[<rows>, <cols>, <channel>] = <value>

[code]
ski.io.imshow(newImage)

[markdown]
<1–2 sentences: why it works / what to watch out for.>

[markdown]
### <Operation 2 as a function>

[code]
def my_operation(image, param_a, param_b):
  canvas = np.zeros((<new_h>, <new_w>, image.shape[2]), dtype=np.uint8)
  for i in range(<rows>):
    for j in range(<cols>):
      canvas[i, j] = <value computed from image>
  return canvas
#------------------------------------------------------------------------------------------------------------------------------------#
param_a = 2
param_b = 2
result = my_operation(image, param_a, param_b)

[code]
result.shape

[code]
ski.io.imshow(result)

[markdown]
## Part 2
...
````

### Template B: algorithm task (Lab 4 style)

Use this for histogram tasks or any "implement it manually, then compare with the library" task.

````text
[markdown]
# Lab: <Title>

<1 short paragraph: what we will do and why it matters.>

We will cover:
1. **<Algorithm 1>**
2. **<Algorithm 2>**

Let's start by importing our libraries and defining a helper function to visualize our results.

[code]
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, data, io

# Helper function to plot an image and its histogram side-by-side
def plot_img_and_hist(image, title, bins=256):
    ...   # copy from lab4_CV.ipynb

[markdown]
---
## 1. <Algorithm 1>

**Concept:** <what it does, when to use it>

**The Math:**
$$<formula>$$

Where <explain the symbols>.

[code]
# ==========================================
# 1. SETUP: <describe the input>
# ==========================================
img_input = data.camera()
plot_img_and_hist(img_input, "Original Image")

# ==========================================
# 2. MANUAL IMPLEMENTATION
# ==========================================
def manual_<algorithm>(img):
    # 1. <first step>
    ...
    # 2. <second step>
    ...
    return result

img_manual = manual_<algorithm>(img_input)
plot_img_and_hist(img_manual, "Manual <Algorithm>")

# ==========================================
# 3. SKIMAGE IMPLEMENTATION
# ==========================================
img_skimage = exposure.<function>(img_input)
plot_img_and_hist(img_skimage, "Skimage <Algorithm>")

[markdown]
---
## 2. <Algorithm 2>
...
````

If you want the smaller-cell style, split the Lab 4 code cell at the three banners (setup / manual / skimage). The teaching order stays the same.

### Which template to use

| The task asks you to... | Copy |
| --- | --- |
| Practice Python or NumPy basics | Lab 1 layout (concept markdown → small example → print) |
| Edit pixels, crop, draw, split channels | Template A, Part 1 only |
| Implement a resizing / pixel-loop algorithm | Template A with the function section (Lab 3) |
| Implement an algorithm and compare with skimage | Template B (Lab 4) |

### Before submitting

- [ ] Title cell and (optionally) a one-line goal cell
- [ ] Imports in a cell before first use
- [ ] Headings in their own markdown cells, short and spelled correctly
- [ ] Every operation followed by a visible result (`imshow` / plot / `print` / `.shape`)
- [ ] Short markdown explaining any tricky step
- [ ] `.copy()` used where the original must stay unchanged
- [ ] `uint8` ranges handled (convert → `np.clip` → convert back)
- [ ] Manual implementation shown before any library shortcut (when the task is about the algorithm)
- [ ] Kernel restarted and all cells run top to bottom; outputs saved
