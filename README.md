# Computer Vision University Labs

This repository collects our university Computer Vision labs, study notes, and practice tasks. We are learning how images work as NumPy arrays, implementing image-processing algorithms step by step, and using the course notebooks as references for future assignments.

The project has two goals: understand the concepts and code in Labs 1-4, and write new task notebooks using the same structure and coding style.

## What we are studying

| Lab | Topics | Main skills |
| --- | --- | --- |
| [Lab 1: Python and NumPy](Lab_1.ipynb) | Variables, mutability, data types, conditions, loops, functions, array creation, indexing, arithmetic, and Boolean masks | Build the Python and array foundations needed for image processing |
| [Lab 2: Image manipulation](Lab2_CV.ipynb) | Reading and displaying images, brightness, cropping, noise, drawing, RGB channels, and OpenCV | Manipulate pixels directly and understand image shapes and data types |
| [Lab 3: Interpolation](Lab_3_CV.ipynb) | Nearest-neighbor enlargement, crop-and-zoom, block-average shrinking, and bilinear interpolation | Implement resizing algorithms using loops, slices, and weighted calculations |
| [Lab 4: Histograms](lab4_CV.ipynb) | Contrast stretching, histogram equalization, and histogram matching | Understand intensity distributions and compare manual implementations with scikit-image |

Our approach is to inspect the input, explain the operation, implement it, and display the result. Manual implementations make the underlying calculations visible; library comparisons help us examine the results.

## Files and guides

```text
cv__lab_uni/
├── README.md
├── Lab_1.ipynb
├── Lab2_CV.ipynb
├── Lab_3_CV.ipynb
├── lab4_CV.ipynb
├── tutorial.md
├── labs_explained.md
└── Tasks_in_Lab/
    └── task120230129_1.ipynb
```

- **[labs_explained.md](labs_explained.md)** explains the four labs in detail, including code, worked calculations, important corrections, and revision questions. Use it to study what the notebooks are doing and why.
- **[tutorial.md](tutorial.md)** is our `tutorial_1.0` style guide. It explains how to structure new notebooks, name variables, write explanations, show results, and follow the examples from the course.
- **[Tasks_in_Lab/](Tasks_in_Lab/)** holds practice and assignment notebooks. The existing `task120230129_1.ipynb` is currently an empty placeholder, not a runnable task.

The original notebooks are course references. The guides identify mistakes and assumptions in those examples; they do not mean that every issue has been fixed in the original notebooks.

## How to work through the project

1. Start with **Lab 1**, then continue through Labs 2, 3, and 4.
2. Read the corresponding explanation in **labs_explained.md** while studying each notebook.
3. Run cells in order and inspect the arrays, image dimensions, data types, and displayed results.
4. Use **tutorial.md** before writing a new task in `Tasks_in_Lab/`.
5. Follow the assignment requirements and use the closest lab as the implementation and presentation reference.
6. Restart the kernel and run the completed task from top to bottom before saving it with its outputs.

## Running the notebooks

The notebooks use Python 3 with NumPy, scikit-image, Matplotlib, and OpenCV. They contain Google Colab conventions; Lab 2 specifically uses `google.colab.patches.cv2_imshow` and `/content/` paths.

### Google Colab

Open a lab notebook in Google Colab by uploading it or selecting this repository through Colab's GitHub option. Run the cells in order and make sure any required image files or downloads are available in that session.

### Local Jupyter

Install the main packages in your Python environment:

```bash
python -m pip install numpy scikit-image matplotlib opencv-python notebook
python -m notebook
```

Open the desired `.ipynb` file in Jupyter. These commands provide a starting environment; the repository does not currently pin package versions.

For local execution, adapt Colab-specific paths and display calls. Some cells load remote images and therefore need internet access. The saved labs also contain deprecation warnings for `ski.io.imshow`; Lab 4 demonstrates Matplotlib-based image display.

Consult the corrections in the guides if a reference cell fails or produces an unexpected result. Examples include the `widht` typo, overflow when adding to `uint8`, and the duplicate bilinear function name.

## Writing future tasks

Keep the work easy to follow:

- Use short headings and explanations that describe the current operation.
- Prefer the familiar NumPy array operations and small functions demonstrated in the labs.
- Show intermediate and final results, including image-and-histogram comparisons when relevant.
- Check row/column order, RGB versus BGR, intensity ranges, and whether an edit needs an independent copy.
- Implement algorithms manually when required; include library comparisons when the assignment asks for them.

The detailed conventions and checklist are in [tutorial.md](tutorial.md). The assignment instructions take priority over the guide.
