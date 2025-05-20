---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [remove-cell]
import os
os.chdir("..")
```

# Notes

These are notes for Introduction to Computer Vision, a demo for the [Davis
Python Users Group][dpug].

[dpug]: https://datalab.ucdavis.edu/davis-python-users-group/

:::{important}
[Click here][data] to access the shared data directory if you want to follow
along with the live demo.

[data]: https://ucdavis.box.com/s/4chzepm36nnxba11jvjsk6of8m3w2e8c
:::


(sec-getting-started)=
## Getting Started with Image Data

Computers represent data with numbers, and there are many different ways to
represent an image with numbers. One common way is to divide an image into a
grid of tiny single-color squares, called **pixels**, and represent the color
of each pixel as a tuple of numbers. Images represented this way are called
**raster graphics**.

<style type="text/css">
    .black, .blue, .gold, .white {
        display: inline-block;
        vertical-align: sub;
        width: 1em;
        height: 1em;
        border: 1px solid;
    }
    .black { background-color: #000000 }
    .blue  { background-color: #0000ff }
    .gold  { background-color: #ffdd00 }
    .white { background-color: #ffffff }
</style>

A **color model** is a system for representing colors as tuples of numbers.
Each element of the tuple is called a **channel**. The range of colors a color
model can represent is called the **gamut** (GAM-it) of the model.

For example, the **red-green-blue (RGB) color model** represents colors as
mixtures of the primary colors of light: red, green, and blue. The [RGB
model][rgb] corresponds to way most projectors, televisions, and computer
monitors actually work. For each channel/primary color (red, green, and blue),
$0.0$ intensity means the light is off, and $1.0$ means the light is as bright
as possible. Some examples:

[rgb]: https://en.wikipedia.org/wiki/RGB_color_model

* $(0, 0, 0)$ is black <span class="black"></span>
* $(0, 0, 1)$ is pure blue <span class="blue"></span>
* $(1.0, 0.87, 0.0)$ is a warm yellow <span class="gold"></span>
* $(1, 1, 1)$ is white <span class="white"></span>

A few other common color models are:

* [Hue-saturation-value (HSV)][hsv] and hue-saturation-lightness (HSL) models
  reparameterize the RGB model in terms of perceived hue, saturation
  (colorfulness), and lightness or value (emitted light). The HSV and HSL
  models are convenient processing related to how colors are perceived, such as
  identifying "red" pixels in an image.
* Cyan-magenta-yellow (CMY) model, which represents colors as mixtures of the
  primary colors of ink. The related [cyan-magenta-yellow-key (CMYK)
  model][cmyk] is widely used in printing and represents black, or *key*,
  separately, to produce more accurate blacks and minimize ink use.
* Grayscale, which only allows shades of gray (including black and white).
  Grayscale only has 1 channel.

[hsv]: https://en.wikipedia.org/wiki/HSL_and_HSV
[cmyk]: https://en.wikipedia.org/wiki/CMYK_color_model

Channels are sometimes encoded as bytes: numbers from 0 to 255, inclusive. For
the RGB model, `0` means $0.0$ and `255` means $1.0$. In this encoding:

* `(0, 0, 0)` is black <span class="black"></span>
* `(0, 0, 255)` is pure blue <span class="blue"></span>
* `(255, 221, 0)` is a warm yellow <span class="gold"></span>
* `(255, 255, 255)` is white <span class="white"></span>


## Packages

:::{note}
This section lists a few general-purpose packages for image processing and
computer vision. For some kinds of image data, such as geospatial images and
medical images, there are specialized packages that are more appropriate and
convenient for common use-cases.
:::

My preferred packages are ⭐'d.

Numerical computing (tensor/linear algebra) frameworks:

* ⭐[NumPy][]: Python's most popular numerical computing package. Supported by
  a wide variety of other packages. Fast and stable. Limited to CPUs.
    * [JAX][]: Extends NumPy to run on other hardware (e.g., GPUs) and support
      deep learning (by adding automatic differentiation). The [PIX][] package
      provides specific support for computer vision.
* ⭐[PyTorch][]: Python's most popular deep learning package. Provides specific
  support for computer vision through the Torchvision subpackage.
* [TensorFlow][]: Another deep learning package.

[NumPy]: https://numpy.org/
[JAX]: https://docs.jax.dev/
[PIX]: https://dm-pix.readthedocs.io/
[PyTorch]: https://pytorch.org/
[TensorFlow]: https://www.tensorflow.org/


Image processing packages:

* ⭐[scikit-image][]: The programming interface is more Pythonic than OpenCV,
  but the package doesn't have as many features and isn't as fast. Will feel
  familiar if you've used [scikit-learn][]. Images are just NumPy arrays. A
  good starting point for most projects unless you know you'll need OpenCV or a
  deep learning package.
* ⭐[OpenCV][] (opencv-python): Developed as C++ library, but has official
  Python bindings. Lots of features, but the programming interface and
  documentation are sometimes a little arcane.
* ⭐[SciPy][]: A good supplement to scikit-image. Images are just NumPy arrays.
  Notably includes fast Fourier transform functions. Slower than OpenCV.
* ⭐[Pillow][]: A mature, general-purpose image processing package. Based on
  the older, unmaintained Python Imaging Library (PIL). Not designed
  specifically for computer vision, but many deep learning packages expect the
  Pillow images as input.
    * [Imageio][]: A readers/writers for over 295 image formats.
* [ITK][]: Originally focused on medical applications, but now broadening to a
  general-purpose package.
    * [SimpleITK][]: Simplifies the ITK programming interface.
* [Wand][]: Bindings to the ImageMagick library.

[OpenCV]: https://opencv.org/
[Pillow]: https://python-pillow.github.io/
[Imageio]: https://imageio.readthedocs.io/
[scikit-image]: https://scikit-image.org/
[scikit-learn]: https://scikit-learn.org/
[SciPy]: https://scipy.org/
[ITK]: https://itk.org/
[SimpleITK]: https://simpleitk.org/
[Wand]: https://docs.wand-py.org/

Other relevant packages:

* ⭐[Matplotlib][]: For visualizing images. Many image processing packages
  provide visualization functions, but Matplotlib is more flexible.
* [Transformers][]: Access deep learning models on [Hugging Face][hf] (a
  popular model hosting site) for computer vision and more. Requires PyTorch,
  TensorFlow, or JAX to run models (depending on the model).

[Matplotlib]: https://matplotlib.org/
[Transformers]: https://huggingface.co/docs/transformers/
[hf]: https://huggingface.co/

If you want to use an off-the-shelf machine learning (not deep learning) model
on image data, use an image processing library to transform the images into
features for your model. Then use [scikit-learn][] to fit and assess the model.

There are many different pretrained deep learning (neural network) models
available for classification, object detection, segmentation, and other tasks.
Cutting-edge models tend to be distributed as standalone packages (and might
eventually make their way into [Transformers][] or [scikit-learn][]). Deep
learning models are typically implemented in [PyTorch][], [TensorFlow][], or
[JAX][], so it's helpful to be familiar with those packages.

:::{tip}
Training your own deep learning model from scratch usually isn't feasible
unless you have lots of training data and compute resources. Instead, it's
common to take a pretrained model and **fine-tune** it by training the last few
layers of the model (the classifier) on custom data.
:::

:::{caution}
It can be difficult to install PyTorch (with Torchvision) and OpenCV in the
same environment, because their dependencies are often incompatible.
:::


## Processing Images with scikit-image

Let's use scikit-image to read and transform an image. We'll use Matplotlib to
display the image. The module name for scikit-image is `skimage`, and it's
conventionally imported as `ski`:

```{code-cell}
import matplotlib.pyplot as plt
import skimage as ski
```

The `ski.io.imread` function can read an image, and Matplotlib's `plt.imshow`
function can display one:

```{code-cell}
flower = ski.io.imread("data/flower.jpg")
plt.imshow(flower)
```

The image is just a NumPy array:

```{code-cell}
type(flower)
```

It has 3 dimensions: height, width, and channels. In this case they are:

```{code-cell}
flower.shape
```

A slice along the 3rd dimension gives the values for a single channel. The
scikit-image package reads most images as RGB by default, so here's the red
channel:

```{code-cell}
flower[:, :, 0]
```

The package provides a variety of filters, transformations, and other
operations; see [the documentation][ski-docs] for details.

[ski-docs]: https://scikit-image.org/docs/stable/

For example, you can rotate an image with the `ski.transform.rotate` function:

```{code-cell}
rotated = ski.transform.rotate(flower, 35)
plt.imshow(rotated)
```

There are also functions for more complicated algorithms, such as edge
detection. For instance, the `ski.feature.canny` function runs the Canny edge
detection algorithm. The algorithm requires a grayscale image, so first we
convert the image with `ski.color.rgb2gray`:

```{code-cell}
gray_flower = ski.color.rgb2gray(flower)
edges = ski.feature.canny(gray_flower)
plt.imshow(edges)
```

Different color models are useful for different tasks. Suppose we want to
select or recolor the flowers in the image. The flowers have a distinct yellow
color, so we can use an HSV model and select the pixels by hue. To get started,
we'll set up a helper function to plot HSV images (Matplotlib assumes RGB or
grayscale) and convert the flower image to HSV:

```{code-cell}
def imshow_hsv(hsv):
    plt.imshow(ski.color.hsv2rgb(hsv))

hsv_flower = ski.color.rgb2hsv(flower)
imshow_hsv(hsv_flower)
```

The scikit-image HSV format uses floating point numbers from $0.0$ to $1.0$
rather than bytes. Yellow is around 0.15 (you can use [an online color
picker][color-picker] to figure this out; many color pickers treat hue in HSV
as an angle between 0 and 360).

[color-picker]: https://www.selecolor.com/en/hsv-color-picker/

```{code-cell}
hue = hsv_flower[:, :, 0]
is_yellow = (0.1 < hue) & (hue < 0.2)
```

The result is a matrix of Boolean values with the same dimensions as the image;
this is called a **mask**. You can visualize a mask with `plt.imshow` (purple
means `False`, yellow means `True`):

```{code-cell}
plt.imshow(is_yellow)
```

The mask looks pretty good. Let's try shifting hue of all of the pixels under
the mask so that they're pale blue. Pale blue is around 0.5, so we'll add about
0.35 to the hue of pixels under the mask:

```{code-cell}
recolor = hsv_flower.copy()
recolor[is_yellow, 0] += 0.35
imshow_hsv(recolor)
```

Some parts of the flowers don't get selected because they have low saturation
(colorfulness) and are actually closer to a green hue (0.3). We can add this
case to the mask:

```{code-cell}
sat = hsv_flower[:, :, 1]
is_yellow = (
    ((0.1 < hue) & (hue < 0.2)) |
    ((0.2 < hue) & (hue < 0.4) & (sat < 0.3))
)

recolor = hsv_flower.copy()
recolor[is_yellow, 0] += 0.35
imshow_hsv(recolor)
```

Filtering on HSV is simple and computationally cheap, so it is often a good
starting point for selecting parts of images. Edge detection algorithms like
the one we saw earlier are also useful for this purpose.


## Images in Memory

There are many different ways to represent images in memory.
{ref}`sec-getting-started` described several color models, but color models are
not the only thing that can differ in practice.

The most common representation for images is RGB with dimensions `(height,
width, channels)`. The RGB values may be encoded as bytes (0 to 255; reflects
how images are often stored in files) or floating-point numbers (0.0 to 1.0;
convenient for transforming images with less loss of precision). This is the
default representation for scikit-image, SciPy, Pillow, and Matplotlib.

Some packages with notably different defaults:

* OpenCV represents images as blue-green-red (BGR) with dimensions `(height,
  width, channels)`.
* PyTorch and TensorFlow represent images as RGB with dimensions `(channels,
  height, width)`.

Most packages provide functions to convert between common representations.


## Machine Learning Models

If you want to train a machine learning model on images, the main thing you
need to think about is how you'll represent the images as **features** (input
variables) for the model. Most models require a fixed-length vector of numbers
as input. While it is possible to use the raw pixel values (for example, in
RGB) as input, doing so means:

* All of your images must be the same size, because all of the feature vectors
  must be the same size.
* Throwing away spatial information about the image, because the 3-dimensional
  image array must be unraveled into a 1-dimensional feature vector.
* Providing the model with very large feature vectors (which can cause poor
  performance for most models) or using low resolution images.

As a result, it's usually better to engineer features from images rather than
using the raw pixel values. There are many different ways to engineer features,
and the best features usually depend on deep understanding of the problem you
want the model to solve. Some examples include representations in different
bases (such as a Fourier Transform, Wavelets, or Principal Components), edge
masks (or other masks), and [scale-invariant feature transformation][sift]
(SIFT).

[sift]: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform

:::{note}
A major reason for the success of neural networks and other deep learning
models is that they engineer features automatically as part of their training
process.
:::

Once you have suitable features for an ML model, you can use any machine
learning package. The premier machine learning package for Python is
[scikit-learn][].

For finding appropriate features and an appropriate model, it can be helpful to
think about how you plan to use the model:

* Classification: the model predicts whether each image is in a particular
  category
* Object detection: the model predicts a category and bounding box for (a
  subset of) objects within the image
* Segmentation: the model predicts a category for every pixel in the image,
  typically in order to partition the different parts of the image


## Deep Learning Models

An easy way to get started with deep learning models is to use the
[Transformers][] package, which can download and run models from [Hugging
Face][hf]. The package documentation includes tutorials for many different
kinds of models.

Let's use try Transformers for object detection and segmentation. The
`pipeline` function is helpful for quickly trying a model:

```{code-cell}
from transformers import pipeline
```

We'll start with object detection, using a pretrained model:

```{code-cell}
detector = pipeline(
    "object-detection",
    "facebook/detr-resnet-50", revision="no_timm"
)
```

Transformers requires Pillow (imported as `PIL`) images as input, so let's read
the cat image:

```{code-cell}
from PIL import Image

cat = Image.open("data/cat.jpg")
cat
```

Now try running the detector:

```{code-cell}
result = detector(cat)
result
```

The detector found the cat! Object detection models can only detect the objects
on which they were trained; this model happens to have been trained on cats
(among other things). Fine-tuning pretrained models to detect a different set
of objects than those they were originally trained on usually works well (and
requires less training data and compute time).

Let's try segmenting the image (with a different model) instead:

```{code-cell}
segmenter = pipeline("image-segmentation", "openmmlab/upernet-convnext-tiny")

segments = segmenter(cat)
segments
```

We can display the segments on the image with a custom plotting function:

```{code-cell}
import numpy as np

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


plt.imshow(cat)
ax = plt.gca()
for x in segments:
    show_mask(np.array(x["mask"]) / 255, ax=ax, random_color=True)
plt.axis("off")
plt.show()
```

