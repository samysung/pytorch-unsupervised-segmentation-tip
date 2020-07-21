# Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering

This pytorch code generates segmentation labels of an input image.

Wonjik Kim\*, Asako Kanezaki\*, and Masayuki Tanaka.
**Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering.** 
*IEEE Transactions on Image Processing*, accepted, 2020.
([arXiv](https://arxiv.org/abs/2007.09990))

\*W. Kim and A. Kanezaki contributed equally to this work.

## Requirements

python3, pytorch, opencv2, scikit-image, tqdm

## Getting started

### Vanilla

    $ python demo.py --input ./BSD500/101027.jpg

### Vanilla + scribble loss

    $ python demo.py --input ./PASCAL_VOC_2012/2007_001774.jpg --scribble

### Vanilla + reference image(s)

    $ python demo_ref.py --input ./BBC/
