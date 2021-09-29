# chaq-sdfgen
chaq-sdfgen is a set of two programs which both generate a signed distance field bitmap from an input image. One version of chaq-sdfgen is written in C and utilizes OpenMP (2.0 compatible) to accelerate the SDF computation. The other version is written in C++ and utilizes OpenCL (2.2 compatible) to compute the SDF with an OpenCL GPU device.

## Example result
![Input](image/sample_input.png)
![Output](image/sample_output.png)

Result generated using `chaq_sdfgen -i sample_input.png -o sample_output.png -s 100 -al`.

## References
[Felzenszwalb/Huttenlocher distance transform](http://cs.brown.edu/people/pfelzens/dt/), which the OpenMP version
implements.

[Other program](https://github.com/dy/bitmap-sdf) which the OpenMP version was loosely based on.

## Aside
The OpenMP version seems to consistently perform better than the OpenCL version. For small images, I assume the overhead of setting up OpenCL is slow. While for large images the OpenCL version uses an asymptotically slower approach which has a runtime of O(n^2 * s^2) -- where *n* is the image's size and *s* is the spread radius -- compared to the OpenMP version which runs in O(n^2).
