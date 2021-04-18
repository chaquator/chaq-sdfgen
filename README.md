# chaq-sdfgen
chaq-sdfgen is a short program written in C with OpenMP (2.0 compatible) which generates a signed distance field
bitmap from an input image.

## Example result
![Input](image/sample_input.png)
![Output](image/sample_output.png)

Result generated using `chaq_sdfgen -i sample_input.png -o sample_output.png -s 100 -al`.

## References
[Original article](http://cs.brown.edu/people/pfelzens/dt/)

[Implementation which chaq-sdfgen is loosely based on](https://github.com/dy/bitmap-sdf)

