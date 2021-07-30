/*
 * combine unsigned distance into signed and remap into image
 * ptr to inside
 * ptr to outside
 * img w, img aux_vertex_heights
 * spread
 * asymmetric
 * ptr to output image bytes
 *
 */

/*
 * map pixel byte value to float (0 or infinity)
 * invert == false --> pixel >= 127 ? inf : 0
 * invert == true --> pixel < 127 ? inf : 0
 */
static float map_byte(uchar byte, uchar invert) {
    unsigned char threshold = 127;
    bool decider = (invert != 0) ? (byte > threshold) : (byte <= threshold);
    return decider ? INFINITY : 0.f;
}
