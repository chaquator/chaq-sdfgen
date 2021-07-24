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
    bool decider = (invert != 0) ? (byte < 127) : (byte >= 127);
    return decider ? INFINITY : 0.f;
}

/*
 * intersect two standard parabolas
 */
static float parabola_intersect(float2 p1, float2 p2) {
    return ((p2.y - p1.y) + ((p2.x * p2.x) - (p1.x * p1.x))) / (2 * (p2.x - p1.x));
}

/*
 * part one of distance transform, takes in image as bytes and calculates df along horizontal axis,
 * writing back to output image in float for part 2
 */
kernel void dist_transform_part1(ulong width, ulong height,                   //
                                 ulong bytes_per_pixel, ulong channel_offset, //
                                 uchar invert,                                //
                                 global uchar* in_image,                      //
                                 global float* out_image,                     //
                                 local ulong* aux_vertices,                   //
                                 local float* aux_vertex_heights,             //
                                 local float* aux_breakpoints) {

#define pixel(index) img_row[2 * index + channel_offset]
#define map(index) map_byte(pixel(index), invert)

    // index image buffer by global id
    size_t y = get_global_id(0);
    uchar* img_row = in_image + (y * bytes_per_pixel * width);

    // part 1: compute lower envelope as set of break points and vertices

    // start at first non-infinity parabola
    size_t offset = 0;
    while (offset < width) {
        if (isinf(map(img_row[offset]))) break;
        ++offset;
    }

    // if every parabola is infinity, row is "empty", is complete as far as we care
    if (offset == width) {
        // fill empty row on transpose
        for (size_t i = 0; i < width; ++i) {
            size_t tpose_idx = y + (width * i);
            out_image[tpose_idx] = INFINITY;
        }
        return;
    }

    // first vertex is the first non-infinity parabola
    aux_vertices[0] = (ulong)offset;
    aux_vertex_heights[0] = map(offset);

    size_t k = 0;
    for (size_t q = offset + 1; q < width; ++q) {
        // skip parabolas at infinite height
        if (isinf(map(offset))) continue;

        float2 p1 = (float2)(aux_vertices[k], map(aux_vertices[k]));
        float2 p2 = (float2)(q, map(q));

        // calculate intersection of current parabola and next candidate
        float s = parabola_intersect(p1, p2);

        // if intersection comes before current left bound, back up and change the necessary breakpoint
        // skip for k == 0 bc there is no left bound to look back on (it is at -INF)
        while (k > 0 && s <= aux_breakpoints[k - 1]) {
            --k;
            float2 p1 = (float2)(aux_vertices[k], map(aux_vertices[k]));
            float s = parabola_intersect(p1, p2);
        }

        // once the intersection comes after the current left bound, add this parabola to the structure
        aux_breakpoints[k] = s;
        ++k;
        aux_vertices[k] = p2.x;
        aux_vertex_heights[k] = p2.y;
    }

    // part 2: populate output with lower envelope from the data structure
    size_t j = 0;
    for (size_t q = 0; q < width; ++q) {
        // seek break point past q
        while (j < k && aux_breakpoints[j] < (float)q) ++j;

        // set height at current position (q) along output to lower envelope
        size_t v_j = aux_vertices[j];
        float displacement = (float)q - (float)v_j;

        // output transposed
        size_t tpose_idx = y + width * q;
        out_image[tpose_idx] = displacement * displacement + aux_vertex_heights[j];
    }

#undef pixel
#undef map
}

// TODO: switch from aux_vertices and aux_vertex_heights to aux_verts which will be float2[]
