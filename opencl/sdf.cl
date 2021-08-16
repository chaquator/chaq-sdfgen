/*
 * map pixel byte value to float (0 or infinity)
 * invert == false --> pixel >= 127 ? inf : 0
 * invert == true --> pixel < 127 ? inf : 0
 */
static bool map_byte(uchar byte, uchar invert) {
    unsigned char threshold = 127;
    bool decider = (invert != 0) ? (byte > threshold) : (byte <= threshold);
    return decider;
}

kernel void sdf(read_write image2d_t img, ulong spread, //
                uchar use_luminance, uchar invert) {
    bool (^read)(int, int) = ^(int x, int y) {
      uint4 pixel = read_imageui(img, (int2)(x, y));
      uchar p_val = use_luminance ? pixel.r : pixel.a;
      return map_byte(p_val, invert);
    };

    // limit spread to size of diagonal by max
    size_t w = get_global_size(0);
    size_t h = get_global_size(1);
    size_t diag = (size_t)ceil(pow(pow((float)w, 2.f) + pow((float)h, 2.f), 0.5f));
    spread = min(spread, (ulong)diag);

    // TODO: come up with different access strategies for the radius and compare performance on large images

    // search in spread radius for closest pixel
    ulong x = (ulong)get_global_id(0);
    ulong y = (ulong)get_global_id(1);
    ulong2 closest_pixel = (ulong2)(x, y);
    bool found_candidate = false;
    ulong cur_closest_d_2 = 0;

    bool this_px = read(x, y);

    ulong lb_y = spread > y ? 0 : y - spread;
    ulong ub_y = spread > (h - y) ? h : y + spread;
    ulong lb_x = spread > x ? 0 : x - spread;
    ulong ub_x = spread > (w - x) ? w : x + spread;
    for (ulong cy = lb_y; cy < ub_y; ++cy) {
        for (ulong cx = lb_x; cx < ub_x; ++cx) {
            long dx = cx - x;
            long dy = cy - y;
            long d_2 = dx * dx + dy * dy;
            if (d_2 >= spread) continue;

            bool search_px = read(cx, cy);

            // find closest pixel not same as this_px
            if (search_px != this_px) {
                if (!found_candidate) {
                    closest_pixel = (ulong2)(cx, cy);
                    cur_closest_d_2 = d_2;
                    found_candidate = true;
                } else if (d_2 < cur_closest_d_2) {
                    closest_pixel = (ulong2)(cx, cy);
                    cur_closest_d_2 = d_2;
                }
            }
        }
    }

    // compute distance to pixel
    // map distacne to output value
    // write back
}
