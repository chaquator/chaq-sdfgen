/*
 * map pixel byte value to float (true or false)
 */
static bool map_byte(uchar byte, uchar invert) {
    unsigned char threshold = 127;
    return byte > threshold;
    /*
    bool decider = invert ? (byte <= threshold) : (byte > threshold);
    return decider;
    */
}

kernel void sdf(read_write image2d_t img, ulong spread, //
                uchar use_luminence, uchar invert, uchar asymmetric) {
    bool (^read)(int, int) = ^(int x, int y) {
      uint4 pixel = read_imageui(img, (int2)(x, y));
      uchar p_val = use_luminence ? pixel.x : pixel.w;
      return map_byte(p_val, invert);
    };

    // limit spread to size of diagonal by max
    size_t w = get_global_size(0);
    size_t h = get_global_size(1);
    size_t diag = (size_t)ceil(pow(pow((float)w, 2.f) + pow((float)h, 2.f), 0.5f));
    spread = min(spread, (ulong)diag);

    // TODO: come up with different access strategies for the radius and compare performance on large images
    // in general, every access strategy needs to return the pixel whose distance to (x,y) is not larger than that of
    // any other pixel and its value is different than that of (x,y)

    // search in spread radius for closest pixel
    ulong x = (ulong)get_global_id(0);
    ulong y = (ulong)get_global_id(1);

    ulong2 closest_pixel = (ulong2)(x, y);
    bool found_candidate = false;
    ulong cur_closest_d_2 = 0;

    bool this_px = read(x, y);

    ulong sp1 = spread + 1;
    ulong lb_y = sp1 > y ? 0 : y - sp1;
    ulong ub_y = sp1 > (h - y) ? h : y + sp1;
    ulong lb_x = sp1 > x ? 0 : x - sp1;
    ulong ub_x = sp1 > (w - x) ? w : x + sp1;
    for (ulong cy = lb_y; cy < ub_y; ++cy) {
        for (ulong cx = lb_x; cx < ub_x; ++cx) {
            long dx = cx - x;
            long dy = cy - y;
            long d_2 = dx * dx + dy * dy;
            if (d_2 > (sp1 * sp1)) continue;

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
    float this_dist;
    bool decider = invert ? this_px == false : this_px == true;
    if (!found_candidate) {
        this_dist = decider ? INFINITY : -INFINITY;
    } else {
        float d = sqrt((float)(cur_closest_d_2));
        this_dist = decider ? d : -(d - 1);
    }
    
    // TODO: current issue looks like some sort of data race, somethings being written back incorrectly

    // map distacne to output value
    // clamped linear remap
    float s_min = asymmetric ? 0 : -(float)spread;
    float s_max = (float)spread;
    float d_min = 0.f;
    float d_max = 255.f;

    float sn = s_max - s_min;
    float nd = d_max - d_min;

    float v = this_dist;
    v = v > s_max ? s_max : v;
    v = v < s_min ? s_min : v;

    float remap = (((v - s_min) * nd) / sn) + d_min;
    uint val = (uint)(remap);

    // write back
    // val = found_candidate ? closest_pixel.x << 4 | closest_pixel.y : 0;
    uint4 col = (uint4)(val, 255, 255, 255);
    write_imageui(img, (int2)(x, y), col);
}
