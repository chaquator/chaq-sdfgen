/*
 * map pixel byte value to bool (true or false)
 */
static bool map_read(uchar byte) {
    unsigned char threshold = 127;
    return byte > threshold;
}

/*
 * Clamped linear remap
 */
static float linear_remap(float val, float src_min, float src_max, float dst_min, float dst_max) {
    val = val > src_max ? src_max : val;
    val = val < src_min ? src_min : val;
    float sn = src_max - src_min;
    float nd = dst_max - dst_min;
    return (((val - src_min) * nd) / sn) + dst_min;
}

static bool read(int x, int y, read_only image2d_t img, uchar use_luminence) {
    uint4 pixel = read_imageui(img, (int2)(x, y));
    uchar p_val = use_luminence ? pixel.x : pixel.w;
    return map_read(p_val);
}

kernel void sdf(read_only image2d_t img_in, write_only image2d_t img_out, ulong spread, //
                uchar use_luminence, uchar invert, uchar asymmetric) {
    size_t w = get_global_size(0);
    size_t h = get_global_size(1);

    // TODO: come up with different access strategies for the radius and compare performance on large images
    // in general, every access strategy needs to return the pixel whose distance to (x,y) is not larger than that of
    // any other pixel and its value is different than that of (x,y)

    // search in spread radius for closest pixel
    ulong x = (ulong)get_global_id(0);
    ulong y = (ulong)get_global_id(1);

    ulong2 closest_pixel = (ulong2)(0, 0);
    bool found_candidate = false;
    ulong cur_closest_d_2 = 0;

    bool this_px = read(x, y, img_in, use_luminence);

    // clamp bounds of for loop
    ulong sp1 = spread + 1;
    ulong lb_y = sp1 > y ? 0 : y - sp1;
    ulong ub_y = sp1 > (h - y) ? h : y + sp1;
    ulong lb_x = sp1 > x ? 0 : x - sp1;
    ulong ub_x = sp1 > (w - x) ? w : x + sp1;

    long dx;
    long dy;
    long d_2;
    uint4 px;
    uchar p_val;
    bool search_px;

    ulong cy, cx;
    for (cy = lb_y; cy < ub_y; ++cy) {
        for (cx = lb_x; cx < ub_x; ++cx) {
            dx = cx - x;
            dy = cy - y;
            d_2 = (dx * dx) + (dy * dy);
            if (d_2 > (sp1 * sp1)) continue;

            search_px = read(cx, cy, img_in, use_luminence);

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
    float this_dist = 0;
    bool decider = invert ? this_px == false : this_px == true;
    if (found_candidate) {
        float d = sqrt((float)(cur_closest_d_2));
        this_dist = decider ? d : -(d - 1);
    } else {
        this_dist = decider ? INFINITY : -INFINITY;
    }

    // map distacne to output value
    float src_min = asymmetric ? 0 : -((float)spread);
    uint val = (uint)linear_remap(this_dist, src_min, (float)spread, 0.f, 255.f);

    // write back
    uint4 col = (uint4)(val, 255, 255, 255);
    write_imageui(img_out, (int2)(x, y), col);
}
