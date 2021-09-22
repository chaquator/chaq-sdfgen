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

// search strategies
// in general, every search strategy must return a pixel--which is different in value from this_val--for which there is
// no other pixel with a smaller distance to this_px (x,y).
// if none found, return this_px

struct opt_ul2 {
    ulong2 point;
    bool valid;
};

static void set_bounds(ulong2 point, ulong2 dim, ulong2* lb, ulong2* ub, ulong spread) {
    // clamp bounds of for loop
    lb->y = spread > point.y ? 0 : point.y - spread;
    ub->y = spread > (dim.y - point.y) ? dim.y : point.y + spread;
    lb->x = spread > point.x ? 0 : point.x - spread;
    ub->x = spread > (dim.x - point.x) ? dim.x : point.x + spread;
}

// basic square search with one early exit optimization
static struct opt_ul2 search_square(bool this_val, ulong2 this_px, ulong2 dim, ulong spread, read_only image2d_t img,
                                    uchar use_luminence) {
    ulong2 closest_pixel = this_px;
    ulong closest_d_2 = 0;
    bool found_candidate = false;

    // clamp bounds of for loop
    ulong2 lb, ub;
    ulong sp1 = spread + 1;
    set_bounds(this_px, dim, &lb, &ub, sp1);

    ulong cy, cx;
    for (cy = lb.y; cy < ub.y; ++cy) {
        for (cx = lb.x; cx < ub.x; ++cx) {
            long dx = this_px.x - cx;
            long dy = this_px.y - cy;
            ulong d_2 = (ulong)(dx * dx) + (ulong)(dy * dy);
            if (d_2 > (sp1 * sp1)) continue;

            bool search_val = read(cx, cy, img, use_luminence);

            // find closest pixel not same as this_val
            if (search_val != this_val) {
                if (d_2 < closest_d_2 || !found_candidate) {
                    closest_pixel = (ulong2)(cx, cy);
                    closest_d_2 = d_2;
                    found_candidate = true;

                    // update bounds for early exit
                    ulong new_bound = floor(sqrt((double)d_2));
                    set_bounds(this_px, dim, &lb, &ub, new_bound);
                }
            }
        }
    }

    struct opt_ul2 ret = {closest_pixel, found_candidate};
    return ret;
}

kernel void sdf(read_only image2d_t img_in, write_only image2d_t img_out, ulong spread, //
                uchar use_luminence, uchar invert, uchar asymmetric) {
    size_t w = get_global_size(0);
    size_t h = get_global_size(1);

    // search in spread radius for closest pixel
    ulong x = (ulong)get_global_id(0);
    ulong y = (ulong)get_global_id(1);
    bool this_val = read(x, y, img_in, use_luminence);

    struct opt_ul2 close_result =
        search_square(this_val, (ulong2)(x, y), (ulong2)(w, h), spread, img_in, use_luminence);
    ulong2 closest_px = close_result.point;
    bool found_candidate = close_result.valid;
    // bool found_candidate = all(closest_px != (ulong2)(x, y));

    // compute distance to pixel
    float this_dist = 0;
    bool decider = invert ? this_val == false : this_val == true;
    if (found_candidate) {
        ulong2 delta_to_closest = closest_px - (ulong2)(x, y);
        float d = sqrt((float)((delta_to_closest.x * delta_to_closest.x) + (delta_to_closest.y * delta_to_closest.y)));
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
