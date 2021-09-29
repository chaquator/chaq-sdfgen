R"STRING_CL(
    // map pixel byte to bool
    static bool
    map_read(uchar byte) {
    unsigned char threshold = 127;
    return byte > threshold;
}

// read value at pixel and map to bool
static bool read(int2 point, read_only image2d_t img, uchar use_luminence) {
    uint4 pixel = read_imageui(img, point);
    uchar p_val = use_luminence ? pixel.x : pixel.w;
    return map_read(p_val);
}

// Clamped linear remap
static float linear_remap(float val, float src_min, float src_max, float dst_min, float dst_max) {
    val = val > src_max ? src_max : val;
    val = val < src_min ? src_min : val;
    float sn = src_max - src_min;
    float nd = dst_max - dst_min;
    return (((val - src_min) * nd) / sn) + dst_min;
}

// set bounds (lower bound, upper bound) based on current point, dimensions, and spread
static void set_bounds(ulong2 point, ulong2 dim, ulong2* lb, ulong2* ub, ulong spread) {
    // clamp bounds of for loop
    lb->y = spread > point.y ? 0 : point.y - spread;
    ub->y = spread > (dim.y - point.y) ? dim.y : point.y + spread;
    lb->x = spread > point.x ? 0 : point.x - spread;
    ub->x = spread > (dim.x - point.x) ? dim.x : point.x + spread;
}

// search strategies
// in general, every search strategy must return a pixel--which is different in value from this_val--for which there is
// no other pixel with a smaller distance to this_px (x,y).
// if none found, return this_px

// basic square search with one early exit optimization
static ulong2 search_square(bool this_val, ulong2 this_px, ulong2 dim, ulong spread, read_only image2d_t img,
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

            bool search_val = read((int2)(cx, cy), img, use_luminence);

            // find closest pixel not same as this_val
            if (search_val != this_val) {
                if (d_2 < closest_d_2 || closest_d_2 == 0) {
                    closest_pixel = (ulong2)(cx, cy);
                    closest_d_2 = d_2;

                    // update bounds for early exit
                    ulong new_bound = floor(sqrt((double)d_2));
                    set_bounds(this_px, dim, &lb, &ub, new_bound);
                }
            }
        }
    }

    return closest_pixel;
}

// explores in an 4-way symmetric triangle originating from this_px
static ulong2 search_triangle(bool this_val, ulong2 this_px, ulong2 dim, ulong spread, read_only image2d_t img,
                              uchar use_luminence) {
    ulong2 closest_px = this_px;

    ulong spread_2 = spread * spread;

    // u - primary direction, v - secondary direction
    ulong u = 1;
    ulong v;
    while ((u * u) <= spread_2) {

#define CHECK_RET(ox, oy)                                                                                              \
    {                                                                                                                  \
        if (read(convert_int2(this_px) + (int2)(ox, oy), img, use_luminence) != this_val) {                            \
            return convert_ulong2(convert_long2(this_px) + (long2)(ox, oy));                                           \
        }                                                                                                              \
    }

#define CHECK_BREAK(ox, oy)                                                                                            \
    {                                                                                                                  \
        if (read(convert_int2(this_px) + (int2)(ox, oy), img, use_luminence) != this_val) {                            \
            closest_px = convert_ulong2(convert_long2(this_px) + (long2)(ox, oy));                                     \
            spread_2 = d_2;                                                                                            \
            break;                                                                                                     \
        }                                                                                                              \
    }

        // test straight ahead on all 4 axes
        // if candidate found, we can return immediately
        ulong2 remain = dim - this_px;

        long2 check_ul = this_px >= (ulong2)(u);
        long2 check_lr = remain >= (ulong2)(u);
        // left
        if (check_ul.x) {
            CHECK_RET(-u, 0);
        }
        // up
        if (check_ul.y) {
            CHECK_RET(0, -u);
        }
        // right
        if (check_lr.x) {
            CHECK_RET(u, 0);
        }
        // down
        if (check_lr.y) {
            CHECK_RET(0, u);
        }

        v = 1;
        ulong d_2 = u * u;
        while (v < u) {
            d_2 += (v << 1) - 1;
            if (d_2 > spread_2) break;

            // test +v and -v on all 4 axes
            // if candidate found, we can limit spread_2 to d_2
            // left (-x)
            if (check_ul.x) {
                // -y
                if (check_ul.y) {
                    CHECK_BREAK(-u, -v);
                }
                // +y
                if (check_lr.y) {
                    CHECK_BREAK(-u, v);
                }
            }
            // up (-y)
            if (check_ul.y) {
                // -x
                if (check_ul.x) {
                    CHECK_BREAK(-v, -u);
                }
                // +x
                if (check_lr.x) {
                    CHECK_BREAK(v, -u);
                }
            }
            // right (+x)
            if (check_lr.x) {
                // -y
                if (check_ul.y) {
                    CHECK_BREAK(u, -v);
                }
                // +y
                if (check_lr.y) {
                    CHECK_BREAK(u, v);
                }
            }
            // down (+y)
            if (check_lr.y) {
                // -x
                if (check_ul.x) {
                    CHECK_BREAK(-v, u);
                }
                // +x
                if (check_lr.x) {
                    CHECK_BREAK(v, u);
                }
            }

            ++v;
        }
        ++u;

#undef CHECK_BREAK
#undef CHECK_RET
    }

    return closest_px;
}

kernel void sdf(read_only image2d_t img_in, write_only image2d_t img_out, ulong spread, //
                uchar use_luminence, uchar invert, uchar asymmetric) {
    size_t w = get_global_size(0);
    size_t h = get_global_size(1);

    // search in spread radius for closest pixel
    ulong x = (ulong)get_global_id(0);
    ulong y = (ulong)get_global_id(1);
    bool this_val = read((int2)(x, y), img_in, use_luminence);

    ulong2 closest_px = search_triangle(this_val, (ulong2)(x, y), (ulong2)(w, h), spread, img_in, use_luminence);
    bool found_candidate = any(closest_px != (ulong2)(x, y));

    // compute distance to pixel
    float this_dist = 0;
    bool decider = invert ^ this_val;
    if (found_candidate) {
        long2 delta_to_closest = convert_long2(closest_px) - (long2)(x, y);
        float d = sqrt((float)((delta_to_closest.x * delta_to_closest.x) + (delta_to_closest.y * delta_to_closest.y)));
        this_dist = decider ? d : -(d - 1);
    } else {
        this_dist = decider ? INFINITY : -INFINITY;
    }

    // map distacne to output value
    float src_min = asymmetric ? 0 : -((float)spread);
    uint val = (uint)linear_remap(this_dist, src_min, (float)spread, 0.f, 255.f);

    // write back
    uint4 col = (uint4)((uint3)(val), 255);
    write_imageui(img_out, (int2)(x, y), col);
}
)STRING_CL"
