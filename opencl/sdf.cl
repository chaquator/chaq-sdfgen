/*
 * compute unsigned distance field float --
 * input image2df
 * img w, img h
 * bytes per pixel
 * offset for channel selection
 * bool invert
 * ptr to output image float
 *
 * combine unsigned distance into signed and remap into image
 * ptr to inside
 * ptr to outside
 * img w, img h
 * spread
 * asymmetric
 * ptr to output image bytes
 *
 */

kernel void testy() {
    /*
    "Work dim: %u\n"
     "Global size: %u\n"
     "Local size: %u\n"
     "Group size: %u\n"
     "Global id: %u\n"
     "Local id: %u\n"
     "Group id: %u\n"
     "----------------\n";
     */

    uint dim = get_work_dim();
    size_t gs = get_global_size(0);
    size_t ls = get_local_size(0);
    size_t gr_s = get_num_groups(0);
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t grid = get_group_id(0);
    printf("%u\n", gid);
}
