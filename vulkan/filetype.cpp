#include "filetype.h"

#include <spdlog/spdlog.h>

namespace filetype {

filetype from_str(std::string_view name, filetype fallback) {
    using namespace std::literals::string_view_literals;

    // compare with lowercase
    std::string lower{name};
    std::transform(lower.cbegin(), lower.cend(), lower.begin(), [](const char c) { return std::tolower(c); });
    spdlog::trace("\"{}\" -> \"{}\"", name, lower);

    using filetype_pair = std::pair<std::string_view, filetype>;
    std::initializer_list<filetype_pair> type_map = {
        {"png"sv, filetype::png}, {"jpeg"sv, filetype::jpeg}, {"jpg"sv, filetype::jpeg},
        {"tga"sv, filetype::tga}, {"bmp"sv, filetype::bmp},
    };
    auto find = std::find_if(type_map.begin(), type_map.end(), [&name = lower](const auto& p) {
        const auto find = name.find(p.first);
        return find != std::string_view::npos;
    });

    if (find == type_map.end()) return fallback;

    return find->second;
}

std::string_view to_str(filetype type) {
    using namespace std::literals::string_view_literals;
    std::unordered_map<filetype, std::string_view> filetype_map = {
        {filetype::bmp, "bmp"sv},
        {filetype::jpeg, "jpg"sv},
        {filetype::png, "png"sv},
        {filetype::tga, "tga"sv},
    };
    return filetype_map[type];
}

} // namespace filetype
