// Vision A8Q4G64 linears at real Qwen3.8-27B Vision extents: FP64-oracle check of the
// cooperative prefill route and event timing of the one-wave WMMA route against it.
#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"

int main() {
    try {
        struct Case {
            Shape shape;
            std::uint32_t tokens;
            const char* name;
        };
        // One 1920x1080 image: P=8256 patches, 2064 merged tokens; one 768x512 image: P=1536.
        const std::vector<Case> cases{
            {{1152U, 1536U, 0U, 0U}, 8256U, "patch"},
            {{3456U, 1152U, 0U, 0U}, 8256U, "qkv"},
            {{1152U, 1152U, 0U, 0U}, 8256U, "proj"},
            {{4304U, 1152U, 0U, 0U}, 8256U, "fc1"},
            {{1152U, 4304U, 0U, 0U}, 8256U, "fc2"},
            {{4608U, 4608U, 0U, 0U}, 2064U, "merger_fc1"},
            {{5120U, 4608U, 0U, 0U}, 2064U, "merger_fc2"},
            {{3456U, 1152U, 0U, 0U}, 1536U, "qkv"},
            {{4304U, 1152U, 0U, 0U}, 1536U, "fc1"},
            {{1152U, 4304U, 0U, 0U}, 1536U, "fc2"},
        };
        double wave_total = 0.0;
        double cta_total = 0.0;
        for (const Case& c : cases) {
            const PrefillResult numerical = run_prefill_case(c.shape, c.tokens, false, false);
            if (numerical.maximum_bf16_steps > 2U) { fail("vision A8Q4 FP64-oracle criterion failed"); }
            const PrefillResult timing = run_prefill_case(c.shape, c.tokens, false, true);
            std::cout << c.name << " N=" << c.shape.rows << " K=" << c.shape.columns
                      << " T=" << c.tokens << " oracle_bf16_steps=" << numerical.maximum_bf16_steps
                      << " wmma32_ms=" << timing.one_wave_ms
                      << " prefill_cta_ms=" << timing.cooperative_cta_ms << '\n';
            if (c.tokens == 8256U || c.tokens == 2064U) {
                const double weight = c.tokens == 2064U || c.shape.columns == 1536U ? 1.0 : 27.0;
                wave_total += weight * timing.one_wave_ms;
                cta_total += weight * timing.cooperative_cta_ms;
            }
        }
        std::cout << "1080p encode linears: wmma32_ms=" << wave_total
                  << " prefill_cta_ms=" << cta_total << '\n'
                  << "vision A8Q4 linear qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vision A8Q4 linear qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
