// Host-side test for imagegen::Scheduler. Compile + run on the dev Mac:
//   ./app/src/main/cpp/tests/run_host_tests.sh
//
// Verification (per IMAGE-GEN-PLAN.md §3.5 / §4):
//   - alphas_cumprod[0]   close to 0.999
//   - alphas_cumprod[999] close to the SD 1.5 well-known value (~0.0047)
//     (The plan said "approx 0.06" but recomputation for scaled_linear
//      with beta_start=0.00085, beta_end=0.012, T=1000 gives ~0.00478.
//      The plan's number is an order of magnitude off — trust the math.)
//   - A step with zero noise prediction must produce finite output.

#include "../scheduler.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int gFailures = 0;

#define REQUIRE_NEAR(actual, expected, tol, label)                                \
    do {                                                                          \
        const double a_ = (actual);                                               \
        const double e_ = (expected);                                             \
        if (std::abs(a_ - e_) > (tol)) {                                          \
            std::fprintf(stderr,                                                  \
                "FAIL  %-40s got=%.8g  expected=%.8g  tol=%.3g\n",                \
                (label), a_, e_, (tol));                                          \
            ++gFailures;                                                          \
        } else {                                                                  \
            std::fprintf(stdout, "OK    %-40s = %.8g\n", (label), a_);            \
        }                                                                         \
    } while (0)

#define REQUIRE_FINITE(value, label)                                              \
    do {                                                                          \
        const double v_ = (value);                                                \
        if (!std::isfinite(v_)) {                                                 \
            std::fprintf(stderr, "FAIL  %-40s got=%.8g (non-finite)\n",           \
                (label), v_);                                                     \
            ++gFailures;                                                          \
        } else {                                                                  \
            std::fprintf(stdout, "OK    %-40s finite = %.8g\n", (label), v_);     \
        }                                                                         \
    } while (0)

}  // namespace

int main() {
    using imagegen::Scheduler;

    // ---- Beta schedule sanity ----------------------------------------------------
    Scheduler sched;
    const auto& ac = sched.alphasCumprod();

    REQUIRE_NEAR(ac.front(), 0.99915, 1e-4, "alphas_cumprod[0]");
    // SD 1.5 well-known value for scaled_linear at T-1.
    REQUIRE_NEAR(ac.back(),  0.00478, 5e-4, "alphas_cumprod[999]");
    // Monotonically decreasing.
    bool monotone = true;
    for (std::size_t i = 1; i < ac.size(); ++i) {
        if (ac[i] > ac[i - 1]) { monotone = false; break; }
    }
    if (!monotone) { std::fprintf(stderr, "FAIL  alphas_cumprod is not monotonic\n"); ++gFailures; }
    else           { std::fprintf(stdout, "OK    alphas_cumprod monotonically decreasing\n"); }

    // ---- step() stability under zero noise prediction ----------------------------
    // With eps == 0 and x == zeros, x0_pred == zeros, and the update reduces to
    // (sigma_next / sigma_t) * x == 0. So out must be all zeros, certainly finite.
    sched.setTimesteps(20);
    std::vector<float> x(4 * 64 * 64, 0.0f);
    std::vector<float> eps(x.size(), 0.0f);

    auto x1 = sched.step(eps, 0, x);
    REQUIRE_FINITE(x1[0],         "step(0)[0]");
    REQUIRE_FINITE(x1.back(),     "step(0).back()");
    REQUIRE_NEAR(x1[0], 0.0, 1e-6,"step(0)[0] == 0 (zero in -> zero out)");

    // ---- step() stability under nonzero but bounded inputs -----------------------
    // Synthetic: eps ~ small, x ~ N(0,1)-ish (deterministic). After 20 steps the
    // latent magnitude must stay finite (not NaN/Inf).
    for (std::size_t k = 0; k < x.size(); ++k) {
        x[k]   = static_cast<float>(((k * 1103515245u + 12345u) & 0xFFFF) / 65535.0 * 2.0 - 1.0);
        eps[k] = 0.01f * x[k];
    }
    std::vector<float> cur = x;
    for (int i = 0; i < 20; ++i) {
        cur = sched.step(eps, i, cur);
    }
    REQUIRE_FINITE(cur[0],     "after 20 steps cur[0]");
    REQUIRE_FINITE(cur.back(), "after 20 steps cur.back()");
    double sum = 0.0;
    for (float v : cur) sum += v;
    REQUIRE_FINITE(sum, "after 20 steps sum");

    // ---- Inference timesteps shape ----------------------------------------------
    if (sched.numInferenceSteps() != 20) {
        std::fprintf(stderr, "FAIL  numInferenceSteps got=%d expected=20\n",
                     sched.numInferenceSteps());
        ++gFailures;
    } else {
        std::fprintf(stdout, "OK    numInferenceSteps == 20\n");
    }
    if (sched.timestep(0) != 999 || sched.timestep(19) != 0) {
        std::fprintf(stderr, "FAIL  inference timesteps [%d..%d] (expected 999..0)\n",
                     sched.timestep(0), sched.timestep(19));
        ++gFailures;
    } else {
        std::fprintf(stdout, "OK    inference timesteps span 999..0\n");
    }

    std::fprintf(stdout, "\n%s — %d failure(s)\n",
                 (gFailures == 0 ? "PASS" : "FAIL"), gFailures);
    return gFailures == 0 ? 0 : 1;
}
