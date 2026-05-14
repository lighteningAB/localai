#include "scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>

namespace imagegen {

namespace {

// Convert an alphas_cumprod entry into the (alpha_t, sigma_t, lambda_t) triple
// used by the DPM-Solver update. Uses the natural (alpha_bar) parameterization
// rather than the karras-sigma form, since SD 1.5 does not use karras sigmas.
std::tuple<float, float, float> alphaSigmaLambda(float alphaCumprod) {
    // Guard against log(0) at the post-final "clean" step.
    constexpr float kFloor = 1e-12f;
    float ac    = std::max(alphaCumprod, kFloor);
    float oneMa = std::max(1.0f - alphaCumprod, kFloor);
    float alpha = std::sqrt(ac);
    float sigma = std::sqrt(oneMa);
    float lambda = std::log(alpha) - std::log(sigma);
    return {alpha, sigma, lambda};
}

}  // namespace

Scheduler::Scheduler() : Scheduler(Config{}) {}

Scheduler::Scheduler(const Config& cfg) : cfg_(cfg) {
    buildBetaSchedule();
}

void Scheduler::buildBetaSchedule() {
    const int T = cfg_.numTrainTimesteps;
    betas_.resize(T);
    alphasCumprod_.resize(T);

    switch (cfg_.schedule) {
        case BetaSchedule::Linear:
            for (int t = 0; t < T; ++t) {
                float fr = static_cast<float>(t) / static_cast<float>(T - 1);
                betas_[t] = cfg_.betaStart + (cfg_.betaEnd - cfg_.betaStart) * fr;
            }
            break;
        case BetaSchedule::ScaledLinear: {
            const float a = std::sqrt(cfg_.betaStart);
            const float b = std::sqrt(cfg_.betaEnd);
            for (int t = 0; t < T; ++t) {
                float fr = static_cast<float>(t) / static_cast<float>(T - 1);
                float v  = a + (b - a) * fr;
                betas_[t] = v * v;
            }
            break;
        }
        case BetaSchedule::SquaredCosCapV2: {
            auto alphaBar = [](float t) {
                constexpr float kHalfPi = 1.57079632679489661923f;
                float x = (t + 0.008f) / 1.008f * kHalfPi;
                float c = std::cos(x);
                return c * c;
            };
            for (int t = 0; t < T; ++t) {
                float t1   = static_cast<float>(t)     / static_cast<float>(T);
                float t2   = static_cast<float>(t + 1) / static_cast<float>(T);
                float beta = 1.0f - alphaBar(t2) / alphaBar(t1);
                betas_[t]  = std::min(beta, 0.999f);
            }
            break;
        }
    }

    float c = 1.0f;
    for (int t = 0; t < T; ++t) {
        c *= (1.0f - betas_[t]);
        alphasCumprod_[t] = c;
    }
}

void Scheduler::setTimesteps(int N) {
    if (N < 1) {
        throw std::invalid_argument("setTimesteps: N must be >= 1");
    }
    timesteps_.resize(N);
    if (N == 1) {
        timesteps_[0] = cfg_.numTrainTimesteps - 1;
    } else {
        const float step = static_cast<float>(cfg_.numTrainTimesteps - 1)
                         / static_cast<float>(N - 1);
        for (int i = 0; i < N; ++i) {
            timesteps_[i] = static_cast<int>(
                std::round(static_cast<float>(cfg_.numTrainTimesteps - 1) - i * step)
            );
        }
    }
    x0History_.clear();
}

int Scheduler::timestep(int i) const {
    return timesteps_.at(static_cast<size_t>(i));
}

std::vector<float> Scheduler::step(const std::vector<float>& eps,
                                   int i,
                                   const std::vector<float>& x) {
    if (eps.size() != x.size()) {
        throw std::invalid_argument("step: eps and x must have the same length");
    }
    const int N = numInferenceSteps();
    if (i < 0 || i >= N) {
        throw std::out_of_range("step: inferenceStepIndex out of range");
    }

    const int tNow  = timesteps_[i];
    const int tNext = (i + 1 < N) ? timesteps_[i + 1] : 0;

    auto [alphaT,    sigmaT,    lambdaT]    = alphaSigmaLambda(alphasCumprod_[tNow]);
    auto [alphaNext, sigmaNext, lambdaNext] = alphaSigmaLambda(alphasCumprod_[tNext]);

    // eps → x_0 prediction.
    std::vector<float> x0(eps.size());
    for (std::size_t k = 0; k < eps.size(); ++k) {
        x0[k] = (x[k] - sigmaT * eps[k]) / alphaT;
    }

    // Append to history (keep at most the last 2 entries for the multistep update).
    x0History_.push_back(std::move(x0));
    if (x0History_.size() > 2) {
        x0History_.erase(x0History_.begin());
    }
    const auto& m0 = x0History_.back();

    const float h          = lambdaNext - lambdaT;
    const float expNegHm1  = std::exp(-h) - 1.0f;
    const float scaleX     = sigmaNext / sigmaT;
    const float scaleD     = -alphaNext * expNegHm1;

    std::vector<float> out(eps.size());

    if (x0History_.size() < 2) {
        // 1st-order DPM-Solver++ for the first step (no previous m to extrapolate from).
        for (std::size_t k = 0; k < eps.size(); ++k) {
            out[k] = scaleX * x[k] + scaleD * m0[k];
        }
    } else {
        // 2nd-order multistep (midpoint variant of DPM-Solver++).
        const auto& m1 = x0History_.front();
        // Lambda at the previous inference step.
        const int tPrev = timesteps_[i - 1];
        auto [alphaPrev, sigmaPrev, lambdaPrev] = alphaSigmaLambda(alphasCumprod_[tPrev]);
        (void)alphaPrev; (void)sigmaPrev;  // only lambdaPrev is needed
        const float h0 = lambdaT - lambdaPrev;
        const float r0 = (std::abs(h) > 1e-12f) ? (h0 / h) : 1.0f;
        for (std::size_t k = 0; k < eps.size(); ++k) {
            const float D0 = m0[k];
            const float D1 = (r0 != 0.0f) ? ((m0[k] - m1[k]) / r0) : 0.0f;
            out[k] = scaleX * x[k] + scaleD * D0 + 0.5f * scaleD * D1;
        }
    }

    return out;
}

}  // namespace imagegen
