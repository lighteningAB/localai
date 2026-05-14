#pragma once

#include <vector>

namespace imagegen {

// DPM-Solver Multistep scheduler (dpmsolver++ algorithm, 2nd-order multistep,
// midpoint solver) for epsilon-prediction diffusion models. Defaults match SD 1.5.
//
// Usage:
//   Scheduler s;                            // default SD 1.5 config
//   s.setTimesteps(20);                     // 20 inference steps
//   // x = gaussian noise, shape (1,4,64,64), flat-row-major as one std::vector<float>
//   for (int i = 0; i < 20; ++i) {
//     // eps = unet.run(x, s.timestep(i), text_embeds);  // epsilon prediction
//     x = s.step(eps, i, x);
//   }
//
// Internal state (cached x_0 predictions) is reset by setTimesteps().
class Scheduler {
public:
    enum class BetaSchedule {
        ScaledLinear,    // SD 1.5 default: linspace(sqrt(a), sqrt(b), T)^2
        Linear,
        SquaredCosCapV2, // Improved DDPM cosine schedule (Nichol & Dhariwal 2021)
    };

    struct Config {
        int          numTrainTimesteps = 1000;
        float        betaStart         = 0.00085f;
        float        betaEnd           = 0.012f;
        BetaSchedule schedule          = BetaSchedule::ScaledLinear;
    };

    Scheduler();                              // SD 1.5 defaults
    explicit Scheduler(const Config& cfg);

    // Pick N inference timesteps (descending from T-1 toward 0) and reset state.
    void setTimesteps(int numInferenceSteps);

    // UNet timestep value for inference step index i (0..N-1).
    int timestep(int inferenceStepIndex) const;

    int numInferenceSteps() const { return static_cast<int>(timesteps_.size()); }

    // One denoising step. `eps` is the UNet's epsilon (noise) prediction; `x` is
    // the current latent. Returns the next latent (one step closer to clean).
    // `eps` and `x` must have the same length; treated as flat tensors.
    std::vector<float> step(const std::vector<float>& eps,
                            int inferenceStepIndex,
                            const std::vector<float>& x);

    const std::vector<float>& betas() const { return betas_; }
    const std::vector<float>& alphasCumprod() const { return alphasCumprod_; }

private:
    Config cfg_;
    std::vector<float> betas_;
    std::vector<float> alphasCumprod_;
    std::vector<int>   timesteps_;
    // Ring buffer of the last two x_0 predictions for the multistep update.
    std::vector<std::vector<float>> x0History_;

    void buildBetaSchedule();
};

}  // namespace imagegen
