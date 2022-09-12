//==============================================================
//==============================================================
// Copyright © D&I Technology
// =============================================================
#include <CL/sycl.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <string>
#include <oneapi/mkl/rng.hpp>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
#include "oneapi/tbb/parallel_reduce.h"
#include "oneapi/tbb/blocked_range.h"

using namespace sycl;
using namespace oneapi;

// Model parameters
const float kappa1 = 0.00;
const float kappa2 = 0.01;
const float sigma1 = 0.03;
const float sigma2 = 0.02;
const float rho = 0.65;

// a constant rate is used for testing. Any term structure can be used.
const float rate = 0.015;

// The lengths of the cap payment period in years, i.e. 3 months recurring payments.
const float yearFraction = 0.25;

// The strike of the cap, i.e. the owner of the cap gets payment if the rate is above the strike.
const float strike = 0.0025;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

float h(float mr, float t)
{
    if (fabs(mr) < 1e-8)
    {
        float tmp0 = mr * t;
        float tmp1 = 0.5 * tmp0 * t;
        return t - tmp1 + tmp0 * tmp1 / 3.0;
    }
    return (1.0 - exp(-mr * t)) / mr;
}

// A total variance of one factor with the speed of mean reversion *mr* and the constant volatility *volatility*.
float variance(float mr, float volatility, float t)
{
    return volatility * volatility * h(-2.0 * mr, t);
}

// A total covariance of with two speeds of mean reversion *mr1* and *mr2*, the constant volatilities *volatility1*
//  and *volatility2*, a constant correlation between factors *correlation*..
float covariance(float mr1, float mr2, float volatility1, float volatility2, float correlation, float t)
{
    return volatility1 * volatility2 * correlation * h(-(mr1 + mr2), t);
}

//**************************************************************************************************
// Demonstrate a simple cap computation using 2-factor Hull-White model.
// More sophisticated payoffs can be easily locally implemented.
//**************************************************************************************************
int main(int argc, char* argv[]) {
    // Change the number of simulations if it was passed as argument
    int n_paths = 10000;
    if (argc > 1)
        n_paths = std::stoi(argv[1]);
    // Change the number of cap periods, i.e. number of times the cash flow is paid.
    int n_times = 40;
    if (argc > 2)
        n_times = std::stoi(argv[2]);

    // Create device selector for the device of your interest.
#if FPGA_EMULATOR
    // DPC++ extension: FPGA emulator selector on systems without FPGA card.
    ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
    // DPC++ extension: FPGA selector on systems with FPGA card.
    ext::intel::fpga_selector d_selector;
#else
    // The default device selector will select the most performant device.
    //default_selector d_selector;
    // One can explicitly select the cpu selector
    //cpu_selector d_selector;
    gpu_selector d_selector;
#endif

    try {
        queue q(d_selector, exception_handler);

        // Print out the device information used for the kernel code.
        std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";
        std::cout << "Number of paths: " << n_paths << "\n";

        int n_numbers = 2 * n_times * n_paths;

        auto device = q.get_device();
        auto context = q.get_context();

        // Start time measurement
        auto start = std::chrono::steady_clock::now();

        // We use Unified Shared Memory (USM) as an alternative to buffers for managing and accessing memory from the host and device.

        // We allocate the memory for random numbers.
        float* x = (float*) cl::sycl::malloc_shared(n_numbers * sizeof(float), device, context);

        // At each iteration we need to advance from current time point to the next using the Cholesky decomposed covariance matrix
        //
        //   L11     0
        //   L21    L22
        //
        float* timelineL11 = (float*) cl::sycl::malloc_shared(n_times * sizeof(float), device, context);
        float* timelineL21 = (float*) cl::sycl::malloc_shared(n_times * sizeof(float), device, context);
        float* timelineL22 = (float*) cl::sycl::malloc_shared(n_times * sizeof(float), device, context);

        // To compute the discount factor at each time point we need the total variance, covariance, discount factors and the model parameter H
        float* timelineVariances = (float*) cl::sycl::malloc_shared(n_times * 2 * sizeof(float), device, context);
        float* timelineCovariances = (float*) cl::sycl::malloc_shared(n_times * sizeof(float), device, context);
        float* timelineHs = (float*) cl::sycl::malloc_shared((n_times + 1) * 2 * sizeof(float), device, context);
        float* timelineDfs = (float*) cl::sycl::malloc_shared((n_times + 1) * sizeof(float), device, context);

        // For each Monte Carlo trajectory we need to sum all periods' cash flows. They will be aggregated after the simulation.
        float* trajectoryPvs = (float*) cl::sycl::malloc_shared(n_paths * sizeof(float), device, context);

        // Initialize model parameters
        float T_star = yearFraction * (1 + n_times);
        float H1_star = h(kappa1, T_star);
        float H2_star = h(kappa2, T_star);
        float D_star = exp(-rate * T_star);

        float var1 = 0.0;
        float var2 = 0.0;
        float cov = 0.0;
        for (size_t iTime = 0; iTime < n_times; iTime ++)
        {
            float t = yearFraction * (1 + iTime);

            // Compute the total variance, covariance
            timelineVariances[2 * iTime] = variance(kappa1, sigma1, t);
            timelineVariances[2 * iTime + 1] = variance(kappa2, sigma2, t);
            timelineCovariances[iTime] = covariance(kappa1, kappa2, sigma1, sigma2, rho, t);

            // Imply the conditional variance, covariance
            float A11 = timelineVariances[2 * iTime] - var1;
            float A21 = timelineCovariances[iTime] - cov;
            float A22 = timelineVariances[2 * iTime + 1] - var2;

            // Perform the Cholesky decomposition
            float L11 = sqrt(A11);
            float L21 = A21 / L11;
            float L22 = sqrt(A22 - L21 * L21);

            timelineL11[iTime] = L11;
            timelineL21[iTime] = L21;
            timelineL22[iTime] = L22;

            var1 = timelineVariances[2 * iTime];
            var2 = timelineVariances[2 * iTime + 1];
            cov = timelineCovariances[iTime];

            timelineHs[2 * iTime] = h(kappa1, t);
            timelineHs[2 * iTime + 1] = h(kappa2, t);
            timelineDfs[iTime] = exp(-rate * t);
        }
        float t = yearFraction * (1 + n_times);
        timelineHs[2 * n_times] = h(kappa1, t);
        timelineHs[2 * n_times + 1] = h(kappa2, t);
        // We asume that the discount factor has the form D(t) = exp(-r x t)
        // Any other term structure of the yield curve can be used instead.
        timelineDfs[n_times] = exp(-rate * t);

        // Generate all random number at once using oneMKL
        const uint32_t SEED = 0;
        oneapi::mkl::rng::mt19937 engine(q, SEED);
        oneapi::mkl::rng::gaussian<float> distr(0.0f, 1.0f);
        auto event1 = oneapi::mkl::rng::generate(distr, engine, n_numbers, x);

        event1.wait_and_throw();

        // Using SYCL submit the job to the queue
        auto event2 = q.parallel_for(n_paths, [=](auto i) {
            int shift = 2 * n_times * i;
            float state1 = 0.0f;
            float state2 = 0.0f;
            trajectoryPvs[i] = 0.0;
            for (int iTime = 0; iTime < n_times; iTime++)
            {
                state1 += timelineL11[iTime] * x[shift + 2 * iTime];
                state2 += timelineL21[iTime] * x[shift + 2 * iTime] + timelineL22[iTime] * x[shift + 2 * iTime + 1];

                float dH1_num = H1_star - timelineHs[2 * iTime];
                float dH2_num = H2_star - timelineHs[2 * iTime + 1];
                float numeraire = D_star / timelineDfs[iTime] * exp(-dH1_num * state1 - dH2_num * state2
                                + 0.5f * (dH1_num * dH1_num * timelineVariances[2 * iTime] + dH2_num * dH2_num * timelineVariances[2 * iTime + 1]
                                          + 2.0f * dH1_num * dH2_num* timelineCovariances[iTime]));

                // we compute a single caplet. Any other payoff can be coded below.
                float D_s = 1.0;

                float dH1_e = H1_star - timelineHs[2 * (iTime + 1)];
                float dH2_e = H2_star - timelineHs[2 * (iTime + 1) + 1];
                float D_e = timelineDfs[iTime + 1] / timelineDfs[iTime] * exp((dH1_e - dH1_num) * state1 + (dH2_e - dH2_num) * state2
                          - 0.5f * ((dH1_e * dH1_e - dH1_num * dH1_num) * timelineVariances[2 * iTime] + (dH2_e * dH2_e - dH2_num * dH2_num) * timelineVariances[2 * iTime + 1]
                                     + 2.0f * (dH1_e * dH2_e - dH1_num * dH2_num) * timelineCovariances[iTime]));

                float D_p = D_e;
                float comp_rate = (D_s / D_e - 1.0f) / yearFraction;

                float coupon = yearFraction * std::max(comp_rate - strike, 0.0f);
                trajectoryPvs[i] += coupon * D_p / numeraire;
            }
        });
        event2.wait_and_throw();

        // Aggregate the final result using oneTBB
        float pv = parallel_reduce(tbb::blocked_range<float*>(trajectoryPvs, trajectoryPvs + n_paths), 0.f, [=](const tbb::blocked_range<float*>& r, float init)->float {
            for (float* a = r.begin(); a != r.end(); ++a)
                init += *a / n_paths;
            return init;
            },
            []( float x, float y )->float {
                return x+y;
            }
        );

        std::cout << "PV: " << pv * D_star << std::endl;

        // Free up the allocated memory. Pass the correct queue object.
        free(x, q);
        free(timelineL11, q);
        free(timelineL21, q);
        free(timelineL22, q);
        free(timelineVariances, q);
        free(timelineCovariances, q);
        free(timelineHs, q);
        free(timelineDfs, q);
        free(trajectoryPvs, q);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    } catch (exception const &e) {
        std::cout << "An exception is caught while running the simulation.\n";
        std::terminate();
    }

    std::cout << "The Monte Carlo simulation is successfully completed on the device.\n";
    return 0;
}

