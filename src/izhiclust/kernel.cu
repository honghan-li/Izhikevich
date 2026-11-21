/* Izhikevich 2-D ODE-with-reset: fixed-step Dormand-Prince 5(4) with FSAL,
 * in-register cascaded-boxcar output stage and in-register diagnostics.
 *
 * SPEC.md sections: §2 (model), §7 (integrator), §7.1 (reset + peak
 * equalisation), §7.2 (anti-aliased output), §7.3 (embedded error estimate),
 * §7.5 (layout), §7.6 (diagnostics).
 *
 * One thread per parameter set; grid-stride.  State, coefficients and every
 * accumulator live in registers for the whole step loop.  Traces are written
 * as traces[t * n_batch + i] so consecutive threads touch consecutive
 * addresses (fully coalesced).
 */

#define NDIAG 12
/* diag[i*NDIAG + j]:
 *  0 n_spikes (analysis window)   1 isi_sum        2 isi_sumsq
 *  3 isi_min                      4 isi_max        5 first_isi
 *  6 last_isi                     7 mean_v         8 sd_v
 *  9 max_local_error             10 finite flag   11 n_spikes (whole run)
 */

/* ---- DP5 (Dormand-Prince 5(4)) tableau ------------------------------- */
#define A21 (1.0/5.0)
#define A31 (3.0/40.0)
#define A32 (9.0/40.0)
#define A41 (44.0/45.0)
#define A42 (-56.0/15.0)
#define A43 (32.0/9.0)
#define A51 (19372.0/6561.0)
#define A52 (-25360.0/2187.0)
#define A53 (64448.0/6561.0)
#define A54 (-212.0/729.0)
#define A61 (9017.0/3168.0)
#define A62 (-355.0/33.0)
#define A63 (46732.0/5247.0)
#define A64 (49.0/176.0)
#define A65 (-5103.0/18656.0)
#define B1  (35.0/384.0)
#define B3  (500.0/1113.0)
#define B4  (125.0/192.0)
#define B5  (-2187.0/6784.0)
#define B6  (11.0/84.0)
/* b - b* (5th minus embedded 4th order), for the local error estimate */
#define E1  (71.0/57600.0)
#define E3  (-71.0/16695.0)
#define E4  (71.0/1920.0)
#define E5  (-17253.0/339200.0)
#define E6  (22.0/525.0)
#define E7  (-1.0/40.0)

#define FV(v, u) (fma(fma(0.04, (v), 5.0), (v), 140.0 - (u) + I))
#define FU(v, u) (aa * (bb * (v) - (u)))

extern "C" __global__ void izhi_sweep(
    const int            n_batch,
    const double* __restrict__ par,      /* [n_batch*4]: a,b,c,d interleaved */
    const double         I,
    const double         v0,
    const double         dt,
    const long long      n_steps,
    const int            L,              /* integration steps per output sample */
    const int            out_start,      /* first stored output-sample index */
    const int            n_out,          /* number of stored output samples */
    const int            store_traces,
    const int            filter_on,      /* 1: cascaded boxcar; 0: peak-aware decimation */
    const int            store_u,
    const long long      win_start_step, /* 1-based step whose end is t_transient */
    const double         v_peak,
    const double         isi_log_lo,
    const double         isi_log_inv_span,
    const int            n_isi_bins,
    float* __restrict__  traces,
    float* __restrict__  utraces,
    float* __restrict__  spike_t,
    const int            max_spikes,
    double* __restrict__ diag,
    int* __restrict__    hist)
{
    const long long n_blocks = n_steps / (long long)L;
    const double inv_L2 = 1.0 / ((double)L * (double)L);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_batch;
         i += blockDim.x * gridDim.x) {

        const double aa = par[4 * i + 0];
        const double bb = par[4 * i + 1];
        const double cc = par[4 * i + 2];
        const double dd = par[4 * i + 3];

        double v = v0;
        double u = bb * v0;                /* paper's initialisation, u = b*v */

        /* boxcar state (§7.2): A = descending weights of this block,
         * Bacc = ascending weights, Bprev carried from the previous block.
         * out = (A + Bprev) / L^2 is exactly two cascaded length-L boxcars. */
        double A = 0.0, Bacc = 0.0, Bprev = 0.0;
        double ulast = u, vlast = v0;
        int    spiked_block = 0;

        /* diagnostics (§7.6) */
        double sum_v = 0.0, sum_v2 = 0.0;
        double isi_sum = 0.0, isi_sumsq = 0.0;
        double isi_min = 1.0e300, isi_max = -1.0e300;
        double first_isi = -1.0, last_isi = -1.0;
        double t_prev_spike = -1.0;
        double max_err = 0.0;
        long long n_win = 0, n_spk = 0, n_spk_all = 0;
        int finite_ok = 1;

        /* FSAL */
        double k1v = 0.0, k1u = 0.0;
        int have_k1 = 0;

        long long step = 0;
        for (long long k = 0; k < n_blocks; ++k) {
            A = 0.0; Bacc = 0.0; spiked_block = 0;

            for (int m = 1; m <= L; ++m) {
                ++step;

                if (!have_k1) { k1v = FV(v, u); k1u = FU(v, u); have_k1 = 1; }

                double yv = v + dt * (A21 * k1v);
                double yu = u + dt * (A21 * k1u);
                double k2v = FV(yv, yu), k2u = FU(yv, yu);

                yv = v + dt * (A31 * k1v + A32 * k2v);
                yu = u + dt * (A31 * k1u + A32 * k2u);
                double k3v = FV(yv, yu), k3u = FU(yv, yu);

                yv = v + dt * (A41 * k1v + A42 * k2v + A43 * k3v);
                yu = u + dt * (A41 * k1u + A42 * k2u + A43 * k3u);
                double k4v = FV(yv, yu), k4u = FU(yv, yu);

                yv = v + dt * (A51 * k1v + A52 * k2v + A53 * k3v + A54 * k4v);
                yu = u + dt * (A51 * k1u + A52 * k2u + A53 * k3u + A54 * k4u);
                double k5v = FV(yv, yu), k5u = FU(yv, yu);

                yv = v + dt * (A61 * k1v + A62 * k2v + A63 * k3v + A64 * k4v + A65 * k5v);
                yu = u + dt * (A61 * k1u + A62 * k2u + A63 * k3u + A64 * k4u + A65 * k5u);
                double k6v = FV(yv, yu), k6u = FU(yv, yu);

                double vn = v + dt * (B1 * k1v + B3 * k3v + B4 * k4v + B5 * k5v + B6 * k6v);
                double un = u + dt * (B1 * k1u + B3 * k3u + B4 * k4u + B5 * k5u + B6 * k6u);

                /* FSAL: k7 = f(t+h, y_{n+1}) is next step's k1 when no reset */
                double k7v = FV(vn, un), k7u = FU(vn, un);

                if (!isfinite(vn) || !isfinite(un)) { finite_ok = 0; break; }

                double vs;                                /* the emitted sample */
                if (vn >= v_peak) {
                    /* §7.1 peak equalisation, then the reset jump */
                    vs = v_peak;
                    v = cc;
                    u = un + dd;
                    have_k1 = 0;                          /* state jumped: refresh k1 */
                    spiked_block = 1;
                    ++n_spk_all;
                    double ts = (double)step * dt;
                    if (max_spikes > 0 && n_spk_all <= (long long)max_spikes)
                        spike_t[(n_spk_all - 1) * (long long)n_batch + i] = (float)ts;
                    if (step >= win_start_step) {
                        ++n_spk;
                        if (t_prev_spike >= 0.0) {
                            double isi = ts - t_prev_spike;
                            isi_sum += isi;
                            isi_sumsq += isi * isi;
                            if (isi < isi_min) isi_min = isi;
                            if (isi > isi_max) isi_max = isi;
                            if (first_isi < 0.0) first_isi = isi;
                            last_isi = isi;
                            int bin = (int)((log10(isi) - isi_log_lo) * isi_log_inv_span
                                            * (double)n_isi_bins);
                            if (bin < 0) bin = 0;
                            if (bin >= n_isi_bins) bin = n_isi_bins - 1;
                            hist[(long long)i * n_isi_bins + bin] += 1;
                        }
                        t_prev_spike = ts;
                    } else {
                        t_prev_spike = ts;   /* so the first window ISI is honest */
                    }
                } else {
                    /* §7.3: the embedded 4(5) estimate is meaningful only on
                     * steps without a reset. */
                    double ev = dt * (E1 * k1v + E3 * k3v + E4 * k4v + E5 * k5v
                                      + E6 * k6v + E7 * k7v);
                    double eu = dt * (E1 * k1u + E3 * k3u + E4 * k4u + E5 * k5u
                                      + E6 * k6u + E7 * k7u);
                    ev = fabs(ev); eu = fabs(eu);
                    double e = ev > eu ? ev : eu;
                    if (e > max_err) max_err = e;
                    vs = vn;
                    v = vn;
                    u = un;
                    k1v = k7v; k1u = k7u;                 /* FSAL */
                }

                A    += (double)(L + 1 - m) * vs;
                Bacc += (double)(m - 1) * vs;
                vlast = vs;
                ulast = u;

                if (step >= win_start_step) {
                    sum_v += vs;
                    sum_v2 += vs * vs;
                    ++n_win;
                }
            }

            if (!finite_ok) break;

            double outv = filter_on ? (A + Bprev) * inv_L2
                                    : (spiked_block ? v_peak : vlast);
            Bprev = Bacc;

            if (store_traces) {
                long long j = k - (long long)out_start;
                if (j >= 0 && j < (long long)n_out) {
                    traces[j * (long long)n_batch + i] = (float)outv;
                    if (store_u) utraces[j * (long long)n_batch + i] = (float)ulast;
                }
            }
        }

        double mean_v = n_win > 0 ? sum_v / (double)n_win : 0.0;
        double var_v = n_win > 1 ? sum_v2 / (double)n_win - mean_v * mean_v : 0.0;
        if (var_v < 0.0) var_v = 0.0;

        double* D = diag + (long long)i * NDIAG;
        D[0]  = (double)n_spk;
        D[1]  = isi_sum;
        D[2]  = isi_sumsq;
        D[3]  = (n_spk > 1) ? isi_min : 0.0;
        D[4]  = (n_spk > 1) ? isi_max : 0.0;
        D[5]  = first_isi;
        D[6]  = last_isi;
        D[7]  = mean_v;
        D[8]  = sqrt(var_v);
        D[9]  = max_err;
        D[10] = (double)finite_ok;
        D[11] = (double)n_spk_all;
    }
}
