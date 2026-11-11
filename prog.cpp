#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#ifdef _OPENMP  
#include <omp.h>
#endif


const double A1 = -3.5, B1 = 3.5;
const double A2 = -5.5, B2 = 5.5;

const double R = 3.0;

struct Solver {
    int M, N;
    double h1, h2, eps, inv_eps2;
    double *Q, *F, *B;
};

void build_coefficients(Solver &s) {
    const int M = s.M, N = s.N;
    s.h1 = (B1 - A1) / M;
    s.h2 = (B2 - A2) / N;
    double h = std::max(s.h1, s.h2);
    s.eps = h * h;
    s.inv_eps2 = 1.0 / (s.eps * s.eps);

    const int size = (M + 1) * (N + 1);
    s.Q = new double[size];
    s.F = new double[size];
    s.B = new double[size];

    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            const int index = i * (N + 1) + j;
            s.Q[index] = 0.0;
            s.F[index] = 0.0;
            s.B[index] = 0.0;
        }
    }

    const double h1 = s.h1, h2 = s.h2;
    const double cell = h1 * h2;
    double sumS = 0.0;
    int n_in = 0, n_part = 0, n_out = 0;

    #pragma omp parallel for schedule(static) reduction(+: sumS, n_in, n_part, n_out)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N; ++j) {
            const double x_i = A1 + i * h1;
            const double y_j = A2 + j * h2;
            const double x_left = x_i - 0.5 * h1, x_right = x_i + 0.5 * h1;
            const double y_lo = y_j - 0.5 * h2, y_hi = y_j + 0.5 * h2;

            double S = 0.0;
            for (int part = 0; part < 2; ++part) {
                double y_bottom, y_top, center_circle_y;
                if (part == 0) {
                    y_bottom = std::max(y_lo, 0.0); 
                    y_top = y_hi;
                    center_circle_y = 2.0;  
                } else { 
                    y_bottom = y_lo; 
                    y_top = std::min(y_hi, 0.0);
                    center_circle_y = -2.0;
                }
                if (y_top <= y_bottom) continue;

                const double corner_x[4] = {x_left, x_right, x_right, x_left};
                const double corner_y[4] = {y_bottom, y_bottom, y_top, y_top};

                double peak_x[8], peak_y[8];
                int n = 0;
                for (int k = 0; k < 4; ++k) {
                    const double begin_x = corner_x[k], begin_y = corner_y[k];
                    const double end_x = corner_x[(k + 1) % 4], end_y = corner_y[(k + 1) % 4];

                    if (begin_x * begin_x + (begin_y - center_circle_y) * (begin_y - center_circle_y) <= R * R) {
                        peak_x[n] = begin_x;
                        peak_y[n] = begin_y;
                        ++n;
                    }

                    const double d_x = end_x - begin_x, d_y = end_y - begin_y;
                    const double r_x = begin_x, r_y = begin_y - center_circle_y;
                    const double coeff_t_2 = d_x * d_x + d_y * d_y;
                    const double coeff_t_1 = d_x * r_x + d_y * r_y;
                    const double coeff_t_0 = r_x * r_x + r_y * r_y - R * R;
                    const double D1 = coeff_t_1 * coeff_t_1 - coeff_t_2 * coeff_t_0;
                    if (D1 > 0.0) {
                        const double sqrt_D1 = std::sqrt(D1);
                        const double t1 = (-coeff_t_1 - sqrt_D1) / coeff_t_2;
                        const double t2 = (-coeff_t_1 + sqrt_D1) / coeff_t_2;
                        if (t1 > 0.0 && t1 < 1.0) {
                            peak_x[n] = begin_x + t1 * d_x;
                            peak_y[n] = begin_y + t1 * d_y; ++n;
                        }
                        if (t2 > 0.0 && t2 < 1.0) {
                            peak_x[n] = begin_x + t2 * d_x;
                            peak_y[n] = begin_y + t2 * d_y; ++n;
                        }
                    }
                }

                double double_area = 0.0;
                for (int k = 0; k < n; ++k) {
                    const int k1 = (k + 1) % n;
                    double_area += (peak_x[k] - x_left) * (peak_y[k1] - y_bottom) - (peak_x[k1] - x_left) * (peak_y[k] - y_bottom);

                }
                S += 0.5 * std::fabs(double_area);
            }

            const int index = i * (N + 1) + j;
            s.Q[index] = (1.0 - S / cell) * s.inv_eps2;
            s.F[index] = S / cell;  

            double B = s.F[index] + s.Q[index] * std::sin(x_i + y_j);
            if (i == 1) {   
                B += std::sin(A1 + y_j) / (h1 * h1);
            }
            if (i == M - 1) {
                B += std::sin(B1 + y_j) / (h1 * h1);
            }
            if (j == 1) {
                B += std::sin(x_i + A2) / (h2 * h2);
            }
            if (j == N - 1) {
                B += std::sin(x_i + B2) / (h2 * h2);
            }
            s.B[index] = B;

            sumS += S;
            if (S >= cell * (1.0 - 1e-12)) {
                ++n_in;
            } else if (S <= cell * 1e-12) {
                ++n_out;
            } else {
                ++n_part;
            }
        }
    }

    const double exact = 18.0 * M_PI - (18.0 * std::acos(2.0 / 3.0) - 2.0 * std::sqrt(20.0));
    std::printf("M = %d, N = %d, h1 = %.6f, h2 = %.6f, eps = %.6e\n", M, N, h1, h2, s.eps);
    std::printf("sum S_ij = %.6f (exact area of D = %.6f, diff = %.3e)\n", sumS, exact, sumS - exact);
    std::printf("cells: inside = %d, partial = %d, outside = %d\n", n_in, n_part, n_out);
}

void solve_cg(const Solver &s, double delta, double *w) {
    const int M = s.M, N = s.N, stride = N + 1;
    const int size = (M + 1) * (N + 1);
    const double h1 = s.h1, h2 = s.h2;
    const double ih1 = 1.0 / (h1 * h1), ih2 = 1.0 / (h2 * h2);
    const double weight = h1 * h2;
    const int max_iter = (M - 1) * (N - 1);
 
    double *r = new double[size];
    double *z = new double[size];
    double *p = new double[size];
    double *Ap = new double[size];
 
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            const int index = i * stride + j;
            w[index] = 0.0;
            r[index] = 0.0;
            z[index] = 0.0;
            p[index] = 0.0;
            Ap[index] = 0.0;
        }
    }

    int iter = 0, restarts = 0;
    bool restart = true;
    bool just_restarted = false;
    double zr = 0.0;
    double J_old = 0.0;
    double diff = 0.0;
    const char *reason = "max iterations reached";
 
    while (iter < max_iter) {
        if (restart) {
            zr = 0.0;
            #pragma omp parallel for schedule(static) reduction(+: zr)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    const int index = i * stride + j;
                    const double Aw = -(w[index + stride] - 2.0 * w[index] + w[index - stride]) * ih1
                                      - (w[index + 1] - 2.0 * w[index] + w[index - 1]) * ih2
                                      + s.Q[index] * w[index];
                    r[index] = s.B[index] - Aw;
                    z[index] = r[index] / (2.0 * ih1 + 2.0 * ih2 + s.Q[index]);
                    p[index] = z[index];
                    zr += z[index] * r[index];
                }
            }
            zr *= weight;
            restart = false;
            just_restarted = true;
        }
 
        double pAp = 0.0;
        #pragma omp parallel for schedule(static) reduction(+: pAp)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                const int index = i * stride + j;
                Ap[index] = -(p[index + stride] - 2.0 * p[index] + p[index - stride]) * ih1
                            - (p[index + 1] - 2.0 * p[index] + p[index - 1]) * ih2
                            + s.Q[index] * p[index];
                pAp += Ap[index] * p[index];
            }
        }
        pAp *= weight;
        const double alpha = zr / pAp;
 
        double brw = 0.0, brw_abs = 0.0, pp = 0.0;
        #pragma omp parallel for schedule(static) reduction(+: brw, brw_abs, pp)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                const int index = i * stride + j;
                w[index] += alpha * p[index];
                r[index] -= alpha * Ap[index];
                const double t = (s.B[index] + r[index]) * w[index];
                brw += t;
                brw_abs += std::fabs(t);
                pp += p[index] * p[index];
            }
        }
        const double J = -0.5 * weight * brw;
        const double J_tol = 1e-12 * 0.5 * weight * brw_abs;
        diff = std::fabs(alpha) * std::sqrt(weight * pp);
        ++iter;
 
        if (J > J_old + J_tol) {
            #pragma omp parallel for schedule(static)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    w[i * stride + j] -= alpha * p[i * stride + j];
                }
            }
            if (just_restarted) {
                reason = "J stopped decreasing";
                break;
            }
            ++restarts;
            restart = true;
            continue;
        }
        J_old = J;
        just_restarted = false;
 
        if (diff < delta) {
            reason = "delta";
            break;
        }
 
        double zr_new = 0.0;
        #pragma omp parallel for schedule(static) reduction(+: zr_new)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                const int index = i * stride + j;
                z[index] = r[index] / (2.0 * ih1 + 2.0 * ih2 + s.Q[index]);
                zr_new += z[index] * r[index];
            }
        }
        zr_new *= weight;
        const double beta = zr_new / zr;
        zr = zr_new;
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                p[i * stride + j] = z[i * stride + j] + beta * p[i * stride + j];
            }
        }
    }
 
    for (int i = 0; i <= M; ++i) {
        const double x = A1 + i * h1;
        w[i * stride] = std::sin(x + A2);
        w[i * stride + N] = std::sin(x + B2);
    }
    for (int j = 0; j <= N; ++j) {
        const double y = A2 + j * h2;
        w[j] = std::sin(A1 + y);
        w[M * stride + j] = std::sin(B1 + y);
    }
 
    delete[] r;
    delete[] z;
    delete[] p;
    delete[] Ap;

    std::printf("Grad: iterations = %d (max %d), restarts = %d, stop: %s\n",
                iter, max_iter, restarts, reason);
}
 
double wall_time() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)std::clock() / CLOCKS_PER_SEC;
#endif
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s M N\n", argv[0]);
        return 1;
    }
 
    Solver s;
    s.M = std::atoi(argv[1]);
    s.N = std::atoi(argv[2]);
    const double delta = 1e-6;
    if (s.M < 2 || s.N < 2) {
        std::fprintf(stderr, "bad arguments\n");
        return 1;
    }
 
#ifdef _OPENMP
    std::printf("OpenMP threads = %d\n", omp_get_max_threads());
#else
    std::printf("sequential version\n");
#endif
 
    const double t0 = wall_time();
    build_coefficients(s);
    const double t1 = wall_time();
    const int size = (s.M + 1) * (s.N + 1);
    double *w = new double[size];
    solve_cg(s, delta, w);
    const double t2 = wall_time();
 
    double wmax = 0.0;
    for (size_t k = 0; k < size; ++k) {
        wmax = std::max(wmax, std::fabs(w[k]));
    }
    const int ic = s.M / 2, jc = s.N / 2;
    std::printf("max|w| = %.10f, w(%.4f, %.4f) = %.10f\n",
                wmax, A1 + ic * s.h1, A2 + jc * s.h2, w[ic * (s.N + 1) + jc]);
    std::printf("time: coefficients = %.4f s, Grad = %.4f s, total = %.4f s\n",
                t1 - t0, t2 - t1, t2 - t0);
    
    delete[] w;
    delete[] s.Q;
    delete[] s.F;
    delete[] s.B;
    return 0;
}
