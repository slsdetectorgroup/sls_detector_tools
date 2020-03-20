#include "fit_tgraph.h"
#include "TF1.h"
#include "TGraph.h"
#include "TMath.h"
#include "TROOT.h"

#include <iomanip>
#include <iostream>
#include <numeric>

/* scurve model including charge sharing and a linear background */
double scurve_model(double *x, double *par) {
    double f =
        (par[0] + par[1] * x[0]) + 0.5 * (1 + TMath::Erf((x[0] - par[2]) / (sqrt(2) * par[3]))) *
                                       (par[4] + par[5] * (x[0] - par[2]));
    return f;
}

double gaus_func(double *x, double *par) {
    double f = par[0] * TMath::Gaus(x[0], par[1], par[2]);
    return f;
    // // A*np.exp(-0.5*((x-mu)/sigma)**2)
    // par[0]*TMath::Exp(-0.5*TMath::Power(x[0]-par[1]/2,2))
}

void gaus_fit(int n, double *x, double *y, double xmin, double xmax, double *result) {

    auto f = new TF1("func", "gaus", xmin, xmax);
    auto g = new TGraph(n, x, y);
    g->Fit("func", "NSQR");
    for (int i = 0; i < f->GetNpar(); ++i) {
        result[i] = f->GetParameter(i);
    }
    delete f;
    delete g;
}
void gaus_fit2(int n, double *x, double *y, double xmin, double xmax, double *result) {

    const int npar = 3;
    auto f = new TF1("func", gaus_func, xmin, xmax, npar);
    f->SetParameter(0, 100);
    f->SetParameter(1, 20);
    f->SetParameter(2, 5);
    // auto f = new TF1( "func", "gaus", xmin, xmax);
    auto g = new TGraph(n, x, y);
    g->Fit("func", "NSQR");
    for (int i = 0; i < f->GetNpar(); ++i) {
        result[i] = f->GetParameter(i);
    }
    delete f;
    delete g;
}

void fit_using_tgraph(double *data, double *x, int shape[3], double *initpar, double *result) {
    const int npar = 6;
    const int n_rows = shape[0];
    const int n_cols = shape[1];
    const int n_elements = shape[2];

    TF1 f("scurve", scurve_model, 0, 2000, npar);
    f.SetParLimits(3, 50, 300);
    f.SetParLimits(2, 0, 2000);

    for (int i = 0; i != npar; ++i) {
        if (initpar[i] == 0) {
            f.FixParameter(i, 0);
        }
    }

    for (int col = 0; col != n_cols; ++col) {
        for (int row = 0; row != n_rows; ++row) {
            f.SetParameters(initpar);
            int sum = std::accumulate(data, data + n_elements, 0);
            if (sum > 0) {
                TGraph g(n_elements, x, data);
                g.Fit(&f, "NSQR");
                f.GetParameters(result);
                result += npar;
            } else {
                for (int i = 0; i != npar; ++i) {
                    *result++ = 0;
                }
            }
            data += n_elements;
        }
    }
    return;
}

void fit_trimbits(double *data, double *x, double *target, int shape[3], double *initpar,
                  double *result) {
    const int npar = 6;
    const int n_rows = shape[0];
    const int n_cols = shape[1];
    const int n_elements = shape[2];

    // Create the function to fit, note that the range is hardcoded
    TF1 f("scurve", scurve_model, 0, 64, npar);

    // We are fitting trimbits disable charge sharing corr
    f.FixParameter(5, 0);

    // Loop over the data as a 3D array using pointers
    for (int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < n_cols; ++col) {

            auto last = data + shape[2] - 1;

            // Set initial values for the fit
            f.SetParameter(0, *data);
            f.SetParLimits(0, 0, 2 * (*data));
            f.SetParameter(1, initpar[1]);
            f.SetParameter(2, initpar[2]);
            f.SetParLimits(2, 0, 64);
            f.SetParameter(3, initpar[3]);
            f.SetParameter(4, *last);
            f.SetParLimits(4, 0, 2 * (*last));

            // Do we have any data?
            auto sum = std::accumulate(data, data + n_elements, 0.0);

            // If there is no data skip the pixel
            if (sum > 0.0) {
                TGraph g(shape[2], x, data);
                g.Fit("scurve", "NRSQ");
                f.GetParameters(result);
                result += npar;

                // Get trimbit
                *result++ = f.GetX(*target, -10., 74);
            } else {
                // Fill the output with zeros
                for (int i = 0; i < npar + 1; i++) {
                    *result++ = 0.0;
                }

            } // end else
            ++target;
            data += n_elements;
        }
    }
    return;
}
