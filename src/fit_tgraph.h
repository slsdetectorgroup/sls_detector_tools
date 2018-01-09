double scurve_model(double *x, double *par);
void gaus_fit(int n, double *x, double *y, double xmin, double xmax, double *result);
void fit_using_tgraph(double *data, double *x, int shape[3], double *initpar, double *result);
void fit_trimbits(double *data, double *x, double *target, int shape[3], double *initpar, double *result);
