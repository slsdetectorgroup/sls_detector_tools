#include "fit_tgraph.h"
#include <iostream>
#include <iomanip>
#include "TGraph.h"
#include "TF1.h"
#include "TMath.h"
#include "Math/MinimizerOptions.h"
#include "TROOT.h"

/* scurve model including charge sharing and a linear background */
double scurve_model(double *x, double *par)
{
   double f = (par[0] + par[1]*x[0]) + 0.5 * (1+TMath::Erf( (x[0]-par[2])/(sqrt(2)*par[3]) ) ) * ( par[4] + par[5]*(x[0]-par[2]));
   return f;
}

void fit_using_tgraph(double *data, double *x, int shape[3], double *initpar, double *result){
    //Number of parameters for the fit
    int npar = 6;

    //Create the function to fit, note that the range is hardcoded
    auto f = new TF1( "scurve", scurve_model, 0, 2000, npar);
    
    // Reasonable bounds for sigma and mu
    f->SetParLimits(3,50,300);
    f->SetParLimits(2,0,2000);



    //Loop over the data as a 3D array using pointers
    for (int col = 0; col<shape[1]; col++){
        if ( (col % (shape[1]/6) == 0) && (col != 0) ){
            std::cout << "Processed: " << std::setprecision(3) << 
            static_cast<float>(col) / static_cast<float>(shape[1]) * 100 << 
            "%" << std::endl;
        }
        for (int row = 0; row<shape[0]; row++){

            //Fix parameters that are zero and set initial values for the other
            for (int i = 0; i<npar; i++){
                if (initpar[i] == 0){
                    f->FixParameter(i,0);      
                }
                else{
                    f->SetParameter(i, initpar[i]);                
                }
            }
          
            //Do we have any data?
            int sum = 0;
            for (int i = 0; i<shape[2]; i++){
                sum = sum + (data + row*shape[2] + col * shape[0]*shape[2])[i];
            }  

            //If we have data fit the pixel
            if (sum>0){
                //Create a TGraph and fit
                auto g = new TGraph(shape[2], x, data + row*shape[2] + col * shape[0]*shape[2]);
                g->Fit( "scurve",  "NSQR");
    
                //Get parameters
                for (int i = 0; i<npar; i++){
                    result[row*npar + col*shape[0]*npar + i] = f->GetParameter(i);
                }
                delete g;

            }else{
                //Fill the output with zeros
                for (int i = 0; i<npar; i++){
                    result[row*npar + col*shape[0]*npar + i] = 0;
                }
            }
        }//end row
    }//end col

    delete f;
    return;
}

void fit_trimbits(double *data, double *x, double *target, int shape[3], double *initpar, double *result){
    //Number of parameters for the fit
    int npar = 6;

    //Create the function to fit, note that the range is hardcoded
    auto f = new TF1( "scurve", scurve_model, 0, 64, npar);
    
    // We are fitting trimbits disable charge sharing corr
    f->FixParameter(5,0);

    //Loop over the data as a 3D array using pointers
    for (int col = 0; col<shape[1]; col++){
        if ( (col % (shape[1]/6) == 0) && (col != 0) ){
            std::cout << "Processed: " << std::setprecision(3) << 
            static_cast<float>(col) / static_cast<float>(shape[1]) * 100 << 
            "%" << std::endl;
        }
        for (int row = 0; row<shape[0]; row++){
        
        
            //Set initial values for the fit
            f->SetParameter(0, initpar[0]);
            f->SetParameter(1, initpar[1]);
            f->SetParameter(2, initpar[2]);
            f->SetParameter(3, initpar[3]);
            f->SetParameter(4, initpar[4]);

            //Do we have any data?
            double sum = 0;
            for (int i = 0; i<shape[2]; i++){
                sum += (data + row*shape[2] + col * shape[0]*shape[2])[i];
            }                
            if (row == 240 && col == 600){
                std::cout << "--------------" << sum << std::endl;
            }
            // If there is no data skip the pixel
            if (sum>(double)0.0){      

                //Create a TGraph and fit
                auto g = new TGraph(shape[2], x, data + row*shape[2] + col * shape[0]*shape[2]);
                g->Fit( "scurve",  "NSQ");
    
                //Get parameters
                for (int i = 0; i<npar; i++){
                    result[row*(npar+1) + col*shape[0]*(npar+1) + i] = f->GetParameter(i);
                }
                
                //Get trimbit
                result[row*(npar+1) + col*shape[0]*(npar+1) + npar] = f->GetX( target[ row + col*shape[0] ] );

                delete g;

            }else{
                //Fill the output with zeros
                for (int i = 0; i<npar; i++){
                    result[row*(npar+1) + col*shape[0]*(npar+1) + i] = (double)0;
                }
                result[row*(npar+1) + col*shape[0]*(npar+1) + npar] = (double)0;
            }//end else
        }
    }
    delete f;
    return;
}
