/*=================================================================
 *
 *ESCALARES (int): maxiter, npart, nclass, stopmax
 *ESCALARES (double): pgrd, dexp, deltav, deltap
 *
 *VETORES: partpos, partclass, potpart, slabel, nsize
 *
 *MATRIZES: distnode, pot, nlist
 *
 *=================================================================*/
#include <math.h>
#include "mex.h"
/* to avoid the warning on rand_s when compiling with MinGW */
//extern errno_t rand_s (unsigned int *randomValue);
/* Input Arguments */
#define maxiter_IN      prhs[0]
#define npart_IN        prhs[1]
#define nclass_IN       prhs[2]
#define earlystop_IN    prhs[3]
#define stopmax_IN      prhs[4]
#define pgrd_IN         prhs[5]
#define dexp_IN         prhs[6]
#define deltav_IN       prhs[7]
#define deltap_IN       prhs[8]
#define potmin_IN       prhs[9]
#define partpos_IN      prhs[10]
#define partclass_IN    prhs[11]
#define potpart_IN      prhs[12]
#define slabel_IN       prhs[13]
#define nsize_IN        prhs[14]
#define distnode_IN     prhs[15]
#define nlist_IN        prhs[16]
#define pot_IN          prhs[17]
#define owndeg_IN       prhs[18]
#define useseed_IN      prhs[19]
#define seed_IN         prhs[20]

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[] )
{ 
    int maxiter, npart, nclass, stopmax; // escalares int
    double pgrd, dexp, deltav, deltap, potmin;  // escalares double
    unsigned int *partpos; // vetores de uint32
    unsigned short int *nsize; // vetores de uint16
    unsigned short int *slabel, *partclass; // vetores de int16
    double *potpart; // vetor de double
    unsigned int *nlist; // matrizes de int
    unsigned char *distnode;
    double *pot, *owndeg;  // matrizes de double
    int qtnode, neibmax;
    bool useseed, earlystop;
    unsigned int seed;

    /* Check for proper number of arguments */
    if (nrhs != 21) { 
        mexErrMsgTxt("21 input arguments are required.");
    } else if (nlhs > 0) {
        mexErrMsgTxt("This function no longer uses output arguments.");
    }

    maxiter   = (int) mxGetScalar(maxiter_IN);
    npart     = (int) mxGetScalar(npart_IN);
    nclass    = (int) mxGetScalar(nclass_IN);
    earlystop = (bool) mxGetScalar(earlystop_IN);
    stopmax   = (int) mxGetScalar(stopmax_IN);
    seed      = (unsigned int) mxGetScalar(seed_IN);
    pgrd      = mxGetScalar(pgrd_IN);
    dexp      = mxGetScalar(dexp_IN);
    deltav    = mxGetScalar(deltav_IN);
    deltap    = mxGetScalar(deltap_IN);
    potmin    = mxGetScalar(potmin_IN);

    partpos   = (unsigned int *)       mxGetData(partpos_IN);
    partclass = (unsigned short int *) mxGetData(partclass_IN);
    potpart   = (double *)             mxGetPr(potpart_IN);
    slabel    = (unsigned short int *) mxGetData(slabel_IN);
    nsize     = (unsigned short int *) mxGetData(nsize_IN);
    distnode  = (unsigned char *)      mxGetData(distnode_IN);
    nlist     = (unsigned int *)       mxGetData(nlist_IN);
    pot       = (double *)             mxGetPr(pot_IN);
    owndeg    = (double *)             mxGetPr(owndeg_IN);
    useseed   = (bool) mxGetScalar(useseed_IN);

    qtnode  = (int) mxGetM(slabel_IN);
    neibmax = (int) mxGetN(nlist_IN);  // quantidade máxima de vizinhos que um nó tem

    if (useseed) {
        srand(seed);
    }

    double maxmmpot = 0;
    int stopcnt = 0;

    /* aloca vetor de probabilidades */
    double *prob = (double *) malloc(sizeof(double) * neibmax);
    if (prob == NULL) {
        mexErrMsgTxt("Failed to allocate memory for probability vector 'prob'.");
    }

    for(int i = 0; i < maxiter; i++)
    {
        for(int j = 0; j < npart; j++)
        {
            double r = ((double) rand()) / RAND_MAX;
            double probsum = 0;
            int greedymov;

            if (r < pgrd) // movimento guloso
            {
                greedymov = 1;
                for(int i2 = 0; i2 < nsize[partpos[j] - 1]; i2++)
                {
                    prob[i2] = DBL_MIN
                        + pot[((partclass[j] - 1) * qtnode
                             + nlist[(qtnode * i2 + partpos[j] - 1)] - 1)]
                        * (1 / pow(1 + distnode[(j * qtnode
                             + nlist[(qtnode * i2 + partpos[j] - 1)] - 1)], dexp));
                    probsum += prob[i2];
                }
            }
            else // movimento aleatório
            {
                greedymov = 0;
                for(int i2 = 0; i2 < nsize[partpos[j] - 1]; i2++)
                    prob[i2] = 1.0;
                probsum = nsize[partpos[j] - 1];
            }

            /* roleta */
            r = ((double) rand()) * probsum / RAND_MAX;
            int k = 0;
            while (prob[k] <= r && k < nsize[partpos[j] - 1] - 1)
            {
                r -= prob[k];
                k++;
            }

            /* nó vizinho sorteado */
            k = nlist[(k * qtnode + partpos[j] - 1)];

            /* se o nó não é rotulado vamos ajustar os potenciais */
            if (slabel[k - 1] == 0)
            {
                double deltapotpartind = potpart[j] * (deltav / (nclass - 1));
                double deltapotpart    = potpart[j] * deltav;

                for(int i2 = 0; i2 < nclass; i2++)
                {
                    if (i2 == partclass[j] - 1) continue;
                    pot[(i2 * qtnode + k - 1)] -= deltapotpartind;
                    if (pot[(i2 * qtnode + k - 1)] < 0)
                    {
                        deltapotpart += pot[(i2 * qtnode + k - 1)];
                        pot[(i2 * qtnode + k - 1)] = 0;
                    }
                }
                pot[((partclass[j] - 1) * qtnode + k - 1)] += deltapotpart;
            }

            /* se foi movimento aleatório, acumula owndeg */
            if (greedymov == 0)
            {
                owndeg[((partclass[j] - 1) * qtnode + k - 1)] += potpart[j];
            }

            /* atualiza potencial da partícula */
            potpart[j] += (pot[((partclass[j] - 1) * qtnode + k - 1)]
                           - potpart[j]) * deltap;

            /* atualiza distnode se caminho melhor */
            if (distnode[(j * qtnode + partpos[j] - 1)] + 1
                < distnode[(j * qtnode + k - 1)])
            {
                distnode[(j * qtnode + k - 1)]
                    = distnode[(j * qtnode + partpos[j] - 1)] + 1;
            }

            /* choque / movimento da partícula */
            double maxpot = 0;
            for(int i2 = 0; i2 < nclass; i2++)
                if (pot[(i2 * qtnode + k - 1)] > maxpot)
                    maxpot = pot[(i2 * qtnode + k - 1)];

            if (pot[((partclass[j] - 1) * qtnode + k - 1)] >= maxpot)
                partpos[j] = k;
        }

        /* early stop */
        if (earlystop && i % 10 == 0)
        {
            double mmpot = 0;
            for(int i2 = 0; i2 < qtnode; i2++)
            {
                double mpot = 0;
                for(int i3 = 0; i3 < nclass; i3++)
                    if (pot[i3 * qtnode + i2] > mpot)
                        mpot = pot[i3 * qtnode + i2];
                mmpot += mpot;
            }
            mmpot /= qtnode;

            if (mmpot > maxmmpot)
            {
                maxmmpot = mmpot;
                stopcnt = 0;
            }
            else
            {
                stopcnt++;
                if (stopcnt > stopmax) break;
            }
        }
    }

    free(prob);
    return;
}
