#include "helper.h"
#include <math.h>


// this function is called to compute the maximum extraction size
// see Eq. (5) of the paper for more details
int max_extract_size(int num_sentences, int mode)
{
    int val;

    if(mode > 0 && mode < 100)
    {
       //interpret mode as a percentage
       val = (int)round(0.01*(float)mode*(float)num_sentences);

       //lower bound at 1
       if(val < 1)
          val = 1;
    }
    else
    {
       printf("ERROR: extraction size mode %d not recognized\n", mode);
       exit(1);
    }

    return val;
}


// this function is called to compute the normalization factor
// see Eq. (3) and Section 4.3 of paper for more details
double extract_norm(int extract_size, int mode)
{
    double val;

    switch(mode)
    {
        case 0:
            {
                val = 1;
                break;
            }
        case 1:
            {
                val = 1.0/(double)extract_size;
                break;
            }
        case 2:
            {
                val = 1.0/sqrt((double)extract_size);
                break;
            }
        default:
            {
               printf("ERROR: extraction norm mode %d is not recognized\n",mode);
               exit(1);
            }
    }

    return val;
}
