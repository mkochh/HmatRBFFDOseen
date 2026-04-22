#include "transpose.hpp"

phmatrix transpose_hmatrix(phmatrix hm)
{
    phmatrix hmt;

    if(hm->f)
    {
        hmt = new_full_hmatrix(hm->cc, hm->rc);
        copy_amatrix(true, hm->f, hmt->f);
    }
    else if(hm->r)
    {
        hmt = new_rk_hmatrix(hm->cc, hm->rc, hm->r->k);
        copy_amatrix(false, &hm->r->A, &hmt->r->B);
        copy_amatrix(false, &hm->r->B, &hmt->r->A);
    }
    else
    {
        assert(hm->rsons > 0 && hm->csons > 0);
        
        hmt = new_super_hmatrix(hm->cc, hm->rc, hm->csons, hm->rsons);
        for(uint i = 0; i < hm->rsons; i++)
        {
            for(uint j = 0; j < hm->csons; j++)
            {
                phmatrix hm1 = transpose_hmatrix(hm->son[i + j * hm->rsons]);
                ref_hmatrix(hmt->son + j + i * hmt->rsons, hm1);
            }
        }
        update_hmatrix(hmt);
    }

    return hmt;
}