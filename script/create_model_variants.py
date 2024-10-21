import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube


def main(args):
    dim = args.dim
    num_variants = args.num_variants
    names = args.names

    assert len(names) == dim

    # Generate maximin latin hypercube design
    lhd = LatinHypercube(dim)
    samples = pd.DataFrame(data=lhd.random(n=num_variants), columns=names)
    samples.reset_index(drop=True, inplace=True)

    # Save to file
    samples.to_csv(args.output, index=False)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create model variants")
    parser.add_argument("--dim", type=int, help="Number of dimensions", default=17)
    parser.add_argument("--num_variants", type=int, help="Number of variants", default=100_000)
    parser.add_argument("--names", type=str, nargs="+", help="Names of dimensions", default=['acure_bl_nuc','acure_ait_width','acure_cloud_ph','acure_carb_bb_diam','acure_prim_so4_diam','acure_sea_spray','acure_anth_so2_r','acure_bvoc_soa','acure_dms','acure_dry_dep_ait','acure_dry_dep_acc','acure_dry_dep_so2','acure_bc_ri','bparam','acure_autoconv_exp_nd','dbsdtbs_turb_0','a_ent_1_rp'])
    parser.add_argument("--output", type=str, help="Output file", default="model_variants.csv")

    args = parser.parse_args()
    main(args)
