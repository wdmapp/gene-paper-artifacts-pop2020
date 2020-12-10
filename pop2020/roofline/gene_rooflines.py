#!/usr/bin/env python3
"""
Script to parse NCU metrics from CSV files (paths passed as args) and generate
roofline tables and plots to current working directory. See doc for
read_roof_csv for details on how to collect the data.

Assumes all computation is double precision, with no usage of Tensor Cores.

This version is for the GENE Optimization paper for PoP Special Issue 2020.

Based on NERSC roofline scripts and methodology:
https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/tree/roofline-hackathon-2020

Copyright (c) 2020 Bryce Allen <bdallen@uchicago.edu>
LICENSE: BSD 3-Clause
See also Roofline-on-NVIDIA-GPUs license:
https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/blob/roofline-hackathon-2020/LICENSE
"""

import os.path
import sys
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


G = 2**30

DISPLAY_FIELDS = ['time', 'est_ai', 'ai',
                  'est_nflops', 'nflops', 'est_flop_rate_g', 'flop_rate_g',
                  'est_nbytes', 'nbytes', 'est_mem_bw_g', 'mem_bw_g']


def main():
    # 2020 pop 1 node run, see
    # https://gitlab.mpcdf.mpg.de/g-bryceallen/parameters_2020pop/-/blob/master/parameters_ITGTEM_ECP_1_gpu
    run = Run(Domain(70, 16, 24, 48, 32, 2), Para(z=3, w=2))

    if len(sys.argv) < 2:
        print("Usage: %s csv_metric_file1 [file2....]")
        sys.exit(1)

    dfs = []
    for csv_file_path in sys.argv[1:]:
        df = parse_roof(run, csv_file_path)
        dfs.append(df)

    df_all = pd.concat(dfs)

    # Change index from the ugly 'Kernel Name' read from CSV files to
    # the friendly 'name' field added in read_roof_csv. Does an inplace
    # index set on the existing column, then re-orders so the plot
    # legend is in the desired order (not inplace).
    df_all.set_index('name', inplace=True)
    df_all = df_all.reindex(index=['dgdxy ij_deriv', 'dgdxy update',
                                   'dgdxy fused', 'dzv'])

    print(df_all[DISPLAY_FIELDS])

    latex_table('gene-rooflines_HBM.tex', df_all)
    roofline_df('gene-rooflines_HBM.pdf', df_all)


class AI(object):
    """Flop count, byte count, and calculated AI."""
    def __init__(self, nflops, nbytes):
        self.nflops = nflops
        self.nbytes = nbytes
        self.ai = float(nflops) / nbytes

    def mem_bw_g(self, time):
        return self.nbytes / time / G

    def flop_rate_g(self, time):
        return self.nflops / time / G


class Para(object):
    """GENE parallelization, in xyzvws order. Supports named
    access and index access."""

    def __init__(self, x=1, y=1, z=1, v=1, w=1, s=1):
        self.x = x
        self.y = y
        self.z = z
        self.v = v
        self.w = w
        self.s = s
        self.n = x*y*z*v*w*s

    def __getitem__(i):
        return self.tuple()(i)

    def tuple(self):
        return (self.x, self.y, self.z, self.v, self.w, self.s)


class Domain(Para):
    """GENE domain, in xyzvws order. Adds methods for deriving local
    domain and calculating AI for different kernels."""

    def get_local(self, mpi_para):
        return Domain(self.x/mpi_para.x, self.y/mpi_para.y, self.z/mpi_para.z,
                      self.v/mpi_para.v, self.w/mpi_para.w, self.s/mpi_para.s)

    def ai_dzv(self):
        # 13 point stencil -> 13*2 + 12*2 = 50 flops per point,
        # from 13 complex-real multiply and 12 complex add
        nflops = 50 * self.n
        # Read rhs domain with ghost points in z and v
        # write rhs without ghost
        # plus real 13 point stencils with no y dependence
        nbytes = (16*(self.x*self.y*(self.z+4)*(self.v+4)*self.w*self.s
                      + self.n)
                  + 8*(13*self.n/self.y))
        return AI(nflops, nbytes)

    def ai_dgdxy_ijderiv(self):
        # 5 point stencil in x -> 5*2 + 4*2 = 18 flops per point,
        # from 5 complex-real multiply and 4 complex add
        # plus y deriv with complex-complex multiply
        nflops = 18 * self.n + 6 * self.n
        # Read g with ghost points in x,
        # write x and y deriv with no ghost
        # stencil is constant
        nbytes = 16 * ((self.x+4)*self.n/self.x + 2 * self.n)
        return AI(nflops, nbytes)

    def ai_dgdxy_update(self):
        # 2 real-complex prefactor multiplies (2 flops each),
        # plus 2 complex additions (2 flops each), total 8 per point
        nflops = 8 * self.n
        # R/W rhs (full 6d)
        # read derivs (full 6d)
        # two real prefactors, no y dependence
        nbytes = 16 * (2 * self.n + 2 * self.n) + 8 * (2 * self.n / self.y)
        return AI(nflops, nbytes)

    def ai_dgdxy_fused(self):
        # 18 flops per point for x deriv 5-point stencil, plus 6 for y deriv,
        # plus 8 for update
        nflops = 32 * self.n
        # Read g with ghost points in x,
        # R/W rhs (full 6d)
        # read derivs (full 6d)
        # read two real prefactors, no y dependence
        nbytes = (16 * ((self.x+4)*self.n/self.x + 2 * self.n)
                  + 8 * (2 * self.n / self.y))
        return AI(nflops, nbytes)


class Run(object):
    """Combine global/local Domain and Parallelization"""
    def __init__(self, domain, para):
        self.g_domain = domain
        self.para = para
        self.l_domain = domain.get_local(para)


def read_roof_csv(fpath):
    """Read csv file generated from ncu with the following metrics:
    
     sm__cycles_elapsed.avg
     sm__cycles_elapsed.avg.per_second
     dram__bytes.sum

     sm__sass_thread_inst_executed_op_dfma_pred_on.sum
     sm__sass_thread_inst_executed_op_dmul_pred_on.sum
     sm__sass_thread_inst_executed_op_dadd_pred_on.sum

    and return a pandas DataFrames, with index on kernel name, and
    values for various metrics. Assumes double precision flops only.

    Given the ncu-qdrep files, this command can be used to produce the csv:
     
     ncu --import /path/to/ncu-qdrep --units base --csv > /path/to/csv

    On summit collecting inst_executed metrics results in major skew of
    the cycles elapsed metric, so they must be collected in separate runs.
    I recommend using the divisions above - dram together with the elapsed
    cycles metrics, and inst metrics in second run. The CSV files can then
    be concatenated (removing the duplicate header line), and will work
    as input to this function. No need to re-sort. It is important that
    each run uses identical parameter files - the only difference should
    be metrics collected by ncu.

    Result DataFrame has row index of the kernel names. The names can be very
    long and contain the kernel function signature. The columns are:

     * time
     * nbytes
     * mem_bw_g (memory bandwidth in GB/s)
     * nflops
     * flop_rate_g (GFLOPS/s)
     * ai

    Base units are used except where specified for the rates.

    Adapted from
    https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/blob/roofline-hackathon-2020/postprocess.py
    """
    with open(fpath) as f:
        df_raw = pd.read_csv(f, thousands=',')
        # Sum across all instances of each kernel
        df_group_sum = df_raw.groupby(['Kernel Name','Metric Name']).sum()
        df = pd.pivot_table(df_group_sum,
                            index='Kernel Name',
                            columns='Metric Name',
                            values='Metric Value')
        n_metrics = df.shape[1]
        id_counts = df_raw.groupby(['Kernel Name']).count()['ID']
        df['instances'] = id_counts.div(n_metrics)

        # this metric is a fixed value, and should not have been summed
        # across instances
        df['cycles_per_second'] = (df['sm__cycles_elapsed.avg.per_second']
                                   / df['instances'])

        df['time'] = (df['sm__cycles_elapsed.avg'] / df['cycles_per_second'] /
                      df['instances'])

        # alternate if collecting duration instead. Appears to be in
        # nano seconds even if --units base is specified.
        #df['time'] = dfmetric['gpu__time_duration.sum'] / 1.0e9

        df['nflops'] = ((
            2 * df['sm__sass_thread_inst_executed_op_dfma_pred_on.sum']
            +   df['sm__sass_thread_inst_executed_op_dmul_pred_on.sum']
            +   df['sm__sass_thread_inst_executed_op_dadd_pred_on.sum'])
          / df['instances'])

        df['nbytes'] = df['dram__bytes.sum'] / df['instances']

        df['flop_rate_g'] = df['nflops'] / df['time'] / G
        df['mem_bw_g']    = df['nbytes'] / df['time'] / G
        df['ai'] = df['nflops'] / df['nbytes']
    return df


def gene_rooflines_old():
    """Obsolete. Based on Kai's Jupyter notebook timings and estimated values,
    before having measured results from NCU."""
    labels = ['dgdxy ij_deriv', 'dgdxy update_rhs', 'dgdxy fused', 'dzv']
    time    = [1.04e-3, 1.287e-3, 1.091e-3, 1.258e-3]
    flops   = [396361728, 132120576, 528482304, 825753600]
    dram    = [830472192, 1056964608, 830472192, 651886592]
    gfps    = [381.117, 102.658, 484.402, 656.402]
    bw      = [798.531, 821.262, 761.203, 518.193]
    ai      =   [0.477, 0.125, 0.636, 1.267]
    theory_ai = [0.5, 0.125, 0.67, 1.17]

    # NB: the NERSC roofline function uses base 2, these numbers from Kai's
    # notebook are base 10
    bw   = [b*1.0e9/2**30 for b in bw]
    gfps = [f*1.0e9/2**30 for f in gfps]

    roofline('gene-rooflines', gflops, ai, None, None, labels, 'HBM')

    #latex_table('gene-rooflines_HBM.tex',
    #            labels, time, gfps, bw, ai, theory_ai)


_LATEX_TABLE_BEGIN = """\\begin{table*}[htb]
  \\centering
  \\begin{tabular}{|l|r|rr|rr|rr|}
  \\hline
  kernel & time (ms) & \\multicolumn{2}{|c|}{flop rate (G/s)}& \\multicolumn{2}{|c|}{memory rate (GB/s)} & \\multicolumn{2}{|c}{AI} \\\\
         &      & est. & mes. & est. & mes. & est. & mes. \\\\
  \\hline
"""
_LATEX_TABLE_END = """  \\hline
  \\end{tabular}
  \caption{Tabular results from Summit single node roofline analysis. See Fig.~\ref{fig:roofline_with_kernels}.}
  \\label{tab:roofline}
\\end{table*}
"""

def latex_table(outpath, df):
    with open(outpath, 'w') as outf:
        outf.write(_LATEX_TABLE_BEGIN)
        for name, row in df.iterrows():
            name = name.replace('_', '{\\_}')
            time_ms = row['time'] * 1000
            outf.write('  {:16s} & {:5.3f}'.format(name, time_ms))

            fmt = ' & {:5.1f}' * 4 + ' & {:5.3}' * 2 + ' \\\\\n'
            fields = ['est_flop_rate_g', 'flop_rate_g',
                      'est_mem_bw_g', 'mem_bw_g',
                      'est_ai', 'ai']
            outf.write(fmt.format(*row[fields].tolist()))
        outf.write(_LATEX_TABLE_END)


def roofline(outpath, FLOPS, AIHBM, AIL2=None, AIL1=None, LABELS=None, flag='HBM'):
    """
    Adapted from
    https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/blob/roofline-hackathon-2020/roofline.py
    """
    font = { 'size'   : 15}
    plt.rc('font', **font)

    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    styles = ['o','v','s','^','D',">","<","*","h","H","+","1","2","3","4","8","p","d","|","_",".",","]

    markersize = 10
    markerwidth = 2
    maxchar = 25

    if not FLOPS:
        print('FLOPS can not be empty!')
        return
    if max(FLOPS)==0:
        print('FLOPS are all 0s!')
        return
    if (not AIHBM) and (not AIL2) and (not AIL1):
        print('AIHBM, AIL2 and AIL1 can not all be empty!')
        return
    if (len(FLOPS) != len(AIHBM)):
        print('FLOPS needs to have the same length as AI!')
        return
    if (flag != 'HBM') and (flag != 'L2') and (flag != 'L1') and (flag != 'all'):
        print('flag needs to be one of HBM, L2, L1, and all!')
        return
    LABELS = [x[:maxchar] for x in LABELS]

    memRoofs = [('HBM', 828.8)] 
    cmpRoofs = [('DP FMA', 7.0689), ('DP no-FMA', 3.5358)]

    fig = plt.figure(1,figsize=(10.67,6.6))
    plt.clf()
    ax = fig.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity [FLOPs/Byte]')
    ax.set_ylabel('Performance [GFLOP/sec]')

    nx   = 10000
    xmin = -3 
    xmax = 3
    ymin = 1
    ymax = 200000

    ax.set_xlim(10**xmin, 10**xmax)
    ax.set_ylim(ymin, ymax)

    ixx = int(nx*0.02)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    scomp_x_elbow  = []
    scomp_ix_elbow = []
    smem_x_elbow   = []
    smem_ix_elbow  = []

    x = np.logspace(xmin,xmax,nx)
    for roof in cmpRoofs:
        for ix in range(1,nx):
            if float(memRoofs[0][1] * x[ix]) >= roof[1]*1024 and (memRoofs[0][1] * x[ix-1]) < roof[1]*1024:
                scomp_x_elbow.append(x[ix-1])
                scomp_ix_elbow.append(ix-1)
                break

    for roof in memRoofs:
        for ix in range(1,nx):
            if (cmpRoofs[0][1]*1024 <= roof[1] * x[ix] and cmpRoofs[0][1]*1024 > roof[1] * x[ix-1]):
                smem_x_elbow.append(x[ix-1])
                smem_ix_elbow.append(ix-1)
                break

    for i in range(len(cmpRoofs)):
        roof = cmpRoofs[i][1]*1024
        y = np.ones(len(x)) * roof
        ax.plot(x[scomp_ix_elbow[i]:],y[scomp_ix_elbow[i]:],c='k',ls='-',lw='2')

    for i in range(len(memRoofs)):
        roof = memRoofs[i][1]
        y = x * roof
        ax.plot(x[:smem_ix_elbow[i]+1],y[:smem_ix_elbow[i]+1],c='k',ls='-',lw='2')


    for i in range(len(AIHBM)):
        if flag == 'L1':
            ax.plot(float(AIL1[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[0],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
        elif flag == 'L2':
            ax.plot(float(AIL2[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[1],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
        elif flag == 'HBM':
            ax.plot(float(AIHBM[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[2],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
        elif flag == 'all':
            ax.plot(float(AIL1[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[0],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
            ax.plot(float(AIL2[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[1],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
            ax.plot(float(AIHBM[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[2],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")

    marker_handles = []  

    if flag == 'L1':
        marker_handles.append(ax.plot([],[],c='k',marker=styles[0],linestyle='None',ms=markersize,\
                markerfacecolor='none',markeredgewidth=markerwidth,label=memRoofs[0][0])[0])
    elif flag == 'L2':
        marker_handles.append(ax.plot([],[],c='k',marker=styles[1],linestyle='None',ms=markersize,\
                markerfacecolor='none',markeredgewidth=markerwidth,label=memRoofs[1][0])[0])
    elif flag == 'HBM':
        marker_handles.append(ax.plot([],[],c='k',marker=styles[2],linestyle='None',ms=markersize,\
                markerfacecolor='none',markeredgewidth=markerwidth,label=memRoofs[0][0])[0])
    elif flag == 'all':
        for i in range(len(memRoofs)):
            marker_handles.append(ax.plot([],[],c='k',marker=styles[i],linestyle='None',ms=markersize,\
                                  markerfacecolor='none',markeredgewidth=markerwidth,label=memRoofs[i][0])[0])            


    for roof in cmpRoofs:
        ax.text(x[-ixx],roof[1]*1024,
              roof[0] + ': ' + '{0:.1f}'.format(roof[1]) + ' TFLOP/s',
              horizontalalignment='right',
              verticalalignment='bottom')

    for roof in memRoofs:
        ang = np.arctan(np.log10(xlim[1]/xlim[0]) / np.log10(ylim[1]/ylim[0])
                                   * fig.get_size_inches()[1]/fig.get_size_inches()[0] )
        if x[ixx]*roof[1] >ymin:
            ax.text(x[ixx],x[ixx]*roof[1]*(1+0.25*np.sin(ang)**2),
              roof[0] + ': ' + '{0:.1f}'.format(float(roof[1])) + ' GB/s',
              horizontalalignment='left',
              verticalalignment='bottom',
              rotation=180/np.pi*ang)
        else:
            ymin_ix_elbow=list()
            ymin_x_elbow=list()
            for ix in range(1,nx):
                if (ymin <= roof[1] * x[ix] and ymin > roof[1] * x[ix-1]):
                    ymin_x_elbow.append(x[ix-1])
                    ymin_ix_elbow.append(ix-1)
                    break
            ax.text(x[ixx+ymin_ix_elbow[0]],x[ixx+ymin_ix_elbow[0]]*roof[1]*(1+0.25*np.sin(ang)**2),
              roof[0] + ': ' + '{0:.1f}'.format(float(roof[1])) + ' GB/s',
              horizontalalignment='left',
              verticalalignment='bottom',
              rotation=180/np.pi*ang)


        
    #leg1 = plt.legend(handles = marker_handles,loc='lower right', ncol=len(flag[0]) if 'all' not in flag else 3,bbox_to_anchor = (1,0))
    #ax.add_artist(leg1)

    patch_handles = list()
    for i in range(0,len(AIHBM)):
        if FLOPS[i] > 0:
            patch_handles.append(mpatches.Patch(color=colors[i%10],label = LABELS[i] if LABELS else "unknown"))

    leg2 = plt.legend(handles = patch_handles,loc=4,ncol=1,bbox_to_anchor = (1,0.1),scatterpoints = 1)

    if outpath is None:
        plt.show()
    else:
        plt.savefig(outpath)


def roofline_df(outpath, df):
    """Helper function to organize data from DataFrame(s) returned by
    parse_roof to pass to the NERSC roofline plot routine."""
    labels = df.index.tolist()
    ai = df['ai'].tolist()
    gflops  = df['flop_rate_g'].tolist()

    roofline(outpath, gflops, ai, None, None, labels, 'HBM')


def _prefix_find(names, prefix):
    for name in names:
        if name.startswith(prefix):
            return name
    return None


# Fused kernel has two derivative assigns using a 6d view of a 7d array,
# that need to be ignored.
_IGNORE_KERNELS = [
  "void gt::detail::kernel_assign_6<gt::gview<gt::gtensor_view<thrust::complex<double>, 7, gt::space::device>, 6>"
]

def parse_roof(run, csv_metric_path):
    df = read_roof_csv(csv_metric_path)

    kernels = [
        ('dgdxy ij_deriv', 'ij_deriv',  run.l_domain.ai_dgdxy_ijderiv()),
        ('dgdxy update',   'add_i_col', run.l_domain.ai_dgdxy_update()),
        ('dgdxy fused',
         'void gt::detail::kernel_assign_6<gt::gtensor_view<thrust::complex<double>, 6, gt::space::device>',
         run.l_domain.ai_dgdxy_fused()),
        ('dzv',
         'void gt::detail::kernel_assign_6<gt::gtensor_view<thrust::complex<double>, 6, gt::space::device>, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::plus, gt::gfunction<gt::ops::multiply, gt::gview<gt::gtensor_view<double, 6, gt::space::device>, 6>, gt::gview<gt::gtensor_view<thrust::complex<double>',
         run.l_domain.ai_dzv())
    ]

    for ignore_name in _IGNORE_KERNELS:
        for idx in df.index.tolist():
            if idx.startswith(ignore_name):
                df.drop(idx, inplace=True)
    #print(df)
    #print("\n".join(df.index.tolist()))

    rt = defaultdict(OrderedDict)
    df_kernels = df.index.tolist()
    for name, prefix, est_ai in kernels:
        idx = _prefix_find(df_kernels, prefix)
        if idx is None:
            continue
        time = df.at[idx, 'time']
        df.at[idx, 'name'] = name
        df.at[idx, 'est_ai'] = est_ai.ai
        df.at[idx, 'est_nflops'] = est_ai.nflops
        df.at[idx, 'est_nbytes'] = est_ai.nbytes
        df.at[idx, 'est_flop_rate_g'] = est_ai.flop_rate_g(time)
        df.at[idx, 'est_mem_bw_g'] = est_ai.mem_bw_g(time)

    return df


if __name__ == '__main__':
    main()
