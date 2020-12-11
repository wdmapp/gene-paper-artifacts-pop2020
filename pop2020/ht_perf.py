#!/usr/bin/env python3
'''
Script to extract preformance data output by the HR perf module for GENE and
optionally plot it using matplotlib. For each performance datum (e.g. ncalls,
time), values from each process are kept so distribution properties can be
calculated.

If passed a single file to parse, will print detailed performance tree data
to stdout. This includes average values and range for each datum displayed,
and the labels are shown indented based on the tree structure.

If passed two files to parse, will generate comparison to stdout and generate
a bar plot.
'''

import re
import statistics
from collections import OrderedDict, defaultdict
import os
import sys
import itertools
import glob
import argparse
import csv

try:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    import numpy as np
except ImportError:
    plt = None

_start_str = '*********** performance results ***********'
_csv_start_str = 'timer_name,id,'
_header_end_str = '-------------------------------------------'

_compare2_fmt = ('{0.name:16s}  {0.ncalls:7d} {1.ncalls:7d}  '
                 '{0.time:8.2f} {1.time:8.2f}  '
                 '{0.percent_t_loop:5.1f} {1.percent_t_loop:5.1f}')

_compare2_tree_fmt = ('{2:32s} {0.ncalls.mean:7.0f} {1.ncalls.mean:7.0f}  '
                      '{0.time.mean:8.2f} {1.time.mean:8.2f}  '
                      '{0.percent_t_loop.mean:7.1f} '
                      '{1.percent_t_loop.mean:7.1f}  '
                      '{0.percent_parent.mean:5.1f} '
                      '{1.percent_parent.mean:5.1f}')
_compare2_tree_header_fmt = '{0:32s} {1:19s} {2:17} {3:14s} {4:12s}'

_single_tree_fmt = ('{1:32s} {0.ncalls.mean:7.0f} '
                    '{0.time.mean:8.2f} '
                    '{0.percent_t_loop.mean:7.1f} {0.percent_parent.mean:5.1f}')

_single_tree_dist_header_fmt = (
 '{:32s} '
 '{:>7s} '
 '{:>8s} {:6s} {:>8s}'
 '{:>5s} '
 '{:>5s}'
)

_single_tree_dist_fmt = (
 '{1:32s} '
 '{0.ncalls.mean:7.0f} '
 '{0.time.mean:8.2f} {0.time.range:05.2f} {0.mpi_time:8.2f} '
 '{0.percent_t_loop.mean:7.1f} '
 '{0.percent_parent.mean:5.1f}'
)


_root_name = 'GENE'
_tloop_key = None
_rk_key = None
_rhs_key = None
_aux_key = None

DEFAULT_COMPARE_LABELS = None

def _set_global_keys(mangle=True):
    """Hack to allow running with unmangled names. Useful to debug whether
    the mangling is introducing an error."""
    global _tloop_key, _rk_key, _rhs_key, _aux_key, DEFAULT_COMPARE_LABELS
    if mangle:
        _tloop_key = _root_name + '.gsub.t_loop'
    else:
        _tloop_key = _root_name + '.gsub.timeloop.t_loop'
    _rk_key = _tloop_key + '.RK_standard'
    _rhs_key = _rk_key + '.CalFRhs0'
    _aux_key = _rk_key + '.calcaux'
    DEFAULT_COMPARE_LABELS = [_rk_key, _rhs_key, _aux_key]


def get_nml_box(nml):
    dim_abbr = OrderedDict([('n_spec', 's'),
                            ('nv0', 'v'),
                            ('nw0', 'w'),
                            ('nx0', 'x'),
                            ('nky0', 'y'),
                            ('nz0', 'z')])
    nml_box = nml['box']
    result = []
    for dim in dim_abbr.keys():
        result.append(dim_abbr[dim] + str(nml_box[dim]))
    return ' '.join(result)


def get_box_par_dict(s):
    parts = s.split(" ")
    return dict((p[0], int(p[1:])) for p in parts)


class Run(object):
    def __init__(self, stats, data, stdout_path):
        self.stats = stats
        self.data = data
        self.runpath = os.path.basename(os.path.dirname(
                            os.path.dirname(stdout_path)))
        self.tag = self.runpath.split('_')[-1]
        self.name = get_runname(stdout_path)
        self.par = stats['parallelization']
        self.box, _ = get_run_box_par(stdout_path)
        if self.box is None:
            import f90nml
            param_fpath = os.path.join(os.path.dirname(stdout_path),
                                       "out", "parameters.dat")
            self.box = get_nml_box(f90nml.read(param_fpath))
        self._unique_name = None

    @property
    def nodes(self):
        return self.stats['nodes']

    @property
    def nmpi(self):
        return self.stats['nmpi']

    @property
    def ppn(self):
        return self.stats['ppn']

    @property
    def is_gpu(self):
        return (float(self.stats['gpu_mb_freed']) > 0
                or int(self.stats['ngpu']) > 0)

    @property
    def unique_name(self):
        if self._unique_name is None:
            par_no_space = self.par.replace(' ', '')
            if self.is_gpu:
                gpu_tag = 'gpu'
            else:
                gpu_tag = 'cpu'
            self._unique_name =  \
                "{0.nodes:d}nodes_{0.ppn:d}ppn_{1:s}_{2:s}".format(
                                 self, par_no_space, gpu_tag)
        return self._unique_name

    def box_par_diff(self, other):
        box_ratios = {}
        par_ratios = {}
        sbox_map = get_box_par_dict(self.box)
        obox_map = get_box_par_dict(other.box)
        spar_map = get_box_par_dict(self.par)
        opar_map = get_box_par_dict(other.par)
        for dim in ('x', 'y', 'z', 'v', 'w', 's'):
            if sbox_map[dim] != obox_map[dim]:
                box_ratios[dim] = sbox_map[dim] / obox_map[dim]
            if spar_map[dim] != opar_map[dim]:
                par_ratios[dim] = spar_map[dim] / opar_map[dim]
        return box_ratios, par_ratios

    def __getitem__(self, key):
        """Hack for old code expecting a tuple of (stats, data)"""
        if key == 0:
            return self.stats
        elif key == 1:
            return self.data
        raise KeyError()


class Distribution(object):
    def __init__(self, value=None):
        self.values = []
        self.merge_counts = []
        self.mean = 0
        self.n = 0
        self.min = None
        self.max = None
        self._stdev = None
        if value is not None:
            self.add(value)

    def scale(self, factor):
        self.values = [v*factor for v in self.values]
        self.mean = self.mean * factor
        if self.min is not None:
            self.min = self.min * factor
        if self.max is not None:
            self.max = self.min * factor
        self._stdev = None

    def add(self, value):
        self.values.append(value)
        self.merge_counts.append(1)
        self.mean = (self.mean * self.n + value) / float(self.n + 1)
        self._stdev = None
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value
        self.n = self.n + 1

    def merge_last(self, addvalue):
        newvalue = self.values[-1] + addvalue
        self.values[-1] = newvalue
        self.merge_counts[-1] = self.merge_counts[-1] + 1
        self.mean = (self.mean * self.n + addvalue) / float(self.n)
        self._stdev = None
        if newvalue > self.max:
            self.max = newvalue
        if newvalue < self.min:
            self.min = newvalue

    @property
    def range_plus(self):
        if not self:
            return 0
        return self.max - self.mean

    @property
    def range_minus(self):
        if not self:
            return 0
        return self.mean - self.min

    @property
    def range(self):
        if not self:
            return 0
        return self.max - self.min

    def stdev(self):
        if len(self) < 2:
            return 0
        if self._stdev is None:
            self._stdev = statistics.stdev(self.values)
        return self._stdev

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]

    def __str__(self):
        return "%f %f (n=%d) [%f - %f]" % (
                self.mean, self.stdev(), self.n, self.min or 0, self.max or 0)

    def plot(self, ax, label=None, color=None, stat_lines=False):
        x = np.arange(0, len(self))
        y = np.array(self.values)
        ax.plot(x, y, label=label, color=color, marker="o", linestyle='')
        if stat_lines:
            ax.plot(x, [self.mean] * len(x), linestyle="--", label="avg")
            ax.plot(x, [self.mean + self.stdev()] * len(x),
                    linestyle="-.", label="stdev+", color='g')
            ax.plot(x, [self.mean - self.stdev()] * len(x),
                    linestyle="-.", label="stdev-", color='g')


class TimerData(object):
    def __init__(self, id_, name, parent_id=None):
        self.id = id_
        self.name = name
        self.full_name = None
        self.parent_id = parent_id
        self.ncalls = Distribution()
        self.time = Distribution()
        self.mpi_time = 0.0
        self.percent = Distribution()
        self.percent_t_loop = Distribution()
        self.percent_parent = Distribution()
        self.parent = None
        self.children = {}

    def scale(self, factor):
        self.time.scale(factor)
        self.mpi_time *= factor

    # based on averages
    def set_mpi_time(self, mpi_time):
        self.mpi_time = mpi_time

    def add_values(self, ncalls, time, percent):
        self.ncalls.add(ncalls)
        # For current minigene, ncalls should be consistent for all
        # MPI processes and all runs with same parameters.
        # Note: this doesn't hold when merging entries from spliced
        # labels, where the values are temporarily different until
        # all the values have been merged.
        #assert self.ncalls.min == self.ncalls.max, \
        #    'ncalls %d != %d' % (self.ncalls[0], ncalls)
        self.time.add(time)
        self.percent.add(percent)

    def merge_values(self, ncalls, time, percent):
        """When merging values, add to current sum rather than creating
        a new value in the distribution."""
        self.ncalls.merge_last(ncalls)
        self.time.merge_last(time)
        self.percent.merge_last(percent)

    def add_child(self, node):
        self.children[node.name] = node

    def remove_child(self, node):
        del self.children[node.name]


ZERO_TIMER = TimerData(0, 'zero')


def smart_open(file_path, mode='rt'):
    if file_path.endswith('.xz'):
        import lzma
        return lzma.open(file_path, mode)
    elif file_path.endswith('.bz2'):
        import bz2
        return bz2.open(file_path, mode)
    elif file_path.endswith('.gz'):
        import gzip
        return gzip.open(file_path, mode)
    else:
        return open(file_path, mode)


_prepended_re = re.compile(r'^[0-9]: *[0-9]: *')
def _remove_prepend(line):
    """Remove the line prefix from using 'jsrun -e prepended' on summit."""
    return _prepended_re.sub('', line)


def _parse_stats(f):
    stats = dict(
        nmpi=0,
        ngpu=0,
        nodes=0,
        cpu_mb_per_rank=0,
        gpu_mb_per_rank=0,
        gpu_mb_freed=0,
        nblocks=0,
        parallelization=0,
        ntimesteps=0,
        time_iv=0,
        time_step=0,
        time_simulation=0,
        time_wall=0,
    )
    for line in f:
        line = _remove_prepend(line)
        m = re.match('We have *(\d+) MPI tasks', line)
        if (m):
            stats['nmpi'] = int(m.group(1))
            continue
        m = re.match('We have *(\d+) MPI processes per node', line)
        if (m):
            stats['ppn'] = int(m.group(1))
            continue
        m = re.match('We have *(\d+) (CUDA|GPU) device', line)
        if (m):
            stats['ngpu'] = int(m.group(1))
            continue
        m = re.match('Using a maximum of *([0-9.]+) MB per core.', line)
        if (m):
            stats['cpu_mb_per_rank'] = float(m.group(1))
            continue
        m = re.match('Using a maximum of *([0-9.]+) MB GPU memory per rank.',
                     line)
        if (m):
            stats['gpu_mb_per_rank'] = float(m.group(1))
            continue
        m = re.match('Choice for parallelization: *(.*)', line)
        if (m):
            ps = m.group(1).split()
            labels = 's v w x y z'.split()
            ps2 = ' '.join(''.join(pair) for pair in zip(labels, ps))
            stats['parallelization'] = ps2
            continue
        m = re.match('nblocks = *(\d+)$', line)
        if (m):
            stats['nblocks'] = int(m.group(1))
            continue
        else:
            m = re.match('Choice for number of blocks: *(\d+)$', line)
            if (m):
                stats['nblocks'] = int(m.group(1))
                continue
        m = re.match('GPU memory freed in.* ([.0-9]+)MB', line)
        if (m):
            stats['gpu_mb_freed'] = int(m.group(1))
            continue
        m = re.match(
             '^Total time of initial value computation: *([.0-9]+) sec',
             line)
        if (m):
            stats['time_iv'] = m.group(1)
            continue
        m = re.match('Computed *(\d+) time steps', line)
        if (m):
            stats['ntimesteps'] = int(m.group(1))
            continue
        m = re.match('Time per time step: *([.0-9]+) sec', line)
        if (m):
            stats['time_step'] = m.group(1)
            continue
        m = re.match('Time for GENE simulation: *([.0-9]+) sec', line)
        if (m):
            stats['time_simulation'] = m.group(1)
            continue
        m = re.match('^Total wallclock time for GENE: *([.0-9]+) sec', line)
        if (m):
            stats['time_wall'] = m.group(1)
            break
    if 'ppn' not in stats:
        # hack for summit, master branch has no ppn output
        stats['ppn'] = 42
    stats['nodes'] = int(stats['nmpi'] / stats['ppn'])
    return stats


def _get_run_label(stats):
    is_gpu = float(stats['gpu_mb_freed']) > 0 or int(stats['ngpu']) > 0
    label = 'nmpi {0[nmpi]} ({0[parallelization]})'.format(stats)
    if is_gpu:
        label += ' w/GPU'
    else:
        label += ' noGPU'
    return label


def _is_mpi_region_wrapped(name):
    return (   name.startswith('ex_')
            or name.startswith('mpi_'))


def _is_mpi_region_nowrap(name):
    return (   name.startswith('ex_')
            or name == 'sum_vwsp'
            or name == 'fldgathw'
            or name == 'fldgathi')


def _add_mpi_times(root, is_mpi_region=_is_mpi_region_nowrap):
    total = 0
    if is_mpi_region(root.name):
        total += root.time.mean
    for c in root.children.values():
        total = total + _add_mpi_times(c)
    root.set_mpi_time(total)
    return total

SPLICE_LABELS = set(['timeloop',
                     'eRK_stage_1', 'eRK_stage_2',
                     'eRK_stage_3', 'eRK_stage_4'])


def _read_text_format(f, data_byid, args, rename_labels, splice_ids):
    header = False
    for line in f:
        line = _remove_prepend(line.strip())
        if line.startswith(_header_end_str):
            header = True
            break
    if not header:
        raise ValueError('no performance data header found')
    for line in f:
        line = _remove_prepend(line.strip())
        if not line:
            # performance table goes on until empty line
            break
        # Note: name may contain spaces
        parts = re.split("   *", line, 1)
        name = parts[0]
        # remaining fields have no spaces and may be only 1 space apart
        parts = parts[1].split()
        if len(parts) != 5:
            raise ValueError("expected 5 parts, got %d, line='%s'"
                             % (len(parts), line))
        # handle differences between minigene and master branch. For
        # consistency with old comparison plots, use minigene names.
        if not args.no_mangle:
            name = rename_labels.get(name, name)
        parent_id = int(parts[0])
        timer_id = int(parts[1])
        if '*' in parts[2]:
            ncalls     = int(parts[2].replace('*', '9'))
        else:
            ncalls     = int(parts[2])
        time           = float(parts[3])
        percent        = float(parts[4])

        # save ids so children can be re-written to skip over timeloop label
        # in the loop below (since they may come before this)
        if not args.no_mangle and name in SPLICE_LABELS:
            splice_ids[timer_id] = parent_id

        #print(name, timer_id, parent_id)
        entry = TimerData(id_=timer_id, name=name, parent_id=parent_id)
        entry.add_values(ncalls, time, percent)
        data_byid[entry.id] = entry


def _read_csv_format(f, data_byid, args, rename_labels, splice_ids):
    for line in f:
        # Note: name may contain spaces
        parts = re.split(",", line)
        if len(parts) != 5:
            raise ValueError("expected 5 columns, got %d, line='%s'"
                             % (len(parts), line))
        name = parts[0]
        # handle differences between minigene and master branch. For
        # consistency with old comparison plots, use minigene names.
        if not args.no_mangle:
            name = rename_labels.get(name, name)
        timer_id = int(parts[1])
        parent_id = int(parts[2])
        ncalls    = int(parts[3])
        time      = float(parts[4])
        percent   = 0.0

        # save ids so children can be re-written to skip over timeloop label
        # in the loop below (since they may come before this)
        if not args.no_mangle and name in SPLICE_LABELS:
            splice_ids[timer_id] = parent_id

        #print(name, timer_id, parent_id)
        entry = TimerData(id_=timer_id, name=name, parent_id=parent_id)
        entry.add_values(ncalls, time, percent)
        data_byid[entry.id] = entry

def _parse_group(f, data, args, first=False):
    data_byid = OrderedDict()
    start = False
    csv = False

    # hack to handle extra nesting in master branch perf marks,
    # and cuda refactor marks
    rename_labels = dict(ts0='ts1',
                         ts1to5='ts2to5',
                         eRK_standard='RK_standard',
                         calc_f='calc_df1',
                         ex_vp='ex_v',
                         exv6d='ex_v',
                         exz6d='ex_z',
                         exz4d='ex_z',
                         exz5d='ex_z')

    splice_ids = {}
    names_seen = set()

    for line in f:
        line = _remove_prepend(line.strip())
        if line == _start_str:
            start = True
            break
        if line.startswith(_csv_start_str):
            csv = True
            break

    if csv:
        _read_csv_format(f, data_byid, args, rename_labels, splice_ids)
    elif start:
        _read_text_format(f, data_byid, args, rename_labels, splice_ids)
    else:
        return False

    # translate ids to dotted hierarchy names, and add to main name
    # based data dict
    for id_, entry in data_byid.items():
        full_name = entry.name
        current = entry
        while current.parent_id != 0:
            # splice out spurious labels and labels we want to combine/
            # sum across
            splice_parent_id = splice_ids.get(current.parent_id)
            if splice_parent_id is not None:
                current.parent_id = splice_parent_id
            current = data_byid[current.parent_id]
            full_name = current.name + '.' + full_name
        existing_entry = data.get(full_name)
        if existing_entry is None:
            data[full_name] = entry
            entry.full_name = full_name
            names_seen.add(full_name)
        else:
            # names_seen is used to distinguish between the cases where
            # a full name is seen again because of a splice, or because
            # a new group is being parsed. In the former case, the values
            # should be merged; the the latter case, a new distribution
            # entry should be added. For example, if eRK_stage_[1-4] are
            # spliced, they should be summed into the current distribution
            # entry rather than averaged as separate distribution entries.
            if full_name in names_seen:
                existing_entry.merge_values(entry.ncalls[0], entry.time[0],
                                            entry.percent[0])
            else:
                # first time seen in this group, create new distribution
                # entry
                existing_entry.add_values(entry.ncalls[0], entry.time[0],
                                          entry.percent[0])
                names_seen.add(full_name)
            data_byid[id_] = existing_entry

    t_loop_entry = data.get('GENE.gsub.t_loop')
    if t_loop_entry:
        for name, entry in data.items():
            entry.percent_t_loop.add(100 * entry.time[-1]
                                     / t_loop_entry.time[-1])
    for name, entry in data.items():
        if entry.parent_id:
            parent = data_byid[entry.parent_id]
            if first:
                entry.parent = parent
            if parent.time and parent.time[-1]:
                entry.percent_parent.add(100 * entry.time[-1]
                                         / parent.time[-1])
            else:
                entry.percent_parent.add(0)
            if first:
                parent.add_child(entry)
    if args.mpi_wrap:
        _add_mpi_times(data['GENE'], is_mpi_region=_is_mpi_region_wrapped)
    else:
        _add_mpi_times(data['GENE'], is_mpi_region=_is_mpi_region_nowrap)
    return True


def _find_perfout_files(base_path):
    outdir = os.path.join(base_path, 'out')
    fpaths = []
    for fname in sorted(os.listdir(outdir)):
        if fname.startswith('perfout.'):
            fpaths.append(os.path.join(outdir, fname))
    return fpaths


def _scale_times(root, factor=None):
    if factor is None:
        assert root.ncalls.range == 0
        factor = 1.0 / root.ncalls.mean
    old = root.time.mean
    root.scale(factor)
    for child in root.children.values():
        _scale_times(child, factor)


def parse_results(infile_path, args, data=None):
    """Parse GENE output file, collecting information about the run
    overall (stats) and detailed timer data for each label (data).
    Returns Run objects, wtih stats and data members.

    By default, scale times by 1 / number of calls to (*)parent RK_standard,
    which corresponds to number of timesteps within that mark."""

    base_path = os.path.dirname(infile_path)
    # More recent runs write machine parsable data for each rank to
    # perfout.(rank).txt under diagdir. We assume convetion that the
    # stdout file is in the parent dir of diagdir, and diagdir is 'out'.
    perfout_files = _find_perfout_files(base_path)
    # If multiple rank data available but not requested, default to
    # rank 1 not rank 0. This was for consistency during PoP special
    # issue paper processing, because of a bug, could be set to use
    # rank 0 in future.
    if not args.all_ranks and len(perfout_files) > 1:
        perfout_files = perfout_files[1:2]

    start = False
    header = False
    first = False
    if data is None:
        data = OrderedDict()
        first = True
    more_data = True
    with smart_open(infile_path) as f:
        stats = _parse_stats(f)
        if len(perfout_files) == 0:
            while more_data:
                try:
                    more_data = _parse_group(f, data, args, first)
                except ValueError as e:
                    raise ValueError("error in file %s: %s"
                                     % (infile_path, e.args[0]))
                first = False

    for fpath in perfout_files:
        more_data = True
        with smart_open(fpath) as f:
            while more_data:
                try:
                    more_data = _parse_group(f, data, args, first)
                except ValueError as e:
                    raise ValueError("error in file %s: %s"
                                     % (fpath, e.args[0]))
                first = False

    if len(data) == 0:
        raise ValueError("no data found in file '" + infile_path + "'")

    # HACK: move bar_emf into calcaux, for master branch as of 201912
    for ts in ['ts1', 'ts2to5', '']:
        base = 'GENE.gsub.t_loop'
        if len(ts) > 0:
            base += '.' + ts
        rk = base + '.RK_standard'
        rk_bar_emf = rk + '.bar_emf'
        calcaux = rk + '.calcaux'
        aux_bar_emf = calcaux + '.bar_emf'
        base_node = data.get(base)
        if base_node is None:
            if len(ts):
                # some runs don't have separate ts1/ts2to5 marks
                continue
            else:
                raise ValueError('No base node: %s' % base)
        rk_node = data.get(rk)
        if rk_node is None:
            base_keys = ' '.join(base.children.keys())
            raise ValueError("No rk node at '%s': %s" % (rk, base_keys))
        calcaux_node = data[calcaux]
        bar_emf_node = rk_node.children.get('bar_emf')
        if bar_emf_node is not None:
            # TODO: fix other values
            calcaux_node.time.mean = (calcaux_node.time.mean + \
                                      bar_emf_node.time.mean)
            rk_node.remove_child(bar_emf_node)
            calcaux_node.add_child(bar_emf_node)
            bar_emf_node.parent = calcaux_node
            bar_emf_node.parent_id = calcaux_node.id
            bar_emf_node.full_name = aux_bar_emf
            del data[rk_bar_emf]
            data[aux_bar_emf] = bar_emf_node
        if args.per_ts:
            _scale_times(rk_node)

    # Number of distribution values in the root entry should give us
    # the number of MPI processes that printed perf data. By default,
    # only rank 0 prints.
    stats['nmpi_perf'] = data[_root_name].ncalls.n
    # Debug print imbalanced labels
    #for k in 'ga_init fld_init ini_flr cal_prem ccdens up_f'.split():
    #    print(k, data[k].time)
    return Run(stats, data, infile_path)


def compare_runs(d1, d2):
    for k in sorted(d1.keys()):
        e1 = d1[k]
        e2 = d2.get(k) or TimerData(k, 0, 0, 0, 0)
        print(_compare2_fmt.format(e1, e2))


def _merge_lists(a, b):
    """Merge two similar lists perserving order."""
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        else:
            # find if they are equal later
            try:
                b_idx_of_ai = b.index(a[i])
                for k in range(j, b_idx_of_ai):
                    out.append(b[k])
                j = b_idx_of_ai
            except ValueError:
                try:
                    a_idx_of_bj = a.index(b[j])
                except ValueError:
                    # mutually exclusive, go on to next iteration
                    out.append(a[i])
                    out.append(b[j])
                    i += 1
                    j += 1
                else:
                    for k in range(i, a_idx_of_bj):
                        out.append(a[k])
                    i = a_idx_of_bj
    out.extend(a[i:len(a)])
    out.extend(b[j:len(b)])
    return out


def compare_runs_tree(e1, e2, level=0, header=False):
    if header:
        print(_compare2_tree_header_fmt.format(
                 'Name', '# Calls', 'Time', '% t_loop', '% parent'))
    if e2 is None:
        if e1 is None:
            raise ValueError('one element must be non-None')
        e2 = TimerData(e1.id, e1.name)
    if e1 is None:
        e1 = TimerData(e2.id, e2.name)
    indented_name = ('  ' * level) + e1.name
    print(_compare2_tree_fmt.format(e1, e2, indented_name))
    # Note: the order printed by GENE can be inconsistent between runs,
    # so it's better to go alphabetically
    e1_keys = list(e1.children.keys())
    e1_keys.sort()
    e2_keys = list(e2.children.keys())
    e2_keys.sort()
    names = _merge_lists(e1_keys, e2_keys)
    for name in names:
        child1 = e1.children.get(name)
        child2 = e2.children.get(name)
        compare_runs_tree(child1, child2, level+1)


def _get_flat_names(e1):
    """Get all performance labels from TimerData hierarchy in a flat list,
    depth first pre-order."""
    names = []
    stack = [e1]
    while stack:
        current = stack.pop()
        names.append(current.full_name)
        if current.children:
            child_nodes = list(current.children.values())
            child_nodes.sort(key=lambda node: node.name)
            stack.extend(child_nodes)
    return names


PLOT_SKIP_NAMES = 'eRK_stage_1 eRK_stage_2 eRK_stage_3 eRK_stage_4'.split()
def compare_bar_chart(ax, d1, d2, label1, label2, min_percent=2):
    root = _rk_key
    tl1 = d1[root]
    tl2 = d2[root]
    means1 = []
    means2 = []
    std1 = []
    std2 = []
    names1 = _get_flat_names(tl1)
    names2 = _get_flat_names(tl2)
    names = _merge_lists(names1, names2)
    names.remove(root)

    plot_names = []

    # only plot labels within t_loop
    #names = [n for n in names if n.startswith('GENE.gsub.t_loop')]
    for name in names:
        base_name = name.split('.')[-1]
        if base_name in PLOT_SKIP_NAMES:
            continue
        e1 = d1.get(name)
        e2 = d2.get(name)
        if e2 is None:
            if e1 is None:
                raise ValueError('one element must be non-None')
            e2 = TimerData(e1.id, e1.name)
        if e1 is None:
            e1 = TimerData(e2.id, e2.name)
        if e1.percent.mean < min_percent and e2.percent.mean < min_percent:
            continue
        means1.append(e1.time.mean)
        means2.append(e2.time.mean)
        std1.append(e1.time.stdev())
        std2.append(e2.time.stdev())
        plot_names.append(name)

    # See https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    ind = 2 * np.arange(len(means1))  # the x locations for the groups
    width = 0.50  # the width of the bars
    rects1 = ax.bar(ind - width/2, means1, width, yerr=std1,
                color='SkyBlue', label=label1)
    rects2 = ax.bar(ind + width/2, means2, width, yerr=std2,
                color='IndianRed', label=label2)

    xlabels = [name.lstrip(root + '.') for name in plot_names]
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('time (s)')
    ax.set_title('GENE time in main loop by region mark')
    ax.set_xticks(ind)
    ax.set_xticklabels(xlabels, rotation=30, horizontalalignment='right')
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.legend()


def single_bar_chart(ax, d1, label1, min_percent=2, show_mpi=True):
    root = _rk_key
    tl1 = d1[root]
    means1 = []
    mpi_times1 = []
    std1 = []
    names = _get_flat_names(tl1)
    names.remove(root)

    plot_names = []

    # only plot labels within t_loop
    #names = [n for n in names if n.startswith('GENE.gsub.t_loop')]
    for name in names:
        base_name = name.split('.')[-1]
        if base_name in PLOT_SKIP_NAMES:
            continue
        e1 = d1.get(name)
        if e1.percent.mean < min_percent:
            continue
        means1.append(e1.time.mean)
        std1.append(e1.time.stdev())
        mpi_times1.append(e1.mpi_time)
        plot_names.append(name)

    # See https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    ind = 2 * np.arange(len(means1))  # the x locations for the groups
    width = 1.00  # the width of the bars
    rects1 = ax.bar(ind, means1, width, yerr=std1,
                color='SkyBlue', label=label1)

    xlabels = [name.lstrip(root + '.') for name in plot_names]
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('time (s)')
    ax.set_title('GENE time in main loop by region mark')
    ax.set_xticks(ind)
    ax.set_xticklabels(xlabels, rotation=30, horizontalalignment='right')
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.legend()

    if show_mpi:
        for i, rect in enumerate(rects1):
            mpi_rect = Rectangle(rect.get_xy(), rect.get_width(),
                                 mpi_times1[i], fill=False,
                                 color='grey', alpha=1.0, hatch='/')
            ax.add_patch(mpi_rect)


def print_hierarchy(node, level=0):
    indented_name = ('  ' * level) + node.name
    print(_single_tree_dist_fmt.format(node, indented_name))
    for child in sorted(node.children.values(),
                        key=lambda node: node.time.mean,
                        reverse=True):
        print_hierarchy(child, level+1)


def print_stats(run):
    maxlen = max(len(k) for k in run.stats.keys())
    fmt = '{:%ds}: {:>s}' % maxlen
    print(fmt.format('box', str(run.box)))
    for k in sorted(run.stats.keys()):
        print(fmt.format(k, str(run.stats[k])))


def compare_labels(runs, labels=None, ts=None, headers=None,
                   per_call=False):
    data_list = [r.data for r in runs]
    stats_list = [r.stats for r in runs]
    if labels is None:
        labels = DEFAULT_COMPARE_LABELS
    if ts is not None:
        labels = [label.replace('t_loop', 't_loop.'+ ts)
                  for label in labels]
    labels = ['ts_avg'] + labels
    rows = [label.split('.')[-1] for label in labels]
    maxrow_len = max(len(h) for h in rows)
    if headers is not None:
        header_fmt = '{:%ds}: {:>8s} ' % maxrow_len \
                   + '{:>14s}' * (len(headers)-1)
        headers = [h[-8:] for h in headers]
        print(header_fmt.format('', *headers))
    fmt = ('{:%ds}: {:8.3f} ' % maxrow_len) \
          + '{:8.3f} ({:2.1f})' * (len(data_list)-1)
    for i, label in enumerate(labels):
        if i == 0:
            values = [float(s['time_step']) for s in stats_list]
        else:
            if per_call:
                values = [d[label].time.mean / d[label].ncalls.mean
                          for d in data_list]
            else:
                values = [d[label].time.mean for d in data_list]
        fmt_data = [(v, values[0]/v) for v in values[1:]]
        print(fmt.format(rows[i], values[0],
              *(itertools.chain(*fmt_data))))


def main_multi_run(stdout_dir, args):
    """
    Process a directory containing multiple .out files of runs with the
    same parameters, and overlay load balance plots for a few perf labels
    of interest. The idea here is to determine if any load imbalances are
    consistent between runs, or if they are random.
    """
    perf_labels = 'dfdxy nonlin dw_rhs'.split()
    if plt is not None:
        fig, axs = plt.subplots(1, len(perf_labels))
    label = None
    for fname in sorted(os.listdir(stdout_dir)):
        if not (fname.endswith('.out') or fname.endswith('.txt')
                or fname.endswith('.xz')):
            continue
        fpath = os.path.join(stdout_dir, fname)
        print("File", fpath)
        run1 = parse_results(fpath, args)
        if label is None:
            label = _get_run_label(run1.stats)
            if plt is not None:
                fig.suptitle(label)
            print_stats(run1)
        root = run1.data[_root_name]
        print(_single_tree_dist_header_fmt.format(
          'name', 'ncalls', 'time', 'rng', 'mpi_time', '%tloop', '%prnt'
        ))
        print_hierarchy(root)
        if plt is not None:
            for i, label in enumerate(perf_labels):
                axs[i].set_title(label)
                if label not in run1.data:
                    continue
                run1.data[label].time.plot(axs[i])
    if plt is not None:
        plt.show()


def main_single_run(stdout_path, args):
    run1 = parse_results(stdout_path, args)
    print_stats(run1)
    main_common([run1], args)
    root = run1.data[_root_name]
    print(_single_tree_dist_header_fmt.format(
      'name', 'ncalls', 'time', 'rng', 'mpi_time', '%tloop', '%prnt'
    ))
    print_hierarchy(root)
    #_test_name = 'GENE.gsub.t_loop.RK_standard.CalFRhs0'
    #print('merge_counts', d1[_test_name].time.merge_counts)

    if not args.plot_compare:
        return
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(bottom=0.35)
    single_bar_chart(ax, run1.data, label1=_get_run_label(run1.stats),
                     show_mpi=args.mpi)
    plt.show()
    return


    # balance plot, disabled for now
    fig, axs = plt.subplots(1, 3)
    fig.suptitle(_get_run_label(run1.stats))
    perf_labels = 'dfdxy nonlin dw_rhs'.split()
    for i, label in enumerate(perf_labels):
        axs[i].set_title(label)
        if label not in run1.data:
            continue
        run1.data[label].time.plot(axs[i])
    plt.show()


def main_compare_two(stdout_path1, stdout_path2, args):
    run1 = parse_results(stdout_path1, args)
    run2 = parse_results(stdout_path2, args)
    print('Comparing "%s" and "%s"' % (stdout_path1, stdout_path2))
    compare_runs_tree(run1.data[_root_name], run2.data[_root_name],
                      header=True)
    main_compare_many([run1, run2], args)
    main_common([run1, run2], args)
    if not args.plot_compare:
        return
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(bottom=0.3)
    compare_bar_chart(ax, run1.data, run2.data,
                      label1=_get_run_label(run1.stats),
                      label2=_get_run_label(run2.stats))
    plt.show()


def main_compare_many(runs, args):
    if args.group_names:
        runnames = args.group_names
    else:
        runnames = None
    if args.group_by_nodes:
        node_counts = set([r.stats['nodes'] for r in runs])
        for nodes in sorted(node_counts):
            node_runs =    [r for r in runs
                            if r.stats['nodes'] == nodes]
            print("##", nodes, " nodes ##")
            _compare_many_tables(node_runs, per_call=args.per_call,
                                 runnames=runnames)
            print("##")
    else:
        _compare_many_tables(runs, per_call=args.per_call,
                             runnames=runnames)


def _compare_many_tables(runs, per_call=False, runnames=None):
    if runnames is None:
        runnames = [r.name for r in runs]
    print('Comparing: ', ' '.join(runnames))
    print('== ts1   ==')
    compare_labels(runs, ts='ts1', headers=runnames,
                   per_call=per_call)
    print('\n== ts2to5 ==')
    compare_labels(runs, ts='ts2to5', headers=runnames,
                   per_call=per_call)
    print('\n== ts6+  ==')
    compare_labels(runs, headers=runnames, per_call=per_call)


def by_cpu_gpu(run):
    ngpu = int(run.stats['ngpu'])
    if ngpu > 1:
        return "GPU"
    return "CPU"


def by_tag(run):
    return run.tag


def by_box(run):
    return run.box


def by_par(run):
    return run.par


def _mean(timer_data):
    return timer_data.time.mean


def _mpi_time(timer_data):
    return timer_data.mpi_time


def get_default_plot_labels(field=_mean):
    return OrderedDict([
      ('RHS',   lambda d: field(d[_rhs_key])),
      ('AUX',   lambda d: field(d[_aux_key])),
      ('Other', lambda d: field(d[_rk_key])
                        - field(d[_rhs_key])
                        - field(d[_aux_key])),
      ('Total', lambda d: field(d[_rk_key])),
    ])


def _get_plot_labels(ref_run1, ref_run2, colors, top_sub_regions=0,
                     field=_mean):
    from matplotlib import cm
    cmap = cm.RdPu
    plot_labels = get_default_plot_labels(field)
    sub_keys = []
    if top_sub_regions > 0:
        rhs_subs = _get_top_sub_regions(ref_run1.data[_rhs_key],
                                        ref_run2.data[_rhs_key],
                                        top_sub_regions, field)
        for i, name in enumerate(rhs_subs):
            label = 'RHS.' + name
            key = _rhs_key + '.' + name
            sub_keys.append(key)
            plot_labels[label] = \
                lambda d, k=key: field(d.get(k, ZERO_TIMER))
            gradiant = float(top_sub_regions - i) / top_sub_regions
            colors[label] = ((0.1, gradiant, 0.7),
                             (0.1, gradiant, 0.6))

        aux_subs = _get_top_sub_regions(ref_run1.data[_aux_key],
                                        ref_run2.data[_aux_key],
                                        top_sub_regions, field)
        for i, name in enumerate(aux_subs):
            label = 'AUX.' + name
            key = _aux_key + '.' + name
            sub_keys.append(key)
            plot_labels[label] = \
                lambda d, k=key: field(d.get(k, ZERO_TIMER))
            gradiant = float(top_sub_regions - i) / top_sub_regions
            colors[label] = ((gradiant, 0.4, 0.05),
                             (gradiant, 0.3, 0.01))
            #colors[label] = (cmap(gradiant), cmap(gradiant))

        plot_labels['Other'] = \
            lambda d: field(d[_rk_key]) \
                      - sum(field(d.get(k, ZERO_TIMER)) for k in sub_keys)
    return plot_labels, sub_keys


def _get_top_sub_regions(root1, root2, nregions, field=_mean):
    """Get a list of longest running child perf regions (by name) of the two
    root nodes. Note that if one of the roots is much slower, this will favor
    showing the regions that are slow for that root (which is good for showing
    performance improvements, but may be misleading if the perf regions are
    not the same between the runs)."""
    children_map = dict((k, field(root1.children[k]))
                        for k in root1.children.keys())
    for k in root2.children.keys():
        value = field(root2.children[k])
        oldvalue = children_map.get(k)
        if oldvalue is not None:
            if value > oldvalue:
                children_map[k] = value
        else:
            children_map[k] = value
    children = list(children_map.keys())
    children.sort(key=lambda k: children_map[k], reverse=True)
    return children[:nregions]


def _diff_dict_to_str(d):
    parts = []
    for k in sorted(d.keys()):
        parts.append("%s*%d" % (k, d[k]))
    return ','.join(parts)


def main_plot_node_bars(args):
    """Grouped by number of nodes CPU vs GPU, stacked by perf marks.
    If args.sub_regions > 0, shows the top sub regions (based on runtimes
    of the base group and the second group, with lowest node counts)
    rather than top level regions (aux fields, rhs, other)."""
    runs = [parse_results(p, args) for p in args.runs]
    args.group_by_nodes = True
    main_compare_many(runs, args)
    main_common(runs, args)

    if args.per_ts:
        if args.scale_ms:
            y_format = '{:0.1f}'
        else:
            y_format = '{:0.2f}'
        y_label = ' per timestep'
    else:
        y_format = '{:0.2f}'
        y_label = ''

    if args.scale_ms:
        # milliseconds
        scale_y = 1000
        y_format += 'ms'
        y_label = 'milliseconds' + y_label
    else:
        scale_y = 1
        y_format += 's'
        y_label = 'seconds' + y_label

    if args.group_by_tag:
        group_by = by_tag
        # hack: use tag of first run specified on command line
        base_group = runs[0].tag
    elif args.group_by_box:
        group_by = by_box
        base_group = runs[0].box
    else:
        group_by = by_cpu_gpu
        base_group = "CPU"

    # map group->node count->Run object. Retain order on command line,
    # but force base_group to be the first if it is not already.
    rmap = OrderedDict()

    node_counts = set()
    for run in runs:
        nodes = int(run.stats['nodes'])
        node_counts.add(nodes)
        group_key = group_by(run)
        if group_key not in rmap:
            rmap[group_key] = {}
        rmap[group_key][nodes] = run

    rmap.move_to_end(base_group, last=False)

    dpi = args.dpi
    if args.plot_resolution:
        plot_size = (args.plot_resolution[0]/dpi,
                     args.plot_resolution[1]/dpi)
        fig, ax = plt.subplots(figsize=plot_size, dpi=dpi)
    elif args.plot_size:
        fig, ax = plt.subplots(figsize=args.plot_size, dpi=dpi)
    else:
        fig, ax = plt.subplots(dpi=dpi)
    #plt.subplots_adjust(left=0.05, bottom=0.13, right=0.96, top=0.93,
    #                    wspace=0.20, hspace=0.20)
    mpi_color = 'aquamarine'
    mpi_fill  = False
    mpi_hatch = '/'
    mpi_alpha = 1.0

    node_counts = list(node_counts)
    node_counts.sort()
    groups = list(rmap.keys())
    groups.remove(base_group)
    groups.insert(0, base_group)
    ngroups = len(groups)

    par_deltas = [None]
    par_deltas_label = [""]
    base_runs = rmap[base_group]
    prev_run = base_runs[node_counts[0]]
    for nodes in node_counts[1:]:
        run = base_runs[nodes]
        diffs = run.box_par_diff(prev_run)
        par_deltas.append(diffs)
        s = "box: %s\npara: %s" \
          % (_diff_dict_to_str(diffs[0]), _diff_dict_to_str(diffs[1]))
        par_deltas_label.append(s)
        prev_run = run
        #print(run.stats["nodes"], par_deltas[-1])

    nbars = len(node_counts)
    ind = np.arange(nbars)
    sp = 0.035 # space between group bars for given node count
    width = (1.0 - (ngroups+3)*sp) / ngroups
    center_index_float = (ngroups-1) / 2.0
    #print(nbars, width, center_index_float)
    #print(ngroups, groups)
    offsets = [(sp+width) * (i - center_index_float) for i in range(ngroups)]
    xticks = np.zeros(nbars*ngroups)
    xlabels = []
    if args.group_names:
        group_names = args.group_names
    else:
        group_names = groups

    for i in range(nbars):
        for j in range(ngroups):
            xticks[i*ngroups + j] = ind[i] + offsets[j]
            xlabel = "%s %d" % (group_names[j], node_counts[i])
            if args.annotate_par_delta and j == 0:
                xlabel += "\n" + par_deltas_label[i]
            xlabels.append(xlabel)

    colors = dict(RHS=('tab:blue', 'tab:cyan'),
                  AUX=('tab:orange', 'gold'),
                  Other=('tab:purple', 'tab:pink'))
 
    base_run = rmap[base_group][node_counts[-1]]
    group1_run = rmap[groups[1]][node_counts[-1]]
    print('Compare ', base_run.name, group1_run.name)
    compare_runs_tree(base_run.data[_tloop_key],
                      group1_run.data[_tloop_key],
                      header=True)

    plot_labels, sub_keys = _get_plot_labels(base_run, group1_run, colors,
                                             args.sub_regions)
    mpi_plot_labels, _ = _get_plot_labels(base_run, group1_run, colors,
                                          args.sub_regions, field=_mpi_time)

    plot_keys = list(plot_labels.keys())
    plot_keys.remove('Total')
    if args.sub_regions > 0:
        plot_keys.remove('RHS')
        plot_keys.remove('AUX')
    top_label = 'Other'

    if args.csv_output:
        write_csv(runs, args.csv_output,
                  sub_regions=[k[len(_rk_key)+1:] for k in sub_keys])

    # map groups to values for that group
    label_group_rects = OrderedDict()
    label_group_speedups = OrderedDict()
    label_group_values = OrderedDict()
    label_group_values_mpi = OrderedDict()
    group_old_values = {}
    for label, fn in plot_labels.items():
        values_map = OrderedDict()
        mpi_values_map = OrderedDict()
        mpi_fn = mpi_plot_labels[label]
        for group in groups:
            values_map[group] = np.array(
                    [fn(run.data) * scale_y
                     for _, run in sorted(rmap[group].items(),
                                          key=lambda kv: kv[0])]
            )
            mpi_values_map[group] = np.array(
                    [mpi_fn(run.data) * scale_y
                     for _, run in sorted(rmap[group].items(),
                                          key=lambda kv: kv[0])]
            )
        label_group_speedups[label] = OrderedDict()
        label_group_rects[label] = OrderedDict()
        label_group_values[label] = values_map
        label_group_values_mpi[label] = mpi_values_map

        base_values = values_map[base_group]
        for group in groups:
            if group == base_group:
                continue
            label_group_speedups[label][group] = (base_values
                                                  / values_map[group])

        if label == 'Total':
            continue

        if args.sub_regions > 0 and label in ('AUX', 'RHS'):
            continue

        for i, group in enumerate(groups):
            if group not in group_old_values:
                group_old_values[group] = np.zeros(nbars)
            offset = offsets[i]
            if group == base_group:
                bar_label = label
            else:
                bar_label = None
            label_group_rects[label][group] = \
                ax.bar(ind+offset, values_map[group], width,
                       bottom=group_old_values[group],
                       color=colors[label][0],
                       label=bar_label)
            group_old_values[group] += values_map[group]

    # annotate bars with RK time, speedups for non-base groups
    for group in groups:
        rects = label_group_rects[top_label][group]
        values = group_old_values[group]
        speedups = label_group_speedups['Total'].get(group)
        speedups_rhs = label_group_speedups['RHS'].get(group)
        speedups_fld = label_group_speedups['AUX'].get(group)
        time_fontsize='small'
        speedup_fontsize='x-small'
        font_family = 'sans-serif'
        for i, rect in enumerate(rects):
            height = values[i]
            rect = rects[i]
            text_y = 3
            text_delta_y = 10
            if args.annotate_total:
                ax.annotate(y_format.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, text_y),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=time_fontsize,
                            fontfamily=font_family)
            if args.show_speedups and group != base_group:
                text_y += text_delta_y
                ax.annotate('RHS {: >4.1f}X'.format(speedups_rhs[i]),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(2, text_y),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=speedup_fontsize,
                            fontfamily=font_family)
                text_y += text_delta_y
                ax.annotate('AUX {: >4.1f}X'.format(speedups_fld[i]),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(2, text_y),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=speedup_fontsize,
                            fontfamily=font_family)
                text_y += text_delta_y
                ax.annotate('ALL {: >4.1f}X'.format(speedups[i]),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(2, text_y),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=speedup_fontsize,
                            fontfamily=font_family)

        # put times within bars
        inbar_fontsize = 'x-small'
        heights = np.zeros(nbars)
        for label in plot_keys:
            rects = label_group_rects[label][group]
            values = label_group_values[label][group]
            mpi_values = label_group_values_mpi[label][group]
            heights += values
            for i, rect in enumerate(rects):
                height = heights[i]
                xy = (rect.get_x() + rect.get_width() / 2,
                      height - rect.get_height() / 2.0)

                if args.annotate_bars:
                    ax.annotate(y_format.format(values[i]),
                                xy=xy,
                                xytext=(0, 0),
                                textcoords="offset points",
                                ha='center', va='center',
                                fontsize=inbar_fontsize,
                                fontfamily=font_family)
                if args.mpi and mpi_values[i] > (values[i] / 100.0):
                    mpi_rect = Rectangle(rect.get_xy(), rect.get_width(),
                                         mpi_values[i], fill=mpi_fill,
                                         color=mpi_color, alpha=mpi_alpha,
                                         hatch=mpi_hatch)
                    ax.add_patch(mpi_rect)


    ax.set_ylabel(y_label)
    ax.set_xlabel('nodes')
    if args.title:
        title = args.title
    else:
        title = None
    ax.set_title(title)
    ax.set_xticks(xticks)
    xtick_rotation = 0 # 20
    ax.set_xticklabels(xlabels, rotation=xtick_rotation,
                       horizontalalignment='center')
    #ax.set_xticklabels(['CPU  %d  GPU' % nodes for nodes in node_counts])
    handles, labels = ax.get_legend_handles_labels()
    if args.legend_loc:
        legend_loc = args.legend_loc
    else:
        legend_loc = 'best'
    if args.mpi:
        handles.insert(0, Patch(color=mpi_color, alpha=mpi_alpha,
                                fill=mpi_fill, hatch=mpi_hatch))
        labels.insert(0, 'MPI')
    ax.legend(reversed(handles), reversed(labels), loc=legend_loc)

    fig.tight_layout()
    plt.show()
    if args.plot_output_file:
        fpath, ext = os.path.splitext(args.plot_output_file)
        ext = ext[1:]
        fig.savefig(args.plot_output_file, format=ext)
        if ext == 'pgf':
            fig.savefig(fpath + '.pdf', format='pdf')


CSV_FIELDNAMES = "tag nodes box par nmpi ngpu_per_node ts_total ts_rhs ts_aux ts_other".split()
def get_csv_row(run, sub_regions):
    data = [run.tag, run.stats['nodes'], run.box, run.par,
            run.stats['nmpi'], run.stats['ngpu'],
            run.data[_rk_key].time.mean,
            run.data[_rhs_key].time.mean,
            run.data[_aux_key].time.mean,
            (run.data[_rk_key].time.mean
             - run.data[_rhs_key].time.mean
             - run.data[_aux_key].time.mean)]
    if sub_regions is not None:
        data += [run.data.get(_rk_key + '.' + region, ZERO_TIMER).time.mean
                 for region in sub_regions]
    return data


def write_csv(runs, outpath, sub_regions=None):
    runs = list(runs)
    runs.sort(key=lambda r: (r.tag, r.stats['nodes'], r.stats['nmpi'],
                             r.stats['ngpu']))
    fieldnames = list(CSV_FIELDNAMES)
    if sub_regions is not None:
        fieldnames += [r.replace('CalFRhs0', 'rhs').replace('calcaux', 'aux')
                       for r in sub_regions]
    with open(outpath, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fieldnames)
        for r in runs:
           csv_writer.writerow(get_csv_row(r, sub_regions))


_LATEX_TABLE_BEGIN = """\\begin{table*}[htb]
  \\centering
  \\begin{tabular}{|c|rrr|rrr|rrr|rrr|}
  \\hline
  $\\#$nodes & \\multicolumn{3}{|c|}{RHS} & \\multicolumn{3}{|c|}{AUX}& \\multicolumn{3}{|c|}{Other} & \\multicolumn{3}{|c|}{Total} \\\\
  & CPU & GPU & Speedup & CPU & GPU & Speedup & CPU & GPU & Speedup & CPU & GPU & Speedup \\\\
  \\hline
"""
_LATEX_TABLE_END = """  \hline
  \end{tabular}
  \caption{Tabular results from Summit scaling runs in Fig.~\\ref{fig:summit_scaling}. Shows time per timestep in seconds for CPU and GPU runs, and the speedup achieved on GPU.}
  \label{tab:scaling}
\end{table*}
"""

def write_latex(runs, outpath):
    """Write latex table with nodes/RHS/Aux/Other/Total, grouped in threes
    with CPU/GPU/Speedup for each region (13 cols). Currently only works
    with cpu vs gpu grouping."""
    runs = list(runs)
    runs.sort(key=lambda r: (r.stats['nodes'], r.stats['ngpu']))
    outdir, outfname = os.path.split(outpath)
    with open(outpath, 'w') as outfile:
        outfile.write(_LATEX_TABLE_BEGIN)
        for cpu, gpu in zip(runs[0::2], runs[1::2]):
            # nodes
            assert cpu.nodes == gpu.nodes
            outfile.write('  {:4d}'.format(cpu.nodes))

            #triple_fmt = '    & {:5.1f} & {:4.1f} & {:4.1f}x'
            triple_fmt = '    & {:5.3f} & {:4.3f} & {:4.1f}x'

            def _get_triple(key):
                return (cpu.data[key].time.mean,
                        gpu.data[key].time.mean,
                        cpu.data[key].time.mean / gpu.data[key].time.mean)

            rhs   = _get_triple(_rhs_key)
            aux   = _get_triple(_aux_key)
            total = _get_triple(_rk_key)
            other = [t - r - a for (t, r, a) in zip(total, rhs, aux)]
            other[2] = other[0] / other[1]

            outfile.write(triple_fmt.format(*rhs))
            outfile.write(triple_fmt.format(*aux))
            outfile.write(triple_fmt.format(*other))
            outfile.write(triple_fmt.format(*total))
            outfile.write(' \\\\\n')

            detail_outpath = '{:s}/{:04d}node_{:s}'.format(outdir, cpu.nodes,
                                                          outfname)
            write_latex_detail(cpu, gpu, detail_outpath)
        outfile.write(_LATEX_TABLE_END)

_LATEX_TABLE_DETAIL_BEGIN = """\\begin{table*}[htb]
  \\centering
  \\begin{tabular}{|l|rrr|}
  \\hline
  region & CPU & GPU & Speedup \\\\
  \\hline
"""
_LATEX_TABLE_DETAIL_END = """  \hline
  \end{tabular}
  \caption{Tabular results from Summit single node run. Shows time per timestep in seconds for CPU and GPU runs, and the speedup achieved on GPU.}
  \label{tab:single_node}
\end{table*}
"""


def write_latex_detail(run1, run2, outpath, root_name=None):
    """Write latex table comparing two runs, including all perf labels,
    with CPU/GPU/Speedup for each region (3 cols)."""
    if root_name is None:
        root_name = _rk_key
    tl1 = run1.data[root_name]
    tl2 = run2.data[root_name]

    lstrip = len(root_name) - len(tl1.name)

    names1 = _get_flat_names(tl1)
    names2 = _get_flat_names(tl2)
    names = _merge_lists(names1, names2)

    def _get_triple(key):
        v1 = run1.data[key].time.mean
        v2 = run2.data[key].time.mean
        if v2 == 0:
            return (v1, v2, 0)
        else:
            return (v1, v2, v1 / v2)

    triple_fmt = '    & {:6.2f} & {:5.2f} & {:5.2f}x'

    with open(outpath, 'w') as outfile:
        outfile.write(_LATEX_TABLE_DETAIL_BEGIN)
        for name in names:
            e1 = run1.data.get(name)
            e2 = run2.data.get(name)
            if e1 is None or e2 is None:
                continue

            triple = _get_triple(name)
            if triple[0] < 0.005 or triple[1] < 0.005:
                continue
            display_name = name[lstrip:].replace('_', '{\\_}')

            outfile.write('  {:52s}'.format(display_name))
            outfile.write(triple_fmt.format(*triple))
            outfile.write(' \\\\\n')
        outfile.write(_LATEX_TABLE_DETAIL_END)


def _write_flame_node(outfile, root, field, scale, ltrim=None):
    if root.name in SPLICE_LABELS:
        return
    if ltrim is None:
        # set trim based on first node, where only it's name is used
        full_name = root.name
        ltrim = len(root.full_name) - len(root.name)
    else:
        full_name = root.full_name.replace('.', ';')
        full_name = full_name[ltrim:]
    value = int(field(root) * scale)
    outfile.write('%s %d\n' % (full_name, value))
    for c in root.children.values():
        _write_flame_node(outfile, c, field, scale, ltrim)


def write_flamegraph(run, outpath, root_name, field=_mean,
                     scale=1000000):
    with open(outpath, 'w') as outfile:
        _write_flame_node(outfile, run.data[root_name], field, scale)


def extend_path(p):
    """Assume mkgenerun directory structure and try to find stdout file."""
    if os.path.isdir(p):
        pattern = os.path.join(p, '20*', 'stdout.txt.xz')
        stdout_paths = glob.glob(pattern)
        if not stdout_paths:
            print("stdout.txt.xz not found under '%s'" % pattern)
            sys.exit(1)
        return stdout_paths[0]
    return p


def get_runname(stdout_path):
    runpath = os.path.basename(os.path.dirname(os.path.dirname(stdout_path)))
    m = re.search('_(\d+)node_', runpath)
    nodes = 0
    if (m):
        nodes = int(m.group(1))
    tail = runpath.split('_')[-1]
    return '%s:%d' % (tail, nodes)


def _add_space(s):
    out = s[0]
    prev = s[0]
    for c in s[1:]:
        if c.isalpha() and prev.isdigit():
            out += " "
        out += c
        prev = c
    return out


def get_run_box_par(stdout_path):
    runpath = os.path.basename(os.path.dirname(os.path.dirname(stdout_path)))
    m = re.search('_box-([^_]+)_([^_]*)_', runpath)
    if m is None:
        return (None, None)
    else:
        box = m.group(1)
        par = m.group(2)
        return (_add_space(box), _add_space(par))


def _int_size_type(s):
    parts = s.split('x')
    if len(parts) != 2:
        raise ValueError('Size format: WxH')
    return (int(parts[0]), int(parts[1]))


def _float_size_type(s):
    parts = s.split('x')
    if len(parts) != 2:
        raise ValueError('Size format: WxH')
    return (float(parts[0]), float(parts[1]))


def get_args():
    parser = argparse.ArgumentParser(
        description='Analyze output of GENE run with HT perflib enabled')
    parser.add_argument('runs', nargs='+')
    parser.add_argument('-p', '--plot-compare', action='store_true')
    parser.add_argument('-b', '--plot-balance', action='store_true')
    parser.add_argument('-g', '--group-by-nodes', action='store_true')
    parser.add_argument('-d', '--runs-directory')
    parser.add_argument('-n', '--plot-node-bars', action='store_true')
    parser.add_argument('-t', '--group-by-tag', action='store_true')
    parser.add_argument('-x', '--group-by-box', action='store_true')
    parser.add_argument('-o', '--plot-output-file')
    parser.add_argument('-s', '--plot-size', type=_float_size_type)
    parser.add_argument('-r', '--plot-resolution', type=_int_size_type)
    parser.add_argument('-c', '--csv-output',
                        help='output csv table, for grouped runs')
    parser.add_argument('-l', '--latex-output',
                        help='output latex table, for grouped runs')
    parser.add_argument('-f', '--flame-output',
                        help='output flamegraph collapsed format per run')
    parser.add_argument('--dpi', type=int, default=80)
    parser.add_argument('--group-names', type=lambda s: s.split(','))
    parser.add_argument('--show-speedups', action='store_true')
    parser.add_argument('--title')
    parser.add_argument('--legend-loc')
    parser.add_argument('--font-size', type=int,
                        help="set global matplotlib fontsize")
    parser.add_argument('--annotate-bars', action='store_true')
    parser.add_argument('--annotate-total', action='store_true')
    parser.add_argument('--annotate-par-delta', action='store_true')
    parser.add_argument('--no-mangle', action='store_true',
                        help='no splicing or renaming of labels')
    parser.add_argument('--mpi-wrap', action='store_true',
                        help='runs use mpi_X wrapped regions')
    parser.add_argument('--per-call', action='store_true')
    parser.add_argument('-u', '--sub-regions', type=int, default=0)
    parser.add_argument('-a', '--all-ranks', action='store_true')
    parser.add_argument('--per-ts', action='store_true',
                        help='scale values to per timestep')
    parser.add_argument('--scale-ms', action='store_true',
                        help='scale values to milliseconds')
    parser.add_argument('--mpi', action='store_true',
                        help='show mpi vs non-mpi breakdown in node bars plot')

    args = parser.parse_args()
    args.runs = [extend_path(p) for p in args.runs]
    args.runnames = [get_runname(p) for p in args.runs]
    return args


def main_common(runs, args):
    """Process common args that require the list of parsed runs.
    TODO: refactor so the parsing can be done in main, this is hacky."""
    if args.csv_output and not args.plot_node_bars:
        write_csv(runs, args.csv_output)

    if args.latex_output:
        write_latex(runs, args.latex_output)

    if args.font_size:
        plt.rcParams.update({'font.size': args.font_size})

    if args.flame_output:
        for run in runs:
            fpath = args.flame_output + '.' + run.unique_name
            write_flamegraph(run, fpath, root_name=_tloop_key)


def main():
    args = get_args()

    _set_global_keys(mangle=not args.no_mangle)

    if args.plot_balance:
        main_multi_run(args.runs_directory, args)
    elif args.plot_node_bars:
        main_plot_node_bars(args)
    elif len(args.runs) == 1:
        main_single_run(args.runs[0], args)
    elif len(args.runs) == 2:
        main_compare_two(args.runs[0], args.runs[1], args)
    else:
        runs = [parse_results(path, args) for path in args.runs]
        main_compare_many(runs, args)
        main_common(runs, args)


if __name__ == '__main__':
    main()
