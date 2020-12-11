#!/usr/bin/env python3

import os
import sys
import math

import f90nml


_HL_STYLE = '\x1b[94m'
_END_STYLE = '\x1b[0m'


def _style_int(olddata, newdata, k1, k2, width=3, color=True):
    format_str = "{:%dd}" % width
    new = newdata[k1][k2]
    new_s = format_str.format(new)
    if olddata is None:
        return new_s
    else:
        old = olddata[k1][k2]
        if color and old != new:
            return _HL_STYLE + new_s + _END_STYLE
        else:
            return new_s


def _style_float(olddata, newdata, k1, k2, width=3, color=True):
    format_str = "{:%d.1f}" % width
    new = newdata[k1][k2]
    new_s = format_str.format(new)
    if olddata is None:
        return new_s
    else:
        old = olddata[k1][k2]
        if color and old != new:
            return _HL_STYLE + new_s + _END_STYLE
        else:
            return new_s


def _style_exp(olddata, newdata, k1, k2, width=3, color=True):
    format_str = "{:%d.3e}" % width
    new = newdata[k1][k2]
    new_s = format_str.format(new)
    if olddata is None:
        return new_s
    else:
        old = olddata[k1][k2]
        if color and old != new:
            return _HL_STYLE + new_s + _END_STYLE
        else:
            return new_s


def _get_run_number(fname):
    if fname.startswith('.'):
        return -1
    parts = fname.split('_')
    # use first part that is an integer
    for p in parts:
        if p.isdigit():
            return int(p)
    return 0


def _get_electrons_value(nml_data, key):
    species = nml_data['species']
    electrons = None
    for spec in species:
        if spec['name'] == 'electrons':
            electrons = spec
            break
    if electrons is None:
        return None
    return electrons[key]


def _calc_rhocheck(nml_data):
    return float(nml_data['geometry']['rhostar']) * float(nml_data['box']['lx'])


def print_table(param_dir, color=True, gpu=False):
    olddata = None
    print("nodes|        x |        y |        z |        v |        w |        s |    lx |    lv |    lw |   rhostar | rhostar * lx")
    for fname in sorted(os.listdir(param_dir), key=_get_run_number):
        if fname.startswith("."):
            continue
        fpath = os.path.join(param_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.startswith("parameters_"):
            continue
        if gpu and not fname.endswith("_gpu"):
            continue
        if not gpu and fname.endswith("_gpu"):
            continue
        name = str(_get_run_number(fname))
        data = f90nml.read(fpath)
        # 1/rhostar*sqrt(temp_electrons) should be smaller than lx
        rhocheck = _calc_rhocheck(data)
        print("{:4s} | {:s} {:s} | {:s} {:s} | {:s} {:s} | {:s} {:s} | {:s} {:s} | {:s} {:s} | {:s} | {:s} | {:s} | {:s} | {:5.3f}".format(
            name,
            _style_int(olddata, data, 'parallelization', 'n_procs_x', 3, color),
            _style_int(olddata, data, 'box', 'nx0', 4, color),
            _style_int(olddata, data, 'parallelization', 'n_procs_y', 3, color),
            _style_int(olddata, data, 'box', 'nky0', 4, color),
            _style_int(olddata, data, 'parallelization', 'n_procs_z', 3, color),
            _style_int(olddata, data, 'box', 'nz0', 4, color),
            _style_int(olddata, data, 'parallelization', 'n_procs_v', 3, color),
            _style_int(olddata, data, 'box', 'nv0', 4, color),
            _style_int(olddata, data, 'parallelization', 'n_procs_w', 3, color),
            _style_int(olddata, data, 'box', 'nw0', 4, color),
            _style_int(olddata, data, 'parallelization', 'n_procs_s', 3, color),
            _style_int(olddata, data, 'box', 'n_spec', 4, color),
            _style_float(olddata, data, 'box', 'lx', 5, color),
            _style_float(olddata, data, 'box', 'lv', 5, color),
            _style_float(olddata, data, 'box', 'lw', 5, color),
            _style_exp(olddata, data, 'geometry', 'rhostar', 5, color),
            rhocheck
            ))
        olddata = data


def main():
    color = True
    gpu = False
    args = list(sys.argv[1:])
    if '-n' in args:
        color = False
        args.remove('-n')
    if '-g' in args:
        gpu = True
        args.remove('-g')
    if (len(args) > 0):
        param_dir = args[0]
    else:
        param_dir = os.getcwd()
    print_table(param_dir, color=color, gpu=gpu)


if __name__ == '__main__':
    main()
