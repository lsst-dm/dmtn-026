from __future__ import absolute_import, division, print_function
import os
import subprocess
import errno
import argparse

script_path = os.path.dirname(os.path.abspath(__file__))

cpp_header = """/*
 * LSST Data Management System
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 * See the COPYRIGHT file
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */
"""
cc_template = cpp_header + """#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

$begin_namespace$

PYBIND11_PLUGIN(_$submodule_name$) {
    py::module mod("_$submodule_name$", "Python wrapper for _$submodule_name$ library");

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}

$end_namespace$
"""

sconscript_template = """## -*- python -*-
#from lsst.sconsUtils import scripts
#scripts.BasicSConscript.pybind11([])
"""

init_template = cpp_header.replace("/*", "#").replace(" *", "#").replace("*/", "#") + """
\"\"\"modulename
\"\"\"
from __future__ import absolute_import
from .libnameLib import *
"""

lib_template = "from __future__ import absolute_import"


def create_namespace(module_name):
    modules = module_name.split('.')
    namespace = "namespace " + " {\nnamespace " .join(modules) + " {"
    close_namespace = '}'*len(modules) + ' // ' + module_name.replace('.', '::')
    return namespace, close_namespace


def get_full_path(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def create_path(path):
    """Create the path if it does not exist.

    If a new file or directory was created return True, otherwise return false
    """
    try:
        os.makedirs(path)
        print("Created new directory", path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        return False
    return True


def create_template(template, new_filename, replacements):
    """Create a template based on old_filename, in new_filename, by making replacements
    """
    new_file = template
    for old_key, new_key in replacements.items():
        new_file = new_file.replace(old_key, new_key)
    print("Creating", new_filename)
    with open(new_filename, 'w+') as f:
        f.write(new_file)


def main(pkg_path, pkg_name=None, keep=False):
    """Run the script

    Parameters
    ----------
    pkg_path
        path to package
    pkg_name
        Name of package; if None then use the last component of pkg_path
    keep
        Keep SWIG .i interface files. This can be handy for reference,
        but if you keep them then you must remember to delete them later.
    """
    if pkg_name is None:
        pkg_name = os.path.basename(pkg_path)
    split_pkg = pkg_name.split('_')

    # Expand the path to a full path
    header_path = get_full_path(os.path.join(pkg_path, 'include', 'lsst', *split_pkg))
    python_path = get_full_path(os.path.join(pkg_path, 'python', 'lsst', *split_pkg))

    # Add test file
    test_path = get_full_path(os.path.join(pkg_path, 'tests'))
    with open(os.path.join(test_path, 'test.txt'), 'w') as f:
        tests = [test for test in os.listdir(test_path) if test.startswith("test") and test.endswith(".py")]
        for test in tests:
            f.write("#tests/" + test + "\n")

    for root, dirs, files in os.walk(header_path):
        relpath = os.path.relpath(root, header_path)
        new_path = get_full_path(os.path.join(python_path, relpath))
        module = os.path.basename(new_path)
        if relpath == '.':
            module_name = '.'.join(['lsst'] + split_pkg)
        else:
            module_name = '.'.join(['lsst'] + split_pkg+relpath.split('/'))
        begin_namespace, end_namespace = create_namespace(module_name)

        # Create subdirectories if necessary
        create_path(new_path)

        # Create blank SConscripts
        scons_path = os.path.join(new_path, 'SConscript')
        create_template(
            sconscript_template,
            scons_path,
            {}
        )

        # Create __init__.py templates
        init_path = os.path.join(new_path, '__init__.py')
        create_template(
            init_template,
            init_path,
            {
                'libname': module,
                'modulename': module_name
            }
        )

        # Create Lib.py templates
        lib_path = os.path.join(new_path, '{0}Lib.py'.format(module))
        create_template(
            lib_template,
            lib_path,
            {}
        )

        # Create .cc templaces
        for f in files:
            submodule_name = '.'.join((f[0].lower() + f[1:]).split('.')[:-1])
            new_filename = submodule_name + '.cc'
            if f.endswith('.h') and not os.path.isfile(os.path.join(new_path, new_filename)):
                create_template(
                    cc_template,
                    os.path.join(new_path, new_filename),
                    {
                        '$pkg$': pkg_name,
                        '$begin_namespace$': begin_namespace,
                        '$end_namespace$': end_namespace,
                        '$submodule_name$': submodule_name
                    }
                )

        # Delete SWIG files
        if not keep:
            python_files = os.listdir(new_path)
            for f in python_files:
                if f.endswith('.i'):
                    swig_file = os.path.join(new_path, f)
                    print("Removing", swig_file)
                    subprocess.call(['rm', swig_file])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare a package for pybind11 wrapping.')
    parser.add_argument("pkg_path", help="Path to package")
    parser.add_argument("--name", help="Package name", default=None)
    parser.add_argument("--keep", action="store_true", help="Keep SWIG .i files")
    args = parser.parse_args()

    args.pkg_path = args.pkg_path.rstrip('/')
    main(pkg_path = args.pkg_path, pkg_name=args.name, keep=args.keep)
