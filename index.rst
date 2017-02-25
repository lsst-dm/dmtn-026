..
  Technote content.

:tocdepth: 1
.. Please do not modify tocdepth; will be fixed when a new Sphinx theme is shipped.

.. warning::

    This document has been superseded by the pybind11 information in the `LSST DM Developer Guide <https://developer.lsst.io>`_.

.. _scope:

Scope of the document
=====================

This document is designed to assist developers involved in porting LSST Science Pipelines
from swig to pybind11 by demonstrating how to wrap a package in pybind11.
It assumes that the reader has already read (but not necessarily completely understood)
the pybind11 documentation and the LSST coding guidelines (see :ref:`additional`).

.. _intro:

.. _additional:

Additional documentation
========================

Also see the:

* `The pybind11 upstream documentation <http://pybind11.readthedocs.io>`_ 
* `The LSST pybind11 coding guidelines <https://dmtn-024.lsst.io>`_ 

.. _installation:

Installation
============

To install all of the currently wrapped code with pybind11, use

.. code-block:: bash

    $ rebuild  -r tickets/DM-NNNN -r tickets/DM-8467 {{package name}}

where ``{{package name}}`` is the name of the package that is currently being wrapped (for instance ``afw``)
and ``NNNN`` is the ticket number for the pybind11 port of the new package.

.. note::

    If you are wrapping a new package, you will first have to prepare the package as described in
    :ref:`new_package` to create the new branch, deactivate the tests, and setup pybind11.
    Otherwise scons will fail and you won't be able to setup the new package.

This will build the most up to date version of the stack that has been ported to pybind11, 
as the tests that have not been wrapped are all commented out (see section :ref:`activate-test`).

Don't forget to tag the new build as current with EUPS:

.. code-block:: bash

    $ eups tags --clone bNNNN current

where ``bNNN`` is the current build number.

.. _new_package:

Wrapping a new package
======================

Since many packages have C++ classes and functions that are not exposed to Python, 
there are large chunks of C++ code that are not currently tested explicitly.
Without tests there is no way to know whether the wrapping was successful, 
so for now we only wrap C++ code that are called from Python tests.
A single test might import from multiple submodules of the current package, 
so we found that it is more efficient to wrap by test as opposed to wrapping by module.
The following section outlines the procedure to begin wrapping a new package.

Preparing Package
-----------------

To begin wrapping a new package you will need to create two new ticket branches in the 
packages repository. First checkout the ``master`` branch of the repository you are going to wrap and

.. code-block:: bash

    $ git checkout -b tickets/DM-8467

which creates a branch for the pybind11 master branch.
Next create a branch for the current ticket

.. code-block:: bash

    $ git checkout -b tickets/DM-NNNN

where NNNN is the ticket number.

Before you can begin wrapping a package it is necessary to modify the structure of the package,
which includes modifying the ``SConscript`` files, ``__init__.py`` files, and ``moduleLib.py`` files;
adding a C++ file for every header file in the ``include`` directory;
and removing all of the SWIG files. This is all done by the script 
`build_templates.py <https://github.com/lsst-dm/dmtn-026/blob/tickets/DM-7720/python/build_templates.py>`_.

If the name of the repository is the same as the directory name on your computer 
(for example "afw" or "meas_deblender") you can execute the script using

.. code-block:: bash

    $ python build_templates.py {{repository directory}}

where ``repository directory`` is a relative or absolute path to the location of the repository 
that is going to be wrapped, for example ``../code/afw``.

.. note::

    To use this syntax ``build_templates.py`` cannot be run from inside the repository,
    and the repo directory must have the same name as the lsst package 
    (for example you can't clone afw into a directory afw2),
    as the script uses the path to infer the name of the package.

Otherwise, if you don't want the code to infer the package name, use the command

.. code-block:: bash

    $ python build_templates.py {{repository directory}} {{package name}}

where ``package name`` is the name of the package.

This step is only necessary if you are the first developer wrapping a new package,
otherwise the template files have already been created.

Updating EUPS
-------------

Scons will not use pybind11 unless it is setup, so in ``{{pkg}}/ups/{{pkg}}.table``,
where ``{{pkg}}`` is the name of the package, you will need to add the dependency
``setupRequired(pybind11)``.
You also need to modify the ``dependencies`` in ``{{pkg}}/ups/{{pkg}}.cfg``, changing
``"swig"`` to ``"pybind11"`` in ``"buildRequired"``.

Cleaning up gitignore
----------------------

Most Swig-based packages ignore files of the form ``*Lib.py``, as these are auto-generated by Swig. In
pybind11, these files are created manually. When ``build_templates.py`` is run, it will create stubs for
these files, but you will need to remove the pattern from ``.gitignore`` for git to recognize them as addable
files. You may also remove ``*_wrap.cc``, as these are also Swig-specific files.

Deactivating the tests
----------------------

In order to rebuild the stack up to the new package,
the tests in the new package you are about to wrap must be deactivated
(otherwise scons will fail to complete the build).
When ``build_templates.py`` is run, it creates a file ``tests/test.txt``,
which contains a list of all of the tests for the current package, commented out with a ``#`` character.
As you are wrapping code, the tests can be re-activated by deleting the comment character.
In order for scons to only run the uncommented tests and ignore the others,
the following lines must be manually inserted into the ``tests/SConscript`` file:

.. code-block:: python

    with open('test.txt', 'r') as f:
        tests = f.readlines()
    # Load the tests that have been wrapped (ignoring the "test/" preceeding the test name)
    pybind11_ported_tests = [t[6:].strip() for t in tests if not t.startswith('#')]

and the line

.. code-block:: python

    scripts.BasicSConscript.tests()

must be changed to

.. code-block:: python

    scripts.BasicSConscript.tests(pyList=pybind11_ported_tests)

.. note::

    It is possible that scripts.BasicSConscript.tests might contain other args or kwargs,
    in which case ``pyList=pybind11_ported_tests`` is inserted as a new kwarg.

Don't forget to immediately commit these changes and push to the github remote so that other developers will
have access to the new files.

.. _all-tests:

Running all Tests
=================

Before merging a test with the main branch ``DM-8467`` you should always ensure that all 
of the tests wrapped with pybind11, not just the new ones wrapped in the current branch, still succeed. 
There is a text file ``tests/test.txt`` that lists all of the tests in the current package.
To run all of the wrapped tests use:

.. code:: bash

    $ py.test `sed -e '/^#/d' tests/test.txt`

.. _new_test:

Wrapping a New Test
===================

Setup
-----

Since the stack has been built using the pybind11 branch of lsstsw,
once lsstsw has been setup you can simply use

.. code-block:: bash

    $ cd <repository directory>
    $ setup -r .

to setup the package currently being wrapped.

.. _locking:

Rebasing
--------

Because the pybind11 stack is a fork of the master lsst packages,
frequent rebasing will occur throughout the pybind11 port.
Additionally, while we strive to have different developers work as much as possible on independent packages,
the numerous interdependencies will sometimes require working on the same package and even in the same 
ticket branch. Thus frequent pushing and rebasing is necessary to keep everyone's stack up to date.
To rebase from the current pybind11 master, DM-8467, use

.. code-block:: bash

    $ git checkout tickets/DM-8467
    $ git fetch
    $ git reset --hard origin/tickets/DM-8467
    $ git checkout <branch>
    $ git rebase --onto tickets/DM-8467 C~ tickets/<branch>

where ``<branch>`` is the branch to update and ``C`` is the first commit made in the current ticket.
This series of commands does a force pull to get the latest version of DM-8467 and then rebases all of the
new commits on top of the rebased DM-8467.

Building the current test
-------------------------

As you wrap the package it can be useful to compile the package using

.. code-block:: bash

    $ scons python lib

which only builds the changes to the package and does not build the docs or run any of the tests,
which can save a substantial amount of time.

.. _activate-test:

Activating and skipping tests
-----------------------------

Many test files have multiple tests and sometimes even multiple test classes inside of them.
It can be useful to only run one test at a time (to prevent a bombardment of errors).
This can be done with 

.. code-block:: bash

    $ py.test -k {{test}} tests/{{test file}}

where ``{{test}}`` is the name of a test class or test method and ``{{test file}}`` is the name of the
test file you are wrapping.

Occasionally there may be an individual test that fails because of a bug in pybind11.
In this case the test cane be skipped using the decorator ``@unittest.skip("TODO:pybind11")``.

Also make sure to uncomment the test in ``tests/test.txt`` so that the test will be run by scons.

Final Steps
-----------

Once an entire package has been wrapped with pybind11, it is necessary to remove
``tests/test.txt``. In ``tests/SConscript`` you will also have to remove the lines

.. code-block:: python

    with open('test.txt', 'r') as f:
        tests = f.readlines()
    # Load the tests that have been wrapped (ignoring the "test/" preceeding the test name)
    pybind11_ported_tests = [t[6:].strip() for t in tests if not t.startswith('#')]

and remove the kwarg ``pyList=pybind11_ported_tests`` from ``scripts.BasicSConscript.tests``.

Tutorial
========

To illustrate how to wrap a test we will use ``afw/tests/testMinimize.py`` as an example.
We start by cloning https://github.com/lsst/afw to our local machine and checkout the correct 
ticket branch for the current test.
In this case ``testMinimize.py`` is in ``tickets/DM-6298``,
so we checkout that branch and set it up with ``setup -r .`` from the main ``afw`` repository directory.

Compiling the Code
------------------

Before we make any changes it's a good idea to compile the cloned repository to make sure that
everything is setup correctly. From the ``afw`` repository main directory run

.. code-block:: bash

    $ git clean -dfx

followed by

.. code-block:: bash

    $ scons

to do a clean build of afw.
Since this is your first build of afw it will take a while but using

.. code-block:: bash

    $ scons lib python

as you make changes will only build the newly wrapped headers, making development much faster than with SWIG.
One should remember to occasionally run all of the wrapped tests

Activate the test
-----------------

Activate the test file by uncommenting it in the ``tests/test.txt`` file as described in :ref:`activate-test`.

.. _test_minimize:

testMinimize.py
---------------

In this case the only test class,
``MinimizeTestCase``, imports two functions from 
``afw.math``: ``PolynomialFunction2D`` from ``afw/math/functionLibrary.h`` and 
``minimize`` from ``afw/math/minimize.h``:

.. code-block:: c++

    class MinimizeTestCase(lsst.utils.tests.TestCase):

        def testMinimize2(self):

            variances = np.array([0.01, 0.01, 0.01, 0.01])
            xPositions = np.array([0.0, 1.0, 0.0, 1.0])
            yPositions = np.array([0.0, 0.0, 1.0, 1.0])

            polyOrder = 1
            polyFunc = afwMath.PolynomialFunction2D(polyOrder)

            modelParams = [0.1, 0.2, 0.3]
            polyFunc.setParameters(modelParams)
            measurements = []
            for x, y in zip(xPositions, yPositions):
                measurements.append(polyFunc(x, y))
            print("measurements=", measurements)

            # Set up initial guesses
            nParameters = polyFunc.getNParameters()
            initialParameters = np.zeros(nParameters, float)
            stepsize = np.ones(nParameters, float)
            stepsize *= 0.1

            # Minimize!
            fitResults = afwMath.minimize(
                polyFunc,
                initialParameters.tolist(),
                stepsize.tolist(),
                measurements,
                variances.tolist(),
                xPositions.tolist(),
                yPositions.tolist(),
                0.1,
            )

            print("modelParams=", modelParams)
            print("fitParams  =", fitResults.parameterList)
            self.assertTrue(fitResults.isValid, "fit failed")
            self.assertFloatsAlmostEqual(np.array(modelParams), np.array(fitResults.parameterList), 1e-11)

We'll start with by wrapping the ``minimize`` function in ``minimize.h``.

.. _new_cpp:

Including a new C++ Header
--------------------------

We first have to tell scons about the new header we want to wrap,
so we modify ``python/lsst/afw/math/SConscript`` to read

.. code-block:: python

    from lsst.sconsUtils import scripts
    scripts.BasicSConscript.pybind11(['minimize'])

by uncommenting every line and adding the name of the new .cc file, in this case ``minimize``.
We also need to tell Python to import the new modules in ``python/lsst/afw/math/mathLib.py``, 
where we add the line

.. code-block:: python

    from __future__ import absolute_import
    from ._minimize import *

Since we are wrapping the header file ``minimize.h`` we must make sure to include it in 
``minimize.cc`` (which is the previously created pybind11 template):

.. code-block:: c++

    #include "lsst/afw/math/minimize.h"

.. _wrap_struct:

Wrapping a struct
-----------------

The header file ``minimize.h`` contains the following code:

.. code-block:: c++

    #include <memory>
    #include "Minuit2/FCNBase.h"

    #include "lsst/daf/base/Citizen.h"
    #include "lsst/afw/math/Function.h"

    namespace lsst {
    namespace afw {
    namespace math {

        struct FitResults {
        public:
            bool isValid;   ///< true if the fit converged; false otherwise
            double chiSq;   ///< chi squared; may be nan or infinite, but only if isValid false
            std::vector<double> parameterList; ///< fit parameters
            std::vector<std::pair<double,double> > parameterErrorList; ///< negative,positive (1 sigma?) error for each parameter
        };

        template<typename ReturnT>
        FitResults minimize(
            lsst::afw::math::Function1<ReturnT> const &function,
            std::vector<double> const &initialParameterList,
            std::vector<double> const &stepSizeList,
            std::vector<double> const &measurementList,
            std::vector<double> const &varianceList,
            std::vector<double> const &xPositionList,
            double errorDef
        );

        template<typename ReturnT>
        FitResults minimize(
            lsst::afw::math::Function2<ReturnT> const &function,
            std::vector<double> const &initialParameterList,
            std::vector<double> const &stepSizeList,
            std::vector<double> const &measurementList,
            std::vector<double> const &varianceList,
            std::vector<double> const &xPositionList,
            std::vector<double> const &yPositionList,
            double errorDef
        );

    }}}   // lsst::afw::math

    #endif // !defined(LSST_AFW_MATH_MINIMIZE_H)


We notice that ``minimize`` is a function that returns type ``FitResults``,
and since ``FitResults`` is an ordinary structure we will wrap it first.
In ``minimize.cc``, ``PYBIND11_PLUGIN`` contains the code to initialize the Python module ``_minimize``,
and all of the methods will be placed in this code block.
So inside the ``PYBIND11_PLUGIN`` code block, and after the module declaration 
``py::module mod("_minimize", "Python wrapper for afw _minimize library");`` we add

.. code-block:: c++

    py::class_<FitResults> clsFitResults(mod, "FitResults");

which creates the class clsFitResults in the current module, linked to ``FitResults`` in the header file.
Next we add the attributes from ``FitResults`` in ``minimize.h`` beneath the new class we just declared:

.. code-block:: c++

    clsFitResults.def_readwrite("isValid", &FitResults::isValid);
    clsFitResults.def_readwrite("chiSq", &FitResults::chiSq);
    clsFitResults.def_readwrite("parameterList", &FitResults::parameterList);
    clsFitResults.def_readwrite("parameterErrorList", &FitResults::parameterErrorList);

This is sufficient to bind the structure to our Python code.

.. note::

    You can also add names for the function arguments if you choose.
    This is only required when using the function has default arguments but can be useful for
    future developers, although including them is not required at this time.
    For more on using named arguments see :ref:`function_kwargs`.

At this time ``minimize.cc`` should look like

.. code-block:: c++

    #include <pybind11/pybind11.h>
    #include <pybind11/stl.h>

    #include "lsst/afw/math/minimize.h"

    namespace py = pybind11;

    namespace lsst {
    namespace afw {
    namespace math {

    PYBIND11_PLUGIN(_minimize) {
        py::module mod("_minimize", "Python wrapper for afw _minimize library");

        py::class_<FitResults> clsFitResults(mod, "FitResults");

        clsFitResults.def_readwrite("isValid", &FitResults::isValid);
        clsFitResults.def_readwrite("chiSq", &FitResults::chiSq);
        clsFitResults.def_readwrite("parameterList", &FitResults::parameterList);
        clsFitResults.def_readwrite("parameterErrorList", &FitResults::parameterErrorList);

        return mod.ptr();
    }
    
    }}} // lsst::afw::math

This is a good time to build our changes (at times the error messages generated by pybind11 
can be obscure so it is useful to recompile after each wrapped class).
From the shell prompt run

.. code-block:: bash

    $ scons lib python

to build all of the changes you made to afw.
If the build failed, go back and verify that all of your method definitions used the 
correct syntax as displayed above.

Wrapping an overloaded function
-------------------------------

Now that we have created the ``FitResults`` structure we can create our ``minimize`` function wrapper.
This is done using the ``def`` method of ``py::module``,
where we must create a definition for each set of parameters.
Looking in the swig ``.i`` file located at 
https://github.com/lsst/afw/blob/master/python/lsst/afw/math/minimize.i we see that there are two
templated types: ``float`` and ``double``.

.. note::

    Whenever you encounter a problem that requires you to look at the swig files you are best off
    looking at the code on github, as the swig files have been deleted in the pybind11 branch
    and switching branches locally will require you to commit or stash your changes,
    which might be inconvenient at the time.

In a minute we will wrap ``minimize`` for both types,
but it is useful to first look at how this would be done for a single type ``double``.
In this case we define ``minimize`` and cast it to a ``FitResults`` function pointer underneath 
our ``clsFitResults`` code using

.. code-block:: c++

    mod.def("minimize", (FitResults (*) (lsst::afw::math::Function1<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         double)) &minimize<double>);

.. note::

    You might notice that we have used a C-style cast, consistent with the pybind11 documentation.
    It is also possible to use the more verbose C++-style cast 
    ``mod.def("f", static_cast<void (*)(int)>(f));`` as opposed to the C-style
    ``mod.def("f", (void (*)(int))f);``.

Notice that for each parameter in the C++ function we include the type
(including a reference if necessary) in our pybind11 function declaration but not the variable name itself.
Similarly, beneath this code we add the second set of parameters for the overloaded function

.. code-block:: c++

    mod.def("minimize", (FitResults (*) (lsst::afw::math::Function2<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         double)) &minimize<double>);

We could copy these lines and change the templates to use type ``float`` if we wanted to,
or instead we can write a function that allow us to template an arbitrarily large number of different types.
This is not necessary with only two function types but it is useful to wrap them this way anyway for clarity,
and as an exercise to illustrate how this is done in pybind11.

Between the namespace declaration and start of the ``PYBIND11_PLUGIN`` macro
we can define a template function to declare the ``minimize`` function:

.. code-block:: c++

    namespace{
    template <typename ReturnT>
    void declareMinimize(py::module & mod) {
        mod.def("minimize", (FitResults (*) (lsst::afw::math::Function1<ReturnT> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             double)) &minimize<ReturnT>);
        mod.def("minimize", (FitResults (*) (lsst::afw::math::Function2<ReturnT> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             double)) &minimize<ReturnT>);
    };
    } // namespace

Notice that the only changes we made to the function definition was to change 
``lsst::afw::math::Function1<double>`` to ``lsst::afw::math::Function1<ReturnT>`` and 
``minimize<double>`` to ``minimize<ReturnT>`` in both definitions.
We also enclosed the function in an anonymous namespace, which is necessary to prevent the declaration
from entering the ``lsst::afw::math`` namespace.
Now we can replace the ``mod.def("minimize", ...`` definitions in ``PYBIND11_PLUGIN`` with

.. code-block:: c++

    declareMinimize<double>(mod);
    declareMinimize<float>(mod);

which declares both templates for minimize.

.. warning::

    In certain cases the order that templates are defined can affect the way in which the code runs.
    For example, notice that above we first defined the ``double`` template followed by ``float``.
    This is because unlike the C++ compiler,
    which finds the tempalte that best matches the given parameters,
    pybind11 will attempt to cast the parameters to a different type.
    So in general it is best to declare ``double`` before ``float``, ``long`` before ``int``, etc.
    This can become even more complicated when using numpy arrays, where much care is needed to ensure
    that overloaded templates are being cast correctly.

Putting it all together, the file ``minimize.cc`` should look like

.. code-block:: c++

    #include <pybind11/pybind11.h>
    #include <pybind11/stl.h>

    #include "lsst/afw/math/minimize.h"

    namespace py = pybind11;

    namespace lsst {
    namespace afw {
    namespace math {

    namespace {
    template <typename ReturnT>
    void declareMinimize(py::module & mod) {
        mod.def("minimize", (FitResults (*) (lsst::afw::math::Function1<ReturnT> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             double)) minimize<ReturnT>);
        mod.def("minimize", (FitResults (*) (lsst::afw::math::Function2<ReturnT> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             std::vector<double> const &,
                                             double)) minimize<ReturnT>);
    };
    }

    PYBIND11_PLUGIN(_minimize) {
        py::module mod("_minimize", "Python wrapper for afw _minimize library");

        py::class_<FitResults> clsFitResults(mod, "FitResults");

        clsFitResults.def_readwrite("isValid", &FitResults::isValid);
        clsFitResults.def_readwrite("chiSq", &FitResults::chiSq);
        clsFitResults.def_readwrite("parameterList", &FitResults::parameterList);
        clsFitResults.def_readwrite("parameterErrorList", &FitResults::parameterErrorList);

        declareMinimize<double>(mod);
        declareMinimize<float>(mod);

        return mod.ptr();
    }
    
    }}} // lsst::afw::math

When casting an overloaded member function of a class ``ClassName``,
the ``(*)`` must be replaced with ``(ClassName::*)``.
So if minimize had been a class method of MinimizeClass, we would have used
    
.. code-block:: c++
    
    mod.def("minimize", (FitResults (MinimizeClass::*) (lsst::afw::math::Function1<ReturnT> const &,
                                                        std::vector<double> const &,
                                                        std::vector<double> const &,
                                                        std::vector<double> const &,
                                                        std::vector<double> const &,
                                                        std::vector<double> const &,
                                                        double)) &MinimizeClass::minimize<ReturnT>);

Another subtlety is encountered when wrapping a static method of a class,
where we use ``def_static`` and once again use ``(*)`` instead of ``FitResults::*``:

.. code-block:: c++

    mod.def_static("minimize", (FitResults (*) (lsst::afw::math::Function1<ReturnT> const &,
                                                std::vector<double> const &,
                                                std::vector<double> const &,
                                                std::vector<double> const &,
                                                std::vector<double> const &,
                                                std::vector<double> const &,
                                                double)) MinimizeClass::minimize<ReturnT>);

.. _wrap_suffix:

Wrapping a Template with a suffix
---------------------------------

We still have not successfully wrapped all of the classes and functions needed to run ``testMinimize.py``, 
as we haven't wrapped PolynomialFunction2D in ``afw/math/functionLibrary.py``.
The relevant code from ``functionLibrary.h`` is shown here:

.. code-block:: c++

    template<typename ReturnT>
    class PolynomialFunction2: public BasePolynomialFunction2<ReturnT> {
    public:
        typedef typename Function2<ReturnT>::Ptr Function2Ptr;

        explicit PolynomialFunction2(
            unsigned int order) ///< order of polynomial (0 for constant)
        :
            BasePolynomialFunction2<ReturnT>(order),
            _oldY(0),
            _xCoeffs(this->_order + 1)
        {}

        explicit PolynomialFunction2(
            std::vector<double> params)  ///< polynomial coefficients (const, x, y, x^2, xy, y^2...);
                                    ///< length must be one of 1, 3, 6, 10, 15...
        :
            BasePolynomialFunction2<ReturnT>(params),
            _oldY(0),
            _xCoeffs(this->_order + 1)
        {}

        virtual ~PolynomialFunction2() {}

        virtual Function2Ptr clone() const {
            return Function2Ptr(new PolynomialFunction2(this->_params));
        }

        virtual ReturnT operator() (double x, double y) const {
            /* Operator code here */
        }

        /* Code not needed for wrapping the current function here */
    };

So we begin with ``Function`` in ``afw/math/FunctionLibrary.h``.
We add ``'functionLibrary'`` to ``afw/math/SConscript``,
``from ._functionLibrary import *`` to ``mathLib.py``,
and ``#include "lsst/afw/math/FunctionLibrary.h"`` to ``functionLibrary.cc`` just like we did for 
``minimize.h`` in :ref:`new_cpp`.

Below ``namespace lsst { namespace afw { namespace math {`` 
and before ``PYBIND11_PLUGIN`` we create the new template function

.. code-block:: c++

    template <typename ReturnT>
    void declarePolynomialFunctions(py::module &mod, std::string const & suffix) {
    };

where ``suffix`` will be a string that represents the return type of the function 
("D" for double, "I" for int, etc.).
We also must uncomment

.. code-block:: c++

    #include <pybind11/stl.h>

to use pybind11 wrappers for the C++ standard library.


Inside the function we declare our class

.. code-block:: c++

        py::class_<PolynomialFunction2<ReturnT>, BasePolynomialFunction2<ReturnT>>
            clsPolynomialFunction2(mod, ("PolynomialFunction2" + suffix).c_str());

This is slightly different than our class declaration in :ref:`wrap_struct` because 
``PolynomialFunction2`` inherits from ``BasePolynomialFunction2``,
which can be seen in the above declaration.
Since ``BasePolynomialFunction2`` is defined in ``Function.h`` we must add
``#include "lsst/afw/math/Function.h"`` at the beginning of ``functionLibrary.cc``.
We will discuss inheritance more in :ref:`wrapping_inheritance`.
Also notice that we combine ``PolynomialFunction2`` with the suffix,
specified when ``declarePolyomialFunctions`` is defined,
that specified the type for the function (for example "D" or "I").

We notice that the constructor is overloaded, so we define ``init`` with both sets of parameters

.. code-block:: c++

    clsPolynomialFunction2.def(py::init<unsigned int>());
    clsPolynomialFunction2.def(py::init<std::vector<double> const &>());


We must also declare the classes in the module,
so inside ``PYBIND11_PLUGIN`` and beneath the module declaration ``py::module mod("_functionLibrary",
"Python wrapper for afw _functionLibrary library");`` we add

.. code-block:: c++

    declarePolynomialFunctions<double>(mod, "D");

where we use the ``double`` type since ``PolynomialFunction2D`` is the method called from
``testMinimize.py``, and specify ``suffix`` as ``"D"``.

The last piece to wrap in ``functionLibrary.cc`` is ``operator()`` method, which can be wrapped using

.. code-block:: c++

    clsPolynomialFunction2.def("__call__", &PolynomialFunction2<ReturnT>::operator());

At this point ``functionLibrary.cc`` should look like:

.. code-block:: c++

    #include <pybind11/pybind11.h>
    #include <pybind11/stl.h>

    #include "lsst/afw/math/functionLibrary.h"
    #include "lsst/afw/math/Function.h"

    namespace py = pybind11;

    namespace lsst {
    namespace afw {
    namespace math {

    namespace {
    template <typename ReturnT>
    void declarePolynomialFunctions(py::module &mod, std::string const & suffix) {
       py::class_<PolynomialFunction2<ReturnT>, BasePolynomialFunction2<ReturnT>>
            clsPolynomialFunction2(mod, ("PolynomialFunction2" + suffix).c_str());
        clsPolynomialFunction2.def(py::init<unsigned int>());
        clsPolynomialFunction2.def(py::init<std::vector<double> const &>());

        /* Operators */
        clsPolynomialFunction2.def("__call__", &PolynomialFunction2<ReturnT>::operator());
    };
    } // namespace

    PYBIND11_PLUGIN(_functionLibrary) {
        py::module mod("_functionLibrary", "Python wrapper for afw _functionLibrary library");

        declarePolynomialFunctions<double>(mod, "D");

        return mod.ptr();
    }
    
    }}} // lsst::afw::math

Of course the test will still not run since ``PolynomialFunction2`` depends on the methods 
``setParameters``and ``getNParameters``, which are inherited.

.. _wrapping_inheritance:

Inheritance
-----------

Now we journey down the rabbit hole that is inheritance and see that ``BasePolynomialFunction2``
inherits from ``Function2`` which inherits from ``Function``,
which inherits from classes outside of afw.
In many cases, it may not be necessary to include all of the inherited classes as use of the
inherited classes might only be used in the C++ code.
So we begin with ``BasePolynomialFunction2`` and work our way down.
This is consistent with our workflow to only wrap the necessary methods to pass a test and
as a bonus can save a significant amount of build time.

Beginning with ``Function`` in ``afw/math/Function.h`` we add ``'function'`` to ``afw/math/SConscript``,
``from ._function import *`` to ``mathLib.py``,
and ``#include "lsst/afw/math/Function.h"`` in ``function.cc`` just like we did for ``minimize.h`` in 
:ref:`new_cpp` and ``functionLibrary.h`` in :ref:`wrap_suffix`.

Below is the relevant part of ``Function.h`` for ``BasePolynomialFunction2``:

.. code-block:: c++

    template<typename ReturnT>
    class BasePolynomialFunction2: public Function2<ReturnT> {
    public:
        typedef typename Function2<ReturnT>::Ptr Function2Ptr;

        explicit BasePolynomialFunction2(
            unsigned int order) ///< order of polynomial (0 for constant)
        :
            Function2<ReturnT>(BasePolynomialFunction2::nParametersFromOrder(order)),
            _order(order)
        {}

        explicit BasePolynomialFunction2(
            std::vector<double> params) ///< polynomial coefficients
        :
            Function2<ReturnT>(params),
            _order(BasePolynomialFunction2::orderFromNParameters(static_cast<int>(params.size())))
        {}

        /* Other methods unnecessary for this wrap hidden */
    };

In this case ``Function``, ``Function2`` and ``BasePolynomialFunction2`` are all templated on the same type.
So we declare them together in one function template:

.. code-block:: c++

    template<typename ReturnT>
    void declareFunctions(py::module &mod, std::string const & suffix){
    };

just like we did in :ref:`wrap_suffix`.
As mentioned above,
we should not assume that we need to inherit from ``Function2``, but in this case we see that
``BasePolynomialFunction2`` is still missing the ``setParamters`` and ``getNParameters``
methods that are needed in ``PolynomialFunction2``,
so we inherit from ``Function2`` by adding the following lines to ``declareFunctions``:

.. code-block:: c++

    py::class_<BasePolynomialFunction2<ReturnT>, Function2<ReturnT> >
        clsBasePolynomialFunction2(mod, ("BasePolynomialFunction2" + suffix).c_str());

There are no other methods of ``BasePolynomialFunction`` needed for the current test so we move on to
``Function2``, with the relevant code below:

.. code-block:: c++

    template<typename ReturnT>
    class Function2 : public afw::table::io::PersistableFacade< Function2<ReturnT> >,
                      public Function<ReturnT>
    {
    public:
        typedef std::shared_ptr<Function2<ReturnT> > Ptr;

        explicit Function2(
            unsigned int nParams)   ///< number of function parameters
        :
            Function<ReturnT>(nParams)
        {}

        explicit Function2(
            std::vector<double> const &params)   ///< function parameters
        :
            Function<ReturnT>(params)
        {}

        /* Other methods unnecessary for this wrap hidden */
    };

So we see that ``Function2`` inherits from both ``Function`` and ``afw::table::io::PersistableFacade``.
In this case it is not immediately obvious that we will need the latter class available to Python,
so we only include ``Function`` in our class declaration
(which we place before our ``BasePolynomialFunction2`` declaration)

.. code-block:: c++

    py::class_<Function2<ReturnT>, Function<ReturnT>> clsFunction2(mod, ("Function2"+suffix).c_str());

We have finally made it to the end of our inheritance chain.
Looking at the relevant part of the code

.. code-block:: c++

    template<typename ReturnT>
    class Function : public lsst::daf::base::Citizen,
                     public afw::table::io::PersistableFacade< Function<ReturnT> >,
                     public afw::table::io::Persistable
    {
    public:
        explicit Function(
            unsigned int nParams)   ///< number of function parameters
        :
            lsst::daf::base::Citizen(typeid(this)),
            _params(nParams),
            _isCacheValid(false)
        {}

        explicit Function(
            std::vector<double> const &params)   ///< function parameters
        :
            lsst::daf::base::Citizen(typeid(this)),
            _params(params),
            _isCacheValid(false)
        {}

        unsigned int getNParameters() const {
            return _params.size();
        }

        void setParameters(
            std::vector<double> const &params)   ///< vector of function parameters
        {
            if (_params.size() != params.size()) {
                throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                    (boost::format("params has %d entries instead of %d") % \
                    params.size() % _params.size()).str());
            }
            _isCacheValid = false;
            _params = params;
        }
    /* Other methods unnecessary for this wrap hidden */
    }

We see that ``Function`` also has multiple inheritances but for now we ignore them
(as it does not appear that we necessarily need them exposed to Python) when we declare it:

.. code-block:: c++

    py::class_<Function<ReturnT>> clsFunction(mod, ("Function"+suffix).c_str());

The constructor is overloaded so beneath the class declaration we need to define ``init`` 
for both sets of parameters:

.. code-block:: c++

    clsFunction.def(py::init<unsigned int>());
    clsFunction.def(py::init<std::vector<double> const &>());

Recall from :ref:`test_minimize` that two methods of ``PolynomialFunction2D`` are needed that are
defined in ``Function``: ``getNParameters`` and ``setParameters``, so we define them with

.. code-block:: c++

     clsFunction.def("getNParameters", &Function<ReturnT>::getNParameters);
     clsFunction.def("setParameters", &Function<ReturnT>::setParameters);

There are no other ``Function`` methods needed for now,
so we leave wrapping them for the future if they are necessary on the Python side of the stack.

At this point ``function.cc`` should look like

.. code-block:: c++

    #include <pybind11/pybind11.h>
    #include <pybind11/stl.h>

    #include "lsst/afw/math/Function.h"

    namespace py = pybind11;

    namespace lsst {
    namespace afw {
    namespace math {

    namespace {
    template<typename ReturnT>
    void declareFunctions(py::module &mod, std::string const & suffix){
        /* Function */
        py::class_<Function<ReturnT>> clsFunction(mod, ("Function"+suffix).c_str());
        /* Function Constructors */
        clsFunction.def(py::init<unsigned int>());
        clsFunction.def(py::init<std::vector<double> const &>());
        /* Function Members */
        clsFunction.def("getNParameters", &Function<ReturnT>::getNParameters);
        clsFunction.def("setParameters", &Function<ReturnT>::setParameters);

        /* Function2 */
        py::class_<Function2<ReturnT>, Function<ReturnT>> clsFunction2(mod, ("Function2"+suffix).c_str());

        /* BasePolynomialFunction2 */
        py::class_<BasePolynomialFunction2<ReturnT>, Function2<ReturnT> >
            clsBasePolynomialFunction2(mod, ("BasePolynomialFunction2" + suffix).c_str());
    };
    } // namespace

    PYBIND11_PLUGIN(_function) {
        py::module mod("_function", "Python wrapper for afw _function library");

        declareFunctions<double>(mod, "D");

        return mod.ptr();
    }
    
    }}} lsst::afw::math

and you should be able to compile the code using ``scons lib python`` (hopefully you have been building
after each new class or you could come across multiple errors at this point).
You should now be able to run ``py.test tests/testMinimize.py`` and pass all of the tests.

testInterpolate.py
------------------

There are still multiple edge cases we have yet to encounter,
including pure virtual functions, ndarrays, and enum types.
All of these cases are needed to wrap ``testInterpolate.py`` with pybind11,
so we use it to illustrate these procedures. Here is the ``testInterpolate.py`` code:

.. code-block:: python

    from __future__ import absolute_import, division
    from builtins import zip
    from builtins import range
    import unittest
    import numpy as np
    import lsst.utils.tests
    import lsst.afw.math as afwMath
    import lsst.pex.exceptions as pexExcept

    class InterpolateTestCase(lsst.utils.tests.TestCase):

        """A test case for Interpolate Linear"""

        def setUp(self):
            self.n = 10
            self.x = np.zeros(self.n, dtype=float)
            self.y1 = np.zeros(self.n, dtype=float)
            self.y2 = np.zeros(self.n, dtype=float)
            self.y0 = 1.0
            self.dydx = 1.0
            self.d2ydx2 = 0.5

            for i in range(0, self.n, 1):
                self.x[i] = i
                self.y1[i] = self.dydx*self.x[i] + self.y0
                self.y2[i] = self.d2ydx2*self.x[i]*self.x[i] + self.dydx*self.x[i] + self.y0

            self.xtest = 4.5
            self.y1test = self.dydx*self.xtest + self.y0
            self.y2test = self.d2ydx2*self.xtest*self.xtest + self.dydx*self.xtest + self.y0

        def tearDown(self):
            del self.x
            del self.y1
            del self.y2

        def testLinearRamp(self):

            # === test the Linear Interpolator ============================
            # default is akima spline
            yinterpL = afwMath.makeInterpolate(self.x, self.y1)
            youtL = yinterpL.interpolate(self.xtest)

            self.assertEqual(youtL, self.y1test)

        def testNaturalSplineRamp(self):

            # === test the Spline interpolator =======================
            # specify interp type with the string interface
            yinterpS = afwMath.makeInterpolate(self.x, self.y1, afwMath.Interpolate.NATURAL_SPLINE)
            youtS = yinterpS.interpolate(self.xtest)

            self.assertEqual(youtS, self.y1test)

        def testAkimaSplineParabola(self):
            """test the Spline interpolator"""
            # specify interp type with the enum style interface
            yinterpS = afwMath.makeInterpolate(self.x, self.y2, afwMath.Interpolate.AKIMA_SPLINE)
            youtS = yinterpS.interpolate(self.xtest)

            self.assertEqual(youtS, self.y2test)

        def testConstant(self):
            """test the constant interpolator"""
            # [xy]vec:   point samples
            # [xy]vec_c: centered values
            xvec = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            xvec_c = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
            yvec = np.array([1.0, 2.4, 5.0, 8.4, 13.0, 18.4, 25.0, 32.6, 41.0, 50.6])
            yvec_c = np.array([1.0, 1.7, 3.7, 6.7, 10.7, 15.7, 21.7, 28.8, 36.8, 45.8, 50.6])

            interp = afwMath.makeInterpolate(xvec, yvec, afwMath.Interpolate.CONSTANT)

            for x, y in zip(xvec_c, yvec_c):
                self.assertAlmostEqual(interp.interpolate(x + 0.1), y)
                self.assertAlmostEqual(interp.interpolate(x), y)

            self.assertEqual(interp.interpolate(xvec[0] - 10), yvec[0])
            n = len(yvec)
            self.assertEqual(interp.interpolate(xvec[n - 1] + 10), yvec[n - 1])

            for x, y in reversed(list(zip(xvec_c, yvec_c))):  # test caching as we go backwards
                self.assertAlmostEqual(interp.interpolate(x + 0.1), y)
                self.assertAlmostEqual(interp.interpolate(x), y)

            i = 2
            for x in np.arange(xvec_c[i], xvec_c[i + 1], 10):
                self.assertEqual(interp.interpolate(x), yvec_c[i])

        def testInvalidInputs(self):
            """Test that invalid inputs cause an abort"""

            self.assertRaises(pexExcept.OutOfRangeError,
                              lambda: afwMath.makeInterpolate(np.array([], dtype=float), np.array([], dtype=float),
                                                              afwMath.Interpolate.CONSTANT)
                              )

            afwMath.makeInterpolate(np.array([0], dtype=float), np.array([1], dtype=float),
                                    afwMath.Interpolate.CONSTANT)

            self.assertRaises(pexExcept.OutOfRangeError,
                              lambda: afwMath.makeInterpolate(np.array([0], dtype=float), np.array([1], dtype=float),
                                                              afwMath.Interpolate.LINEAR))


    class TestMemory(lsst.utils.tests.MemoryTestCase):
        pass

    def setup_module(module):
        lsst.utils.tests.init()

    if __name__ == "__main__":
        lsst.utils.tests.init()
        unittest.main()

Here we see that there is only one class called from this test: ``lsst::afw::math::Interpolate``.
We make sure to add the appropriate lines to ``mathLib.py``, ``Sconscript``, and ``interpolate.cc``
as we saw in :ref:`new_cpp`.

Below is the ``interpolate.h`` code:

.. code-block:: c++

    #include "lsst/base.h"
    #include "ndarray_fwd.h"

    namespace lsst {
    namespace afw {
    namespace math {

     /**
     * @brief Interpolate values for a set of x,y vector<>s
     * @ingroup afw
     * @author Steve Bickerton
     */
    class Interpolate {
    public:
        enum Style {
            UNKNOWN = -1,
            CONSTANT = 0,
            LINEAR = 1,
            NATURAL_SPLINE = 2,
            CUBIC_SPLINE = 3,
            CUBIC_SPLINE_PERIODIC = 4,
            AKIMA_SPLINE = 5,
            AKIMA_SPLINE_PERIODIC = 6,
            NUM_STYLES
        };

        friend PTR(Interpolate) makeInterpolate(std::vector<double> const &x, std::vector<double> const &y,
                                                Interpolate::Style const style);

        virtual ~Interpolate() {}
        virtual double interpolate(double const x) const = 0;
        std::vector<double> interpolate(std::vector<double> const& x) const;
        ndarray::Array<double, 1> interpolate(ndarray::Array<double const, 1> const& x) const;
    protected:
        /**
         * Base class ctor
         */
        Interpolate(std::vector<double> const &x, ///< the ordinates of points
                    std::vector<double> const &y, ///< the values at x[]
                    Interpolate::Style const style=UNKNOWN ///< desired interpolator
                   ) : _x(x), _y(y), _style(style) {}
        Interpolate(std::pair<std::vector<double>, std::vector<double> > const xy,
                    Interpolate::Style const style=UNKNOWN);

        std::vector<double> const _x;
        std::vector<double> const _y;
        Interpolate::Style const _style;
    private:
        Interpolate(Interpolate const&);
        Interpolate& operator=(Interpolate const&);
    };

    PTR(Interpolate) makeInterpolate(std::vector<double> const &x, std::vector<double> const &y,
                                     Interpolate::Style const style=Interpolate::AKIMA_SPLINE);
    PTR(Interpolate) makeInterpolate(ndarray::Array<double const, 1> const &x,
                                     ndarray::Array<double const, 1> const &y,
                                     Interpolate::Style const style=Interpolate::AKIMA_SPLINE);
    Interpolate::Style stringToInterpStyle(std::string const &style);
    Interpolate::Style lookupMaxInterpStyle(int const n);
    int lookupMinInterpPoints(Interpolate::Style const style);

    }}}

    #endif // LSST_AFW_MATH_INTERPOLATE_H

.. _smart_ptr:

Smart Pointers
^^^^^^^^^^^^^^

When declaring a class that will be accessed as a ``std::shared_ptr``,
it is necessary to also include ``std::shared_ptr<ClassName>>`` in the definition of ``ClassName``.
In this case, for the ``Interpolate`` class that means adding

.. code-block:: c++

    py::class_<Interpolate, std::shared_ptr<Interpolate>> clsInterpolate(mod, "Interpolate");

to the module section of ``interpolate.cc``.

.. warning::

    One of the most frequent causes of segfaults in class wrapped in pybind11 is to inherit from a
    class with a shared_pointer but not include the std_shared parameter. For example, if a class
    ``BetterInterpolate`` inherits from interpolate, it must include ``std::shared_ptr<BetterInterpolate``
    in its class definition. See section :ref:`segfaults` for more.

Enum types
^^^^^^^^^^

The first method is an enum called ``Style``.
We declare a value for each keyword that points to the corresponding value in the header file,
with an ``export_values()`` method at the end:

.. code-block:: c++

    py::enum_<Interpolate::Style>(clsInterpolate, "Style")
        .value("UNKNOWN", Interpolate::Style::UNKNOWN)
        .value("CONSTANT", Interpolate::Style::CONSTANT)
        .value("LINEAR", Interpolate::Style::LINEAR)
        .value("NATURAL_SPLINE", Interpolate::Style::NATURAL_SPLINE)
        .value("CUBIC_SPLINE", Interpolate::Style::CUBIC_SPLINE)
        .value("CUBIC_SPLINE_PERIODIC", Interpolate::Style::CUBIC_SPLINE_PERIODIC)
        .value("AKIMA_SPLINE", Interpolate::Style::AKIMA_SPLINE)
        .value("AKIMA_SPLINE_PERIODIC", Interpolate::Style::AKIMA_SPLINE_PERIODIC)
        .value("NUM_STYLES", Interpolate::Style::NUM_STYLES)
        .export_values();

.. warning::

    Do not forget to add the ``.export_values()`` at the end or your enumerated types will not be added to the class!

.. _virtual_functions:

Lambda Functions and abstract Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Notice from ``Interpolate.h`` that the constructor for Interpolate is protected,
so a new instance can only be created using the ``makeInterpolate`` function, making it an abstract class.

We will wrap ``makeInterpolate`` in :ref:`function_kwargs` but first we finish wrapping ``Interpolate``.
The main function is the method ``interpolate``, which can be called with a double, list, or ndarray.
From ``Interpolate.h`` we see that the list and ndarray declarations are trivial, but when a double is
used the method is pure virtual:

.. code-block:: c++

    virtual double interpolate(double const x) const = 0;

so we cannot wrap it directly (since there is nothing to wrap).

Instead we create a lambda function:

.. code-block:: c++

    clsInterpolate.def("interpolate", [](Interpolate &t, double const x) -> double {
            return t.interpolate(x);
    });

This defines the function ``Interpolate::interpolate``,
which will call the overwritten method ``interpolate`` of the ``Interpolate`` object directly.

.. _ndarray:

NDArrays
^^^^^^^^

Since the ``interpolate`` method is an overloaded function, only one of which is virtual,
we can wrap the other function definitions in the traditional way:

.. code-block:: c++

    clsInterpolate.def("interpolate",
                       (std::vector<double> (Interpolate::*) (std::vector<double> const&) const)
                           &Interpolate::interpolate);
    clsInterpolate.def("interpolate",
                       (ndarray::Array<double, 1> (Interpolate::*) (ndarray::Array<double const, 1> const&)
                           const) &Interpolate::interpolate);

However, since we are using ndarray's we also need to include the numpy and ndarray headers at the top of 
``interpolate.cc``

.. code-block:: c++

    #include "numpy/arrayobject.h"
    #include "ndarray/pybind11.h"
    #include "ndarray/converter.h"

It is also necessary to check that numpy has been installed and setup
(otherwise unexpected segfaults will occur), so in the module definition we add

.. code-block:: c++

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

.. _function_kwargs:

Wrapping Functions with Default Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final method remaining to wrap in ``interpolate.h`` is ``makeInterpolate``,
which creates an ``Interpolate`` object from the virtual class.

This is an overloaded function, so we define it in the usual way but add ``"parameter"_``
for *all* of the arguments of the function (not just the ones that we need to give default values).
In this case

.. code-block:: c++

    mod.def("makeInterpolate", 
            (PTR(Interpolate) (*)(std::vector<double> const &,
                                  std::vector<double> const &,
                                  Interpolate::Style const)) makeInterpolate,
            "x"_a, "y"_a, "style"_a=Interpolate::AKIMA_SPLINE);
    mod.def("makeInterpolate", 
            (PTR(Interpolate) (*)(ndarray::Array<double const, 1> const &,
                                  ndarray::Array<double const, 1> const &y,
                                  Interpolate::Style const)) makeInterpolate,
            "x"_a, "y"_a, "style"_a=Interpolate::AKIMA_SPLINE);

This format requires adding ``using namespace pybind11::literals;`` to the top of
``interpolate.cc`` (without using pybind11::literals parameters are defined using the more
clunky ``py::arg(x)=...`` notation).

.. note::

    If pybind11 returns an error during wrapping that the number of arguments does not match,
    check that you have wrapped all of the arguments with the correct types.
    Also make sure that you are defining the function in the correct place
    (ie. is it defined in the module or inside of a class).

.. _wrapped_interpolate:

Wrapped interpolate.cc
^^^^^^^^^^^^^^^^^^^^^^

When finished ``interpolate.cc`` should look like:

.. code-block:: c++

    #include <pybind11/pybind11.h>
    #include <pybind11/stl.h>

    #include "numpy/arrayobject.h"
    #include "ndarray/pybind11.h"
    #include "ndarray/converter.h"

    #include "lsst/afw/math/interpolate.h"

    namespace py = pybind11;
    using namespace pybind11::literals;

    namespace lsst {
    namespace afw {
    namespace math {

    PYBIND11_PLUGIN(_interpolate) {
        py::module mod("_interpolate", "Python wrapper for afw _interpolate library");

        if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
        }

        mod.def("makeInterpolate", 
                           (PTR(Interpolate) (*)(std::vector<double> const &,
                                                 std::vector<double> const &,
                                                 Interpolate::Style const)) makeInterpolate,
                           "x"_a, "y"_a, "style"_a=Interpolate::AKIMA_SPLINE);
        mod.def("makeInterpolate", 
                           (PTR(Interpolate) (*)(ndarray::Array<double const, 1> const &,
                                                 ndarray::Array<double const, 1> const &y,
                                                 Interpolate::Style const)) makeInterpolate,
                           "x"_a, "y"_a, "style"_a=Interpolate::AKIMA_SPLINE);
        /* Module level */

        /* Member types and enums */

        /* Constructors */

        /* Operators */

        /* Members */
        
        py::class_<Interpolate, std::shared_ptr<Interpolate>> clsInterpolate(mod, "Interpolate");
        py::enum_<Interpolate::Style>(clsInterpolate, "Style")
            .value("UNKNOWN", Interpolate::Style::UNKNOWN)
            .value("CONSTANT", Interpolate::Style::CONSTANT)
            .value("LINEAR", Interpolate::Style::LINEAR)
            .value("NATURAL_SPLINE", Interpolate::Style::NATURAL_SPLINE)
            .value("CUBIC_SPLINE", Interpolate::Style::CUBIC_SPLINE)
            .value("CUBIC_SPLINE_PERIODIC", Interpolate::Style::CUBIC_SPLINE_PERIODIC)
            .value("AKIMA_SPLINE", Interpolate::Style::AKIMA_SPLINE)
            .value("AKIMA_SPLINE_PERIODIC", Interpolate::Style::AKIMA_SPLINE_PERIODIC)
            .value("NUM_STYLES", Interpolate::Style::NUM_STYLES)
            .export_values();

        clsInterpolate.def("interpolate", [](Interpolate &t, double const x) -> double {
                return t.interpolate(x);
        });
        clsInterpolate.def("interpolate",
                           (std::vector<double> (Interpolate::*) (std::vector<double> const&) const)
                               &Interpolate::interpolate);
        clsInterpolate.def("interpolate",
                           (ndarray::Array<double, 1> (Interpolate::*) (ndarray::Array<double const, 1> const&)
                               const) &Interpolate::interpolate);

        return mod.ptr();
    }
    
    }}} // lsst::afw::math

Other Useful Tips
=================

Operators
---------

You may find it necessary to wrap operators.
While pybind11 contains a useful syntax to easily wrap operators,
we have found that it doesn't work as often as one would like.
Instead, we wrap an operator with a lambda function,
for example to overload the multiplication operator for a class A we use

.. code-block:: c++

    cls.def("__mul__", [](A const & self, A const & other) {
        return self * other;
    }, py::is_operator());

.. note::

    The ``py::is_operator()`` informs pybind11 that the wrapped function is an operator which should
    trigger a ``NotImplementedError`` instead of a ``TypeError`` when called with the wrong type.

.. _python-code:

Python Code
-----------

In some cases C++ classes are extended to include methods specific to the python API,
or to make C++ objects and methods more pythonic.
Unlike SWIG, which has a specific ``extend`` method,
monkey-patching like this is frowned upon in python and no formal method exists to extend a C++ class.
The following example provides the recommended method for extending C++ classes in our stack.

``afw::table`` contains an ``Arrays.h`` header file that defines the
``ArrayFKey`` and ``ArrayIKey`` objects.
The relevant pybind11 wrapper code ``arrays.cc`` is shown below:

.. code-block:: c++

    template <typename T>
    void declareArrayKey(py::module & mod, std::string const & suffix) {
        py::class_<ArrayKey<T>,
                   std::shared_ptr<ArrayKey<T>>,
                   FunctorKey<ndarray::Array<T const, 1, 1>>> clsArrayKey(mod, ("Array"+suffix+"Key").c_str());
    
        clsArrayKey.def(py::init<>());
        clsArrayKey.def("_get_", [](ArrayKey<T> & self, int i) {
            return self[i];
        });
        clsArrayKey.def("getSize", &ArrayKey<T>::getSize);
        clsArrayKey.def("slice", &ArrayKey<T>::slice);
    };

    PYBIND11_PLUGIN(_arrays) {
        py::module mod("_arrays", "Python wrapper for afw _arrays library");
    
        if (_import_array() < 0) {
                PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
                return nullptr;
        };

        /* Module level */
        declareArrayKey<float>(mod, "F");
        declareArrayKey<double>(mod, "D");

        return mod.ptr();
    }

In this case it is useful to make the ``get`` method in
``ArrayFKey`` and ``ArrayDKey`` more pythonic by allowing them to
accept slices as well as indices, so we create a new file ``arrays.py``
(notice the difference between this and the ``_arrays`` module, which is created by pybind11)
that begins with

.. code-block:: python


    from __future__ import absolute_import, division, print_function
    from ._arrays import ArrayFKey, ArrayDKey

We then define the function

.. code-block:: python

    def _getitem_(self, index):
        """
        operator[] in C++ only returns a single item, but `Array` has a method to get a slice of the
        array. To make the code more python we automatically check for a slice and return either
        a single item or slice as requested by the user.
        """
        if isinstance(index, slice):
            start, stop, stride = index.indices(self.getSize())
            if stride != 1:
                raise IndexError("Non-unit stride not supported")
            return self.slice(start, stop)
        return self._get_(index)

which uses the ``getSize``, ``slice``, and ``_get_`` methods defined in the pybind11 wrapper to
generate a slice (if necessary).
To make this the ``__getitem__`` method in ``ArrayFKey`` and ``ArrayIKey`` we add

.. code-block:: python


    ArrayFKey.__getitem__ = _getitem_
    ArrayDKey.__getitem__ = _getitem_
    del _getitem_

which assigns the ``__getitem__`` method to the classes and deletes the temporary function so that
it doesn't pollute the namespace.
Finally we must add ``from .arrays import *`` to ``tableLib.py`` to ensure that the stack updates
both classes. The complete ``arrays.py`` file should be

.. code-block:: python

    from __future__ import absolute_import, division, print_function
    from ._arrays import ArrayFKey, ArrayDKey

    def _getitem_(self, index):
        """
        operator[] in C++ only returns a single item, but `Array` has a method to get a slice of the
        array. To make the code more python we automatically check for a slice and return either
        a single item or slice as requested by the user.
        """
        if isinstance(index, slice):
            start, stop, stride = index.indices(self.getSize())
            if stride != 1:
                raise IndexError("Non-unit stride not supported")
            return self.slice(start, stop)
        return self._get_(index)

    ArrayFKey.__getitem__ = _getitem_
    ArrayDKey.__getitem__ = _getitem_


In most cases, the SWIG files from the current stack will contain the necessary python code and one can
simply copy and paste the code from the SWIG file into the new python file with little modification.

.. _fep:

Frequently Encountered Problems
===============================

There are a number of errors, issues, and other problems that you are likely to come across during wrapping.
This section has some hints on what might be causing a particular problem you are encountering.

Casting
-------

SWIG and pybind11 handle inheritance in different ways. In SWIG, if a class B inherits from A,
a pointer that clones B can return a type A, which is undesirable.
There was a lot of machinery, including a ``.cast`` method that was used to recase A as B.
This is not necessary with pybind11 so all casting procedures can be removed
(or at the very least commented out) and tests for casting can be skipped with a 
``@unittest.skip("Skip for pybind11")``.

.. _segfaults:

Segmentation Faults
-------------------

Smart Pointers
^^^^^^^^^^^^^^

The vast majority of the segfaults you encounter will be caused by inheriting a class that is defined
with a smart pointer, but not using the same pointer in the template definition of the new class
(see `smart_ptr`_). For example if a class A is defined using

.. code-block:: c++

    py::class_<A, std::shared_ptr<A>> clsA(mod, "A");

then a class B that inherits from A must include ``std::shared_ptr<B>``:

.. code-block:: c++

    py::class_<B, std::shared_ptr<B>, A> clsB(mod, "B");

NDArrays
^^^^^^^^

The other main cause of segfaults is forgetting to include

.. code-block:: c++

    #include "numpy/arrayobject.h"
    #include "ndarray/pybind11.h"
    #include "ndarray/converter.h"

and

.. code-block:: c++

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

when using ndarrays (see `ndarray`_).

Import Issues
-------------

You might find that a particular class has been wrapped in a different module,
but pybind11 fails to find a wrapped version of the class.
For instance, if class ``A`` is wrapped from header ``foo.h``,
and header ``bar.h`` has a class ``B`` with a method that returns an object with class ``A``,
then a python script using class ``B`` must import from both ``_foo`` and ``_bar``.
If module ``_bar`` will (nearly) always need classes or functions from ``_foo``,
it can be useful to add the following to module.py:

.. code-block:: python

    from _foo import A
    from _bar import *

where we make sure that any wrapped classes are always imported.

Missing or Broken Class Methods
-------------------------------

Sometimes a method called in a test is either not defined in the header or is defined but appears broken.
In many cases this is because there is a SWIG file in the current stack that extends the classes with
a more pythonic interface.
In some cases the methods are completely new while in others the C++ methods are overwritten.
To extend the classes in python see :ref:`python-code`.

.. _gitlock: https://github.com/lsst-dm/gitlock
.. _inheritance: https://pybind11.readthedocs.io/en/latest/classes.html#inheritance
