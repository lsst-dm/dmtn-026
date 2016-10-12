..
  Technote content.

:tocdepth: 1
.. Please do not modify tocdepth; will be fixed when a new Sphinx theme is shipped.

.. _scope:

Scope of the document
=====================

This document is designed to assist developers involved in porting LSST Science Pipelines
from swig to pybind11. For more details on the styling conventions see ??.

.. _intro:

.. _installation:

Installation
============

To install all of the currently wrapped code with lsstsw, use

.. code-block:: bash

    rebuild -r tickets/DM-6168 {{package name}}

where {{package name}} is the name of the package that is currently being wrapped (for instance afw).
This will build the most up to date version of the stack that has been ported to pybind11, as the tests that have not been wrapped are all commented out (see section :ref:`activate_test`).

Don't forget to tag the new build as current with EUPS:

.. code-block:: bash

    eups tags --clone bNNNN current

where bNNN is the current build number.

You will also need to install `gitlock`_ to allow you to get a soft lock on files that you are working on.
Follow the instructions on the `gitlock`_ page for details.

.. _new_package:

Wrapping a new package
======================

Since many packages have C++ classes and functions that are not exposed to Python, there are large chunks of code that are not currently tested explicitely.
Without tests there is no way to know whether these methods have been properly wrapped, so for now we only wrap methods that are called from Python tests.
A single test might import from multiple submodules of the current package, so we found that it is more efficient to wrap by test as opposed to wrapping by module.
The following section outlines the procedure to begin wrapping a new package.

Preparing Package
-----------------

Before you can begin wrapping a package it is useful to make templates of all of the header files that will be filled in during the pybind11 port. After creating a new ticket branch for pybind11, this can be done by running the ``build_templates`` script using

.. code-block:: bash

    python build_templates <python path> <header path>

where ``python path`` is the full path to the main location of the Python packages
(for example ``afw/python/lsst/afw``) and ``header path`` is the location of the include files
(for example ``afw/include/lsst/afw``).

This is step is only necessary if you are the first developer wrapping a new package, otherwise the template files have already been created.
Don't forget to commit these changes and push to the github remote since others will likely need to work on the same package.

.. _new_test:

Wrapping a New Test
===================

Setup
-----

Since the stack has been built using lsstsw, you can simply use

.. code-block:: bash

    cd <repository directory>
    setup -r .

to setup the package currently being wrapped.

.. _locking:

Rebasing and Locking Files
--------------------------

Because the pybind11 stack is a fork of the master lsst packages, frequent rebasing will occur throughout the pybind11 port.
Additionally, the while we strive to have different developers work as much as possible on independent packages, the numerous
interdependencies will sometimes require working on the same package and even in the same ticket branch.
Thus frequent pushing and rebasing is necessary to keep everyones stack up to date.
To prevent unnecessary merge conflicts it is best to lock files that you are currently working on.
Git does not have a true lock, in that locking a file does not prevent others from working on it and pushing their commits.
Instead the `gitlock`_ package can be used to lock a particular file and notify the group that a file is being worked on. Once gitlock is setup you can lock a file by using

.. code-block:: bash

    gitlock lock {{package name}} -f <relative path to the file>

The script will notify you if you were able to successfully lock the file or if it is already locked by another user. Once you have finished working on a file, using

.. code-block:: bash

    gitlock unlock {{package name}} -f <relative path to the file>

will unlock the file and allow others to work on it.

.. warning::

    Remember that gitlocks do not prevent you or other users from modifying files and committing changes.
    Do your best to be considerate of other developers and try to lock and unlock files as needed.

.. _activate_test:

Activate the Test
-----------------

All of the tests that have yet to be wrapped are commented out using the tag ``#pybind11#``. The script "activate_test.py" can be used to remove the comments so that the test runs properly.
For tests with multiple test classes and methods it may be useful to add the decorator ``@unittest.skip("temporary skip wihile wrapping")`` to functions in a test that have not been wrapped yet.

Tutorial
========

To illustrate how to wrap a test we will use ``afw/tests/testMinimize.py`` as an example. We start by cloning https://github.com/lsst/afw to our local machine and checkout the correct ticket branch for the current test. In this case ``testMinimize.py`` is in ``tickets/DM-6298``, so we checkout that branch and set it up with ``setup -r .`` from the main ``afw`` repository directory.

Compiling the Code
------------------

Before we make any changes it's a good idea to compile the cloned repository to make sure that everything is setup correctly. From the ``afw`` repository main directory run

.. code-block:: shell

    scons

to build afw.
Since this is your first build of afw it will take a while but as you make changes, using

.. code-block:: shell

    scons python lib

only builds the newly wrapped headers, so development is much faster than with SWIG).

Locking Files
-------------

Before we start working we want to lock the current test using

.. code-block:: bash

    gitlock lock afw -f tests/testMinimize.py

from the main afw repository directory (see :ref:`locking` for more on locking and unlocking files).
Next we activate the test. Enter the ``pb11_scripts`` directory and type

.. code-block:: bash

    python activate_test.py <path to test>

For example, if afw is contained in ``$LSST/code/afw`` use

.. code-block:: bash

    python activate_test $LSST/code/afw/tests/testMinimize.py

This removes all of the lines commented out to allow pybind11 to build the package.

.. _test_minimize:

testMinimize.py
---------------

In this case the only test class, ``MinimizeTestCase``, imports two functions from ``afw.math``: ``PolynomialFunction2D`` from ``afw/math/functionLibrary.h`` and ``minimize`` from ``afw/math/minimize.h``:

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

We first have to tell scons about the new header we want to wrap, so we modify ``python/lsst/afw/math/SConscript`` to read.

.. code-block:: python

    from lsst.sconsUtils import scripts
    scripts.BasicSConscript.pybind11(['minimize'])

.. note::

    It is important to change ``scripts.BasicSConscript.python`` (which uses swig) to ``scripts.BasicSConscript.pybind11`` (which uses pybind11).

We also need to tell Python to import the new modules in ``python/lsst/afw/math/mathLib.py``, where we add the line

.. code-block:: python

    from __future__ import absolute_import
    from ._minimize import *

Since we are wrapping the header file ``minimize.h`` we must make sure to include it in ``minimize.cc`` (which is the previously created pybind11 template):

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


We notice that ``minimize`` is a function that returns type ``FitResults``, and since ``FitResults`` is an ordinary structure we will wrap it first.
In ``minimize.cc``, ``PYBIND11_PLUGIN`` contains the code to initialize the Python module ``minimize``, and all of the methods will be placed in this structure.
So inside the ``PYBIND11_PLUGIN`` structure, and after the module declaration ``py::module mod("_minimize", "Python wrapper for afw _minimize library");`` we add

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

At this time ``minimize.cc`` should look like

.. code-block:: c++

    #include <pybind11/pybind11.h>
    //#include <pybind11/operators.h>
    #include <pybind11/stl.h>

    #include "lsst/afw/math/minimize.h"

    namespace py = pybind11;

    using namespace lsst::afw::math;

    PYBIND11_PLUGIN(_minimize) {
        py::module mod("_minimize", "Python wrapper for afw _minimize library");

        py::class_<FitResults> clsFitResults(mod, "FitResults");

        clsFitResults.def_readwrite("isValid", &FitResults::isValid);
        clsFitResults.def_readwrite("chiSq", &FitResults::chiSq);
        clsFitResults.def_readwrite("parameterList", &FitResults::parameterList);
        clsFitResults.def_readwrite("parameterErrorList", &FitResults::parameterErrorList);

        return mod.ptr();
    }

This is a good time to build our changes (at times the error messages generated by pybind11 can be obscure so it is useful to recompile after each wrapped class).
From the shell prompt run

.. code-block:: bash

    scons python lib

to build all of the changes you made to afw.
If the build failed, go back and verify that all of your function definitions used the correct syntax as displayed above.

Wrapping an overloaded function
-------------------------------

Now that we have created the ``FitResults`` structure we can create our ``minimize`` function wrapper.
This is done using the ``def`` method of ``py::module``, where we must create a definition for each set of parameters.
Looking in the swig ``.i`` file located at https://github.com/lsst/afw/blob/master/python/lsst/afw/math/minimize.i we see that there are two templated types: ``float`` and ``double``.

.. note::

    Whenever you encounter a problem that requires you to look at the swig files you are best off looking at the code on github, as the swig files have been deleted in the pybind11 branch and switching branches locally will require you to commit or stash your changes, which might be inconvenient at the time.

In a minute we will wrap ``minimize`` for both types, but it is useful to first look at how this would be done for a single type ``double``.
In this case we define ``minimize`` and cast it to a ``FitResults`` function pointer underneath our ``clsFitResults`` code using

.. code-block:: c++

    mod.def("minimize", (FitResults (*) (lsst::afw::math::Function1<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         double)) minimize<double>);

Notice that for each parameter in the C++ function we include the type (including a reference if necessary) in our pybind11 function declaration but not the variable name itself.
Similarly, beneath this code we add the second set of parameters for the overloaded function

.. code-block:: c++

    mod.def("minimize", (FitResults (*) (lsst::afw::math::Function2<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         std::vector<double> const &,
                                         double)) minimize<double>);

We could copy these lines and change the templates to use type ``float`` if we wanted to, or instead we can write a function that allow us to template an arbitrarily large number of different types. This is not necessary with only two function types but it is useful to wrap them this way anyway for clarity, and as an exercise to illustrate how this is done in pybind11.

Between the namespace declaration (``using namespace lsst::afw::math;``) and start of the plugin (``PYBIND11_PLUGIN(``) lines we can define a template function to declare the minimize function.

.. code-block:: c++

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

Notice that the only changes we made to the function definition was to change ``lsst::afw::math::Function1<double>`` to ``lsst::afw::math::Function1<ReturnT>`` and ``minimize<double>`` to ``minimize<ReturnT>`` in both definitions. Now we can replace the ``mod.def("minimize", ...`` definitions in ``PYBIND11_PLUGIN`` with

.. code-block:: c++

    declareMinimize<double>(mod);
    declareMinimize<float>(mod);

which declares both templates for minimize.
Putting it all together, the file ``minimize.cc`` should look like

.. code-block:: c++

    #include <pybind11/pybind11.h>
    //#include <pybind11/operators.h>
    #include <pybind11/stl.h>

    #include "lsst/afw/math/minimize.h"

    namespace py = pybind11;

    using namespace lsst::afw::math;

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

.. _wrap_suffix:

Wrapping a Template with a suffix
---------------------------------

We still have not successfully wrapped all of the classes and functions need to run ``testMinimize.py``, as we haven't wrapped PolynomialFunction2D in ``afw/math/functionLibrary.py``.
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

So we begin with ``Function`` in ``afw/math/FunctionLibrary.h`` by adding ``'functionLibrary'`` to ``afw/math/SConscript``, ``from ._functionLibrary import *`` to ``mathLib.py``, and ``#include "lsst/afw/math/FunctionLibrary.h"`` in ``functionLibrary.cc`` just like we did for ``minimize.h`` in :ref:`new_cpp`.

Below ``using namespace lsst::afw::math;`` and before ``PYBIND11_PLUGIN`` we create the new template function

.. code-block:: c++

    template <typename ReturnT>
    void declarePolynomialFunctions(py::module &mod, const std::string & suffix) {
    };

where ``suffix`` will be a string that represents the return type of the function ("D" for double, "I" for int, etc.).
Inside the function we declare our class

.. code-block:: c++

        py::class_<PolynomialFunction2<ReturnT>, BasePolynomialFunction2<ReturnT>>
            clsPolynomialFunction2(mod, ("PolynomialFunction2" + suffix).c_str());

This is slightly different than our class declaration in :ref:`wrap_struct` because ``PolynomialFunction2`` inherits from ``BasePolynomialFunction2``, which can be seen in the above declaration.
Since ``BasePolynomialFunction2`` is defined in ``Function.h`` we must ``#include "lsst/afw/math/Function.h"`` at the beginning of ``functionLibrary.cc``.
We will discuss inheritance more in :ref:`wrapping_inheritance`.
Also notice that we combine ``PolynomialFunction2`` with the suffix, specified when ``declarePolyomialFunctions`` is defined, that specified the type for the function (for example "D" or "I").

We notice that the constructor is overloaded, so we define ``init`` with both sets of parameters

.. code-block:: c++

    clsPolynomialFunction2.def(py::init<unsigned int>());
    clsPolynomialFunction2.def(py::init<std::vector<double> const &>());


We must also declare the classes in the module, so inside ``PYBIND11_PLUGIN`` and beneath the module declaration ``py::module mod("_functionLibrary", "Python wrapper for afw _functionLibrary library");`` we add

.. code-block:: c++

    declarePolynomialFunctions<double>(mod, "D");

where we use the ``double`` type since ``PolynomialFunction2D`` is the method called from ``testMinimize.py`` and specify ``suffix`` as ``"D"``.

The last piece to wrap in ``functionLibrary.cc`` is the ``__call__`` method, since ``testMinimize.py`` makes use of it.
Most operators can be wrapped with the helpers in ``pybind11/operators.h``, but for function call we need to specify the operator
ourselves by binding a lambda.

.. code-block:: c++

    clsPolynomialFunction2.def("__call__", [](PolynomialFunction2<ReturnT> &t, double &x, double &y)
        -> ReturnT {
            return t(x,y);
    }, py::is_operator());

.. note

    The ``py::is_operator()`` informs pybind11 that the wrapped function is an operator which should trigger a ``NotImplementedError``
    instead of a ``TypeError`` when called with the wrong type.

where we call the C++ ``operator()`` function from the lambda.
At this point ``functionLibrary.cc`` should look like:

.. code-block:: c++

    #include <pybind11/pybind11.h>
    //#include <pybind11/operators.h>
    //#include <pybind11/stl.h>

    #include "lsst/afw/math/functionLibrary.h"
    #include "lsst/afw/math/Function.h"

    namespace py = pybind11;

    using namespace lsst::afw::math;

    template <typename ReturnT>
    void declarePolynomialFunctions(py::module &mod, const std::string & suffix) {
       py::class_<PolynomialFunction2<ReturnT>, BasePolynomialFunction2<ReturnT>>
            clsPolynomialFunction2(mod, ("PolynomialFunction2" + suffix).c_str());
        clsPolynomialFunction2.def(py::init<unsigned int>());
        clsPolynomialFunction2.def(py::init<std::vector<double> const &>());

        /* Operators */
        clsPolynomialFunction2.def("__call__", [](PolynomialFunction2<ReturnT> &t, double &x, double &y) -> ReturnT {
                return t(x,y);
        }, py::is_operator());
    };

    PYBIND11_PLUGIN(_functionLibrary) {
        py::module mod("_functionLibrary", "Python wrapper for afw _functionLibrary library");

        declarePolynomialFunctions<double>(mod, "D");

        return mod.ptr();
    }

Of course the test will still not runs since ``PolynomialFunction2`` depends on the methods ``setParameters`` and ``getNParameters`` that are inherited.

.. _wrapping_inheritance:

Inheritance
-----------

Now we journey down the rabbit hole that is inheritance and see that ``BasePolynomialFunction2`` inherits from ``Function2`` which inherits from ``Function``, which inherits from classes outside of afw. In many cases, it may not be necessary to include all of the inherited classes as use of the inherited classes might only be used in the C++ code, so we begin with ``BasePolynomialFunction2`` and work our way down. This is consistent with our workflow to only wrap the necessary methods to pass a test and as a bonus can save a significant amount of build time.

So we begin with ``Function`` in ``afw/math/Function.h`` by adding ``'function'`` to ``afw/math/SConscript``, ``from ._function import *`` to ``mathLib.py``, and ``#include "lsst/afw/math/Function.h"`` in ``function.cc`` just like we did for ``minimize.h`` in :ref:`new_cpp` and ``functionLibrary.h`` in :ref:`wrap_suffix`.

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

In this case ``Function``, ``Function2`` and ``BasePolynomialFunction2`` are all templated on the same type. So we declare them together in one function template.

.. code-block:: c++

    template<typename ReturnT>
    void declareFunctions(py::module &mod, const std::string & suffix){
    };

just like we did in :ref:`wrap_suffix`.
As mentioned above, we should not assume that we need to inherit from ``Function2`` but in this case we see that ``BasePolynomialFunction2`` is still missing the ``setParamters`` and ``getNParameters`` methods that are needed in ``PolynomialFunction2``, so we inherit from ``Function2`` in the declaration we add to ``declareFunctions``:

.. code-block:: c++

    py::class_<BasePolynomialFunction2<ReturnT>, Function2<ReturnT> >
        clsBasePolynomialFunction2(mod, ("BasePolynomialFunction2" + suffix).c_str());

There are no other methods of ``BasePolynomialFunction`` needed for the current test so we move on to ``Function2``, with the relevant code below:

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
In this case it is not immediately obvious that we will need the latter class available to Python, so we only include ``Function`` in our class declaration (which we place before our ``BasePolynomialFunction2`` declaration)

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

We see that ``Function`` also has multiple inheritances but for now we ignore them (as it does not appear that we necessarily need them exposed to Python) when we declare it:

.. code-block:: c++

    py::class_<Function<ReturnT>> clsFunction(mod, ("Function"+suffix).c_str());

The constructor is overloaded so beneath the class declaration we need to define ``init`` for both sets of parameters:

.. code-block:: c++

    clsFunction.def(py::init<unsigned int>());
    clsFunction.def(py::init<std::vector<double> const &>());

Recall from :ref:`test_minimize` that two methods of ``PolynomialFunction2D`` are needed that are defined in ``Function``: ``getNParameters`` and ``setParameters``, so we define them with

.. code-block:: c++

     clsFunction.def("getNParameters", &Function<ReturnT>::getNParameters);
     clsFunction.def("setParameters", &Function<ReturnT>::setParameters);

There are no other ``Function`` methods needed for now, so we leave wrapping them for the future if they are necessary on the Python side of the stack.

At this point ``function.cc`` should look like

.. code-block:: c++

    #include <pybind11/pybind11.h>
    //#include <pybind11/operators.h>
    #include <pybind11/stl.h>

    #include "lsst/afw/math/Function.h"

    namespace py = pybind11;

    using namespace lsst::afw::math;

    template<typename ReturnT>
    void declareFunctions(py::module &mod, const std::string & suffix){
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

    PYBIND11_PLUGIN(_function) {
        py::module mod("_function", "Python wrapper for afw _function library");

        declareFunctions<double>(mod, "D");

        return mod.ptr();
    }

and you should be able to compile the code (hopefully you have been building after each new class or you could come across multiple errors at this point) using ``scons python lib``.
You should now be able to run ``py.test tests/testMinimize.py`` and pass all of the tests.

testInterpolate.py
------------------

There are still multiple edge cases we have yet to encounter, including virtual funcitons, ndarrays, and enum types. All of these cases are needed to wrap testInterpolate.py with pybind11, so we use it to illustrate these procedures.

.. code-block:: c++

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

        #@unittest.skip("testing")
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

Here we see that there is only one class called from this test: ``lsst::afw::math::Interpolate``. We make sure to add the appropriate lines to ``mathLib.py``, ``Sconscript``, and ``interpolate.cc`` as we saw in :ref:`new_cpp`.

Smart Pointers
^^^^^^^^^^^^^^

We declare the class in the standard way, adding

.. code-block:: c++

    py::class_<Interpolate, std::shared_ptr<Interpolate>> clsInterpolate(mod, "Interpolate");

to the module section of ``interpolate.cc``.
``Interpolate`` itself is a virtual class and makes use of a shared pointer (see :ref:`virtual_functions`), so notice that here we added ``std::shared_ptr<Interpolate>`` as an inherited class, which is necessary to access Interpolate as a pointer.

Enum types
^^^^^^^^^^

The first method is an enum called ``Style``.
We declare a value for each keyword that points to the corresponding value in the header file, with an ``export_values()`` method at the end:

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

Virtual Functions and Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Notice that ``Interpolate`` is a virtual class that cannot be called directly.
Through examination of ``testInterpolate.py`` we see that ``Interpolate`` objects are created by using the ``makeInterpolate`` function, which is of type ``PTR(Interpolate)``.
We will wrap ``makeInterpolate`` in :ref:`function_kwargs` but first we finish wrapping ``Interpolate``.
The main function is the method ``interpolate``, which can be called with a double, list, or ndarray, however calling interpolate with a double is actually a virtual function, so we cannot wrap it directly.

Instead we create a lambda function:

.. code-block:: c++

    clsInterpolate.def("interpolate", [](Interpolate &t, double const x) -> double {
            return t.interpolate(x);
    });

This defines the function ``Interpolate::interpolate``, which then calls the virtual function ``interpolate`` of the ``Interpolate`` object directly (the method exists, just not in a way that it can be wrapped by pybind11).

NDArray's
^^^^^^^^^

Since the ``interpolate`` method is an overloaded function, only one of which is virtual, we can wrap the other function definitions in the traditional way:

.. code-block:: c++

    clsInterpolate.def("interpolate",
                       (std::vector<double> (Interpolate::*) (std::vector<double> const&) const)
                           &Interpolate::interpolate);
    clsInterpolate.def("interpolate",
                       (ndarray::Array<double, 1> (Interpolate::*) (ndarray::Array<double const, 1> const&)
                           const) &Interpolate::interpolate);

However, since we are using ndarray's we also need to include the numpy and ndarray headers at the top of ``interpolate.cc``

.. code-block:: c++

    #include "numpy/arrayobject.h"
    #include "ndarray/pybind11.h"
    #include "ndarray/converter.h"

It is also necessary to check that numpy has been installed and setup (otherwise unexpected segfaults will occcur), so in the module definition we add

.. code-block:: c++

    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
        }

.. _function_kwargs:

Wrapping Functions with Default Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final method remaining to wrap in ``interpolate.h`` is ``makeInterpolate``, which creates an ``Interpolate`` object from the virtual class.

This is an overloaded function, so we define it in the usual way but add ``py::arg("paremeter")`` for *all* of the arguments of the function (not just the ones that we need to give default values).
In this case

.. code-block:: c++

    mod.def("makeInterpolate", 
                       (PTR(Interpolate) (*)(std::vector<double> const &,
                                             std::vector<double> const &,
                                             Interpolate::Style const)) makeInterpolate,
                       py::arg("x"), py::arg("y"), py::arg("style")=Interpolate::AKIMA_SPLINE);
    mod.def("makeInterpolate", 
                       (PTR(Interpolate) (*)(ndarray::Array<double const, 1> const &,
                                             ndarray::Array<double const, 1> const &y,
                                             Interpolate::Style const)) makeInterpolate,
                       py::arg("x"), py::arg("y"), py::arg("style")=Interpolate::AKIMA_SPLINE);

This can be slightly simplified by adding ``using namespace pybind11::literals;`` to the top of ``interpolate.cc``, which allows us to replace ``py:arg("parameter")`` with ``"parameter"_a``.

.. note::

    If pybind11 returns an error during wrapping that the number of arguments does not match, check that you have wrapped all of the arguments with the correct types. Also make sure that you are defining the function in the correct place (ie. is it defined in the module or inside of a class).

.. _wrapped_interpolate:

Wrapped interpolate.cc
^^^^^^^^^^^^^^^^^^^^^^

When finished ``interpolate.cc`` should look like:

.. code-block:: c++

    #include <pybind11/pybind11.h>
    //#include <pybind11/operators.h>
    #include <pybind11/stl.h>

    #include "numpy/arrayobject.h"
    #include "ndarray/pybind11.h"
    #include "ndarray/converter.h"

    #include "lsst/afw/math/interpolate.h"

    namespace py = pybind11;
    using namespace pybind11::literals;

    using namespace lsst::afw::math;

    PYBIND11_DECLARE_HOLDER_TYPE(MyType, std::shared_ptr<MyType>);

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

.. _fep:

Frequently Encountered Problems
===============================

There are a number of errors, issues, and other problems that you are likely to come across during wrapping.
This section has some hints on what might be causing a particular problem you are encountering.

Casting
-------

SWIG and pybind11 handle inheritance in different ways. In SWIG, if a class B inherits from A, a pointer that clones B can return a type A, which is undesirable. There was a lot of machinery, including a ``.cast`` method that was used to recase A as B. This is not necessary with pybind11 so all casting procedures can be removed (or at the very least commented out) and tests for casting can be skipped with a ``@unittest.skip("Skip for pybind11")``.

.. _gitlock: https://github.com/lsst-dm/gitlock
.. _inheritance: https://pybind11.readthedocs.io/en/latest/classes.html#inheritance
