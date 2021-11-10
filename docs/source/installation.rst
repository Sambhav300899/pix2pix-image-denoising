Installation
================
To install the library follow the instructions below.

Installing dependencies
-------------------------
Install SNAP and GDAL according to the instructions given `here <https://spacesenseai.atlassian.net/wiki/spaces/TECH/pages/237371393/Configure+a+GCP+VM>`_

Installing spacesenseds
-------------------------
Clone the library

.. code-block:: bash

    git clone https://gitlab.com/spacesense/spacesense-labs/spacesense-ds/-/tree/master/src/spacesenseds

Install the library

.. code-block:: bash

    cd spacesenseds
    pip install .

Testing your installation
--------------------------
run pytest in the root of the library. If it finishes running without any errors then spacesenseds has been
installed successfully.

.. code-block:: bash

    pytest
