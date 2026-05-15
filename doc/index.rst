Preface
=================================================================

.. include:: links.rst

.. sidebar:: Links and contact

   `issue tracker <https://github.com/devpi/devpi/issues>`_,
   `mailing list <https://mail.python.org/mm3/mailman3/lists/devpi-dev.python.org/>`_
   `repo of server/web/client <https://github.com/devpi/devpi/>`_

Introduction and Goals
-----------------------------------------
The Goal of `devpi`_ is to provide a fast and reliable package cache to `pypi`_
as well as a mechanism to share packages in various states of development
amongst users and developers, prior to pushing to the *outside world*.

This implies that users can:

   * Work unaffected if PyPI fails
   * store "closed source" packages internally, that can be accessed like
     any other packages as if they were residing on PyPI.

Getting Started
-----------------------------------------
Getting started with devpi, setting up a client and server, and pushing packages.

.. toctree::
  :maxdepth: 3

  tutorial/index

.. include:: glossary.rst

Useful Links
-----------------------------------------

* Uisng Docker with Devpi:  https://github.com/JonasAlfredsson/docker-devpi


* If you want to help the project, you can visit the :doc:`contribution/index` section

* Bugs can be reported on https://github.com/devpi/devpi/issues

* For professional support, contact mail (at) pyfidelity.com

Documentation Overview
-----------------------------------------
.. toctree::
  :maxdepth: 2

  Tutorials, and Getting Started <tutorial/index>
  Using devpi and devpi server <usage/index>
  server-admin/index
  Customization <customizing/index>
  Command line options, index, and user settings <reference/index>
  Plugin and HTTP API Reference <developing/index>
  Release Announcements <appendix/release-announcements>
  Changelogs <appendix/changelogs/index>

.. include:: contribution/index.rst
