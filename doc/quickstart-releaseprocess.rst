.. include:: links.rst

.. _quickstart-releaseprocess:

Quickstart: uploading, testing, pushing releases 
------------------------------------------------

This quickstart document walks you through setting up a self-contained
pypi release upload, testing and staging system for your Python packages.

Installing devpi client and server
++++++++++++++++++++++++++++++++++

We want to run the full devpi system on our laptop::

    pip install -U devpi-web devpi-client

Note that the ``devpi-web`` package will pull in the core
``devpi-server`` package.  If you don't want a web interface you 
can just install the latter only.

initializing a basic server and index
+++++++++++++++++++++++++++++++++++++

..
    $ rm -rf ~/.devpi/server

We need to perform a couple of steps to get an index
where we can upload and test packages:

- start a background devpi-server at ``http://localhost:3141``

- configure the client-side tool ``devpi`` to connect to the newly
  started server

- create and login a user, using as defaults your current login name 
  and an empty password.

- create an index and directly use it.

So let's first initialize devpi-server::

    $ devpi-init
    INFO  NOCTX Loading node info from /tmp/home/.devpi/server/.nodeinfo
    INFO  NOCTX generated uuid: 446e22e0db5e41a5989fd671e98ec30b
    INFO  NOCTX wrote nodeinfo to: /tmp/home/.devpi/server/.nodeinfo
    INFO  NOCTX DB: Creating schema
    INFO  [Wtx-1] setting password for user 'root'
    INFO  [Wtx-1] created user 'root'
    INFO  [Wtx-1] created root user
    INFO  [Wtx-1] created root/pypi index
    INFO  [Wtx-1] fswriter0: committed at 0

To start ``devpi-server`` in the background we use supervisor as an example.
First we create the config file for it::

    $ devpi-gen-config
    It is highly recommended to use a configuration file for devpi-server, see --configfile option.
    wrote gen-config/crontab
    wrote gen-config/net.devpi.plist
    wrote gen-config/launchd-macos.txt
    wrote gen-config/nginx-devpi.conf
    wrote gen-config/nginx-devpi-caching.conf
    wrote gen-config/supervisor-devpi.conf
    wrote gen-config/supervisord.conf
    wrote gen-config/devpi.service
    wrote gen-config/windows-service.txt

Then we start supervisord using a config which includes the generated file,
see :ref:`quickstart-server` for more details::

    $ supervisord -c gen-config/supervisord.conf

..
    $ waitforports -t 60 3141
    Waiting for 127.0.0.1:3141

Then we point the devpi client to it::

    $ devpi use http://localhost:3141
    using server: http://localhost:3141/ (not logged in)
    no current index: type 'devpi use -l' to discover indices
    venv for install/set commands: /tmp/docenv
    only setting venv pip/uv config, no global configuration changed
    /tmp/docenv/pip.conf: no config file exists
    /tmp/docenv/uv.toml: no config file exists
    always-set-cfg: no

Then we add our own "testuser"::

    $ devpi user -c testuser password=123
    user created: testuser

Then we login::

    $ devpi login testuser --password=123
    logged in 'testuser', credentials valid for 10.00 hours

And create a "dev" index, telling it to use the ``root/pypi`` cache as a base
so that all of pypi.org packages will appear on that index::

    $ devpi index -c dev bases=root/pypi
    http://localhost:3141/testuser/dev?no_projects=:
      type=stage
      bases=root/pypi
      volatile=True
      acl_upload=testuser
      acl_toxresult_upload=:ANONYMOUS:
      mirror_whitelist=
      mirror_whitelist_inheritance=intersection

Finally we use the new index::

    $ devpi use testuser/dev
    current devpi index: http://localhost:3141/testuser/dev (logged in as testuser)
    supported features: push-no-docs, push-only-docs, push-register-project, server-keyvalue-parsing
    venv for install/set commands: /tmp/docenv
    only setting venv pip/uv config, no global configuration changed
    /tmp/docenv/pip.conf: no config file exists
    /tmp/docenv/uv.toml: no config file exists
    always-set-cfg: no

We are now ready to go for uploading and testing packages.

.. _`quickstart_release_steps`:

devpi install: installing a package
+++++++++++++++++++++++++++++++++++

We can now use the ``devpi`` command line client to trigger a ``pip
install`` of a pypi package using the index from our already running server::

    $ devpi install pytest==6.2.4 attrs==23.1.0 iniconfig==2.0.0 pluggy==0.13.1 py==1.11.0 toml==0.10.2
    --> .$ /tmp/docenv/bin/pip --version
    --> .$ /tmp/docenv/bin/pip install -U pytest==6.2.4 attrs==23.1.0 iniconfig==2.0.0 pluggy==0.13.1 py==1.11.0 toml==0.10.2  [PIP_INDEX_URL=URL('http://****:****@localhost:3141/testuser/dev/+simple/'),PIP_PRE='1',PIP_USE_WHEEL='1']
    Looking in indexes: http://testuser:****@localhost:3141/testuser/dev/+simple/
    Collecting pytest==6.2.4
      Downloading http://localhost:3141/root/pypi/%2Bf/91e/f2131a9bd6be8/pytest-6.2.4-py3-none-any.whl (280 kB)
    Collecting attrs==23.1.0
      Downloading http://localhost:3141/root/pypi/%2Bf/1f2/8b4522cdc2fb4/attrs-23.1.0-py3-none-any.whl (61 kB)
    Collecting iniconfig==2.0.0
      Downloading http://localhost:3141/root/pypi/%2Bf/b6a/85871a79d2e3b/iniconfig-2.0.0-py3-none-any.whl (5.9 kB)
    Collecting pluggy==0.13.1
      Downloading http://localhost:3141/root/pypi/%2Bf/966/c145cd83c9650/pluggy-0.13.1-py2.py3-none-any.whl (18 kB)
    Collecting py==1.11.0
      Downloading http://localhost:3141/root/pypi/%2Bf/607/c53218732647d/py-1.11.0-py2.py3-none-any.whl (98 kB)
    Collecting toml==0.10.2
      Downloading http://localhost:3141/root/pypi/%2Bf/806/143ae5bfb6a3c/toml-0.10.2-py2.py3-none-any.whl (16 kB)
    Requirement already satisfied: packaging in /tmp/docenv/lib/python3.9/site-packages (from pytest==6.2.4) (25.0)
    Installing collected packages: toml, py, pluggy, iniconfig, attrs, pytest
    Successfully installed attrs-23.1.0 iniconfig-2.0.0 pluggy-0.13.1 py-1.11.0 pytest-6.2.4 toml-0.10.2

The ``devpi install`` command configured a pip call, using the
pypi-compatible ``+simple/`` page on our ``testuser/dev`` index for
finding and downloading packages.  The ``pip`` executable was searched
in the ``PATH`` and found in ``docenv/bin/pip``.

Let's check that ``pytest`` was installed correctly::

    $ py.test --version
    pytest 6.2.4

You may invoke the ``devpi install`` command a second time which will
even work when you have no network.

.. _`devpi upload`:

devpi upload: uploading one or more packages
++++++++++++++++++++++++++++++++++++++++++++

We are going to use ``devpi`` command line tool facilities for 
performing uploads (you can also 
:ref:`use plain setup.py <configure pypirc>`).

Let's verify we are logged in to the correct index::

    $ devpi use
    current devpi index: http://localhost:3141/testuser/dev (logged in as testuser)
    supported features: push-no-docs, push-only-docs, push-register-project, server-keyvalue-parsing
    venv for install/set commands: /tmp/docenv
    only setting venv pip/uv config, no global configuration changed
    /tmp/docenv/pip.conf: no config file exists
    /tmp/docenv/uv.toml: no config file exists
    always-set-cfg: no

Now go to the directory of a ``setup.py`` file of one of your projects  
(we assume it is named ``example``) to build and upload your package
to our ``testuser/dev`` index::

    example $ devpi upload
    using workdir /tmp/devpi0
    pre-build: cleaning dist
    --> .$ /tmp/docenv/bin/python -m build
    built: dist/example-1.0-py3-none-any.whl 1kb
    built: dist/example-1.0.tar.gz 1kb
    file_upload of example-1.0-py3-none-any.whl to http://localhost:3141/testuser/dev/
    file_upload of example-1.0.tar.gz to http://localhost:3141/testuser/dev/

There are three triggered actions:

- detection of a VCS (git/hg/svn/bazaar) repository, leading to copying all versioned
  files to a temporary work dir.  If you are not using mercurial,
  the copy-step is skipped and the upload operates directly on your source
  tree.

- registering the ``example`` release as defined in ``setup.py`` to 
  our current index

- building and uploading a ``gztar`` formatted release file from the
  workdir to the current index (using a ``setup.py`` invocation under
  the hood).

We can now install the freshly uploaded package::

    $ devpi install example
    --> .$ /tmp/docenv/bin/pip --version
    --> .$ /tmp/docenv/bin/pip install -U example  [PIP_INDEX_URL=URL('http://****:****@localhost:3141/testuser/dev/+simple/'),PIP_PRE='1',PIP_USE_WHEEL='1']
    Looking in indexes: http://testuser:****@localhost:3141/testuser/dev/+simple/
    Collecting example
      Downloading http://localhost:3141/testuser/dev/%2Bf/0b1/6414c21b576b1/example-1.0-py3-none-any.whl (1.4 kB)
    Installing collected packages: example
    Successfully installed example-1.0

This installed your just uploaded package from the ``testuser/dev``
index where we previously uploaded the package.

.. note::

    ``devpi upload`` allows to simultaneously upload multiple different 
    formats of your release files such as ``sdist.zip`` or ``bdist_egg``.
    The default is ``sdist.tgz``.



devpi test: testing an uploaded package
+++++++++++++++++++++++++++++++++++++++

If you have a package which uses tox_ for testing you may now invoke::

    $ devpi test --tox-args="-q" example  # package needs to contain tox.ini
    using workdir /tmp/devpi-test0
    only universal wheels supported, found example-1.0-py3-none-any.whl
    received http://testuser:****@localhost:3141/testuser/dev/+f/853/34ff3d48c83ba/example-1.0.tar.gz
    unpacking /tmp/devpi-test0/downloads/example-1.0.tar.gz to /tmp/devpi-test0/targz
    --> .$ /home/devpi/devpi/bin/tox --installpkg /tmp/devpi-test0/downloads/example-1.0.tar.gz --recreate --result-json /tmp/devpi-test0/targz/toxreport.json -c /tmp/devpi-test0/targz/example-1.0/tox.ini -q  [PIP_INDEX_URL=URL('http://****:****@localhost:3141/testuser/dev/+simple/')]
    ============================= test session starts ==============================
    platform linux -- Python 3.9.22, pytest-8.3.5, pluggy-1.6.0
    cachedir: .tox/py/.pytest_cache
    rootdir: /home/runner/work/devpi/devpi
    configfile: pyproject.toml
    collected 1 item

    test_example.py .                                                        [100%]

    ============================== 1 passed in 0.01s ===============================
      py: OK (10.57 seconds)
      congratulations :) (10.85 seconds)
    posting tox result data to http://localhost:3141/testuser/dev/+f/853/34ff3d48c83ba/example-1.0.tar.gz
    success

Here is what happened:

- devpi got the latest available version of ``example`` from the current
  index

- it unpacked it to a temp dir, found the ``tox.ini`` and then invoked
  tox, pointing it to our ``example-1.0.tar.gz``, forcing all installations
  to go through our current ``testuser/dev/+simple/`` index and instructing
  it to create a ``json`` report.

- after all tests ran, we send the ``toxreport.json`` to the devpi server
  where it will be attached precisely to our release file.
 
We can verify that the test status was recorded via::

    $ devpi list example
    http://localhost:3141/testuser/dev/+f/0b1/6414c21b576b1/example-1.0-py3-none-any.whl
    http://localhost:3141/testuser/dev/+f/853/34ff3d48c83ba/example-1.0.tar.gz

.. versionadded:: 2.6

With ``--index`` you can get the release from another index. Full URLs to
another devpi-server are also supported.

.. note::

    Since version 2.2.0 testing of universal wheels is supported if there
    also is an sdist which contains the necessary tox.ini and tests files.
    Wheels typically don't contain them as they are a pure installation
    package.

devpi push: staging a release to another index
++++++++++++++++++++++++++++++++++++++++++++++

Once you are happy with a release file you can push it either
to another devpi-managed index or to an outside pypi index server.

Let's create another ``staging`` index::

    $ devpi index -c staging volatile=False
    http://localhost:3141/testuser/staging?no_projects=:
      type=stage
      bases=
      volatile=False
      acl_upload=testuser
      acl_toxresult_upload=:ANONYMOUS:
      mirror_whitelist=
      mirror_whitelist_inheritance=intersection

We created a non-volatile index which means that one can not 
overwrite or delete release files. See :ref:`non_volatile_indexes` for more info
on this setting.

We can now push the ``example-1.0.tar.gz`` from above to
our ``staging`` index::

    $ devpi push example==1.0 testuser/staging
       200 register example 1.0 -> testuser/staging
       200 store_releasefile testuser/staging/+f/0b1/6414c21b576b1/example-1.0-py3-none-any.whl
       200 store_releasefile testuser/staging/+f/853/34ff3d48c83ba/example-1.0.tar.gz
       200 store_toxresult testuser/staging/+f/853/34ff3d48c83ba/example-1.0.tar.gz.toxresult-20210510144323-0

This will determine all files on our ``testuser/dev`` index belonging to
the specified ``example==1.0`` release and copy them to the
``testuser/staging`` index. 

devpi push: releasing to an external index
++++++++++++++++++++++++++++++++++++++++++

Let's check again our current index::

    $ devpi use
    current devpi index: http://localhost:3141/testuser/dev (logged in as testuser)
    supported features: push-no-docs, push-only-docs, push-register-project, server-keyvalue-parsing
    venv for install/set commands: /tmp/docenv
    only setting venv pip/uv config, no global configuration changed
    /tmp/docenv/pip.conf: no config file exists
    /tmp/docenv/uv.toml: no config file exists
    always-set-cfg: no

Let's now use our ``testuser/staging`` index::

    $ devpi use testuser/staging
    current devpi index: http://localhost:3141/testuser/staging (logged in as testuser)
    supported features: push-no-docs, push-only-docs, push-register-project, server-keyvalue-parsing
    venv for install/set commands: /tmp/docenv
    only setting venv pip/uv config, no global configuration changed
    /tmp/docenv/pip.conf: no config file exists
    /tmp/docenv/uv.toml: no config file exists
    always-set-cfg: no

and check the test result status again::

    $ devpi list example
    http://localhost:3141/testuser/staging/+f/0b1/6414c21b576b1/example-1.0-py3-none-any.whl
    http://localhost:3141/testuser/staging/+f/853/34ff3d48c83ba/example-1.0.tar.gz

Good, the test result status is still available after the push
from the last step.

We may now decide to push this release to an external
pypi-style index which we have configured in the ``.pypirc`` file::

    $ devpi push example-1.0 pypi:testrun
    no pypirc file found at: /tmp/home/.pypirc

this will push all release files of the ``example-1.0`` release
to the external ``testrun`` index server, using credentials
and the URL found in the ``pypi`` section in your
``.pypirc``.

index inheritance re-configuration
++++++++++++++++++++++++++++++++++

At this point we have the ``example-1.0`` release and release file
on both the ``testuser/dev`` and ``testuser/staging`` indices.
If we rather want to always use staging packages in our development
index, we can reconfigure the inheritance 
``bases`` for ``testuser/dev``::

    $ devpi index testuser/dev bases=testuser/staging
    /testuser/dev bases=testuser/staging
    http://localhost:3141/testuser/dev?no_projects=:
      type=stage
      bases=testuser/staging
      volatile=True
      acl_upload=testuser
      acl_toxresult_upload=:ANONYMOUS:
      mirror_whitelist=
      mirror_whitelist_inheritance=intersection

If we now switch back to using ``testuser/dev``::

    $ devpi use testuser/dev
    current devpi index: http://localhost:3141/testuser/dev (logged in as testuser)
    supported features: push-no-docs, push-only-docs, push-register-project, server-keyvalue-parsing
    venv for install/set commands: /tmp/docenv
    only setting venv pip/uv config, no global configuration changed
    /tmp/docenv/pip.conf: no config file exists
    /tmp/docenv/uv.toml: no config file exists
    always-set-cfg: no

and look at our example release files::

    $ devpi list example
    http://localhost:3141/testuser/dev/+f/0b1/6414c21b576b1/example-1.0-py3-none-any.whl
    http://localhost:3141/testuser/dev/+f/853/34ff3d48c83ba/example-1.0.tar.gz
    http://localhost:3141/testuser/staging/+f/0b1/6414c21b576b1/example-1.0-py3-none-any.whl
    http://localhost:3141/testuser/staging/+f/853/34ff3d48c83ba/example-1.0.tar.gz

we'll see that ``example-1.0.tar.gz`` is contained in both
indices.  Let's remove the ``testuser/dev`` ``example`` release::

    $ devpi remove -y example
    About to remove the following releases and distributions
    version: 1.0
      - http://localhost:3141/testuser/dev/+f/0b1/6414c21b576b1/example-1.0-py3-none-any.whl
      - http://localhost:3141/testuser/dev/+f/853/34ff3d48c83ba/example-1.0.tar.gz
      - http://localhost:3141/testuser/dev/+f/853/34ff3d48c83ba/example-1.0.tar.gz.toxresult-20210510144323-0
    Are you sure (yes/no)? yes (autoset from -y option)

If you don't specify the ``-y`` option you will be asked to confirm
the delete operation interactively.

The ``example-1.0`` release remains accessible through ``testuser/dev``
because it inherits all releases from its ``testuser/staging`` base::

    $ devpi list example
    http://localhost:3141/testuser/staging/+f/0b1/6414c21b576b1/example-1.0-py3-none-any.whl
    http://localhost:3141/testuser/staging/+f/853/34ff3d48c83ba/example-1.0.tar.gz

Now shutdown supervisord which was started at the beginning of this tutorial::

    $ supervisorctl -c gen-config/supervisord.conf shutdown
    Shut down

running devpi-server permanently
+++++++++++++++++++++++++++++++++

If you want to configure a permanent devpi-server install,
you can go to :ref:`quickstart-server` to get some help.


