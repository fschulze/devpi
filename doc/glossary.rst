Common Terms
-----------------------------------------

.. glossary::
    index
        A (package) index is a repository for release metadata and release files.
        devpi-server allows the creation of multiple independent indices per user.

    remote index
        A remote index is a view to an external server like `pypi`_ and is read-only.

    local index
        A local index stores packages on the devpi server and can be uploaded to via
        packaging tools such as twine. By default files are replicated locally when
        requested by an installer like pip, but a configuration option exists to
        redirect to the original URL of the file on the external server.

    non volatile index
        An index which can not be deleted or modified in a destructive manner.
        Typically, **non volatile indexes** are indexes shared amongts users
        part of a project or an organization. For more details see
        :ref:`non_volatile_indexes`.

    volatile index
        A index that can be modified and deleted at will by its owner.
        On upload, release files will overwrite existing release files.
        A Development index is an example of volatile indexes. A volatile
        index normally pertains to a single user.

    mirror index
        These indexes mirror externally stored packages. By default *root/pypi* is
        such an index, which mirrors https://pypi.org/simple/.
        You can’t upload or push any packages to mirror indexes.
        They update themselves whenever they are used. For example when you try
        to install a package via pip.

    user
        A user can be a person, project or team.

    devpi server
        A devpi server is the server responsible for providing a PyPI index cache
        and managing sub-indices. The devpi server is compatible with most python
        installer tools.
    devpi client
        The devpi client provides the user interface (command line) to interface
        with the devpi server.

    acl
        Access Control List

    upload
        Action consisting of loading one or more release files from a file
        system to an index.

    push
        Action of transferring a package from one index to another.

    remove
        Action of deleting a project or a specific release from an index.

    release file
        A release file contains a project's artifact (.py files, etc) required
        for the installation. A release file is versioned. The term *package*
        is often use as well, but might be confused with a *python package*.

    volatile
        Indices have a boolean "volatile" property.  If true, release files can
        be deleted or replaced.  If False, attempts at modifications will
        cause an Error.
