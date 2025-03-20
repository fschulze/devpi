def test_macros(dummyrequest, pyramidconfig):
    pyramidconfig.include("pyramid_chameleon")
    pyramidconfig.include("devpi_web.macroregistry")
    pyramidconfig.scan("devpi_web.macros")
    macros = dummyrequest.registry["macros"]
    assert {k: macros.get_group(k) for k in macros.get_groups()} == {
        "html_head": [
            "favicon",
            "html_head_css",
            "html_head_scripts",
        ],
        "index": [
            "index_packages",
            "index_description",
            "index_permissions",
            "index_bases",
            "index_whitelist",
        ],
        "index_header": [
            "title",
            "subnavigation",
        ],
        "main_footer": [
            "footer_about",
            "footer_versions",
        ],
        "main_header": [
            "header_status",
        ],
        "main_header_top": [
            "logo",
            "header_search",
        ],
        "main_navigation": [
            "header_breadcrumbs",
            "status_badge",
        ],
        "project": [
            "blocked_indexes",
            "project_refresh",
            "project_latest_version",
            "project_versions",
        ],
        "project_header": [
            "title",
            "subnavigation",
        ],
        "root": [
            "user_index_list",
        ],
        "user": [
            "user_index_list",
        ],
        "version": [
            "version_metadata",
            "blocked_indexes",
            "version_files",
            "version_description",
            "version_docs",
        ],
        "version_header": [
            "title",
            "subnavigation",
            "version_summary",
        ],
    }
