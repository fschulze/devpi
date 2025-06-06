# Makefile for Sphinx documentation
#

VERSION       = $(shell python -c "import conf ; print(conf.version)")
DOCZIP        = devpi-$(VERSION).doc.zip
# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

export HOME=/tmp/home
export TESTHOME=$(HOME)

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

# This variable is not auto generated as the order is important.
USER_MAN_CHAPTERS = devpi_commands\
                    devpi_user\
                    devpi_indices\
                    devpi_packages\
#                 userman/index.rst\
#                 userman/devpi_misc.rst\
#                 userman/devpi_concepts.rst\
          
                                 
#export DEVPI_CLIENTDIR=$(CURDIR)/.tmp_devpi_user_man/client
#export DEVPISERVER_SERVERDIR=$(CURDIR)/.tmp_devpi_user_man/server
export PIP_PROGRESS_BAR=off

chapter = commands

.PHONY: help clean html dirhtml singlehtml pickle json htmlhelp qthelp devhelp \
        epub latex latexpdf text man changes linkcheck doctest gettext install \
        quickstart-releaseprocess quickstart-pypimirror quickstart-server regen \
        prepare-quickstart\
        regen.server-fresh regen.server-clean\
        regen.uman-all regen.uman

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  texinfo    to make Texinfo files"
	@echo "  info       to make Texinfo files and run them through makeinfo"
	@echo "  gettext    to make PO message catalogs"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"
	@echo 
	@echo "User Manual Regen Targets"
	@echo "  regen.uman        regenerates page. of the user manual chapeter e.g. regen.uman chapter=..."
	@echo "  regen.uman-all    regenerates the user manual"
	@echo "  regen.uman-clean  stop temp server and clean up directory"
	@echo "  Chapter List: $(USER_MAN_CHAPTERS)"

clean:
	-rm -rf $(BUILDDIR)/*

version:
	@echo "version $(VERSION)"

doczip: html
	python doczip.py $(DOCZIP) _build/html

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml:
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."

pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp:
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/devpi.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/devpi.qhc"

devhelp:
	$(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(BUILDDIR)/devhelp
	@echo
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/devpi"
	@echo "# ln -s $(BUILDDIR)/devhelp $$HOME/.local/share/devhelp/devpi"
	@echo "# devhelp"

epub:
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(BUILDDIR)/epub
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

text:
	$(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(BUILDDIR)/text
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

man:
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(BUILDDIR)/man
	@echo
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."

texinfo:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(BUILDDIR)/texinfo
	@echo
	@echo "Build finished. The Texinfo files are in $(BUILDDIR)/texinfo."
	@echo "Run \`make' in that directory to run these through makeinfo" \
	      "(use \`make info' here to do that automatically)."

info:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(BUILDDIR)/texinfo
	@echo "Running Texinfo files through makeinfo..."
	make -C $(BUILDDIR)/texinfo info
	@echo "makeinfo finished; the Info files are in $(BUILDDIR)/texinfo."

gettext:
	$(SPHINXBUILD) -b gettext $(I18NSPHINXOPTS) $(BUILDDIR)/locale
	@echo
	@echo "Build finished. The message catalogs are in $(BUILDDIR)/locale."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."


regen: example-1.0.tar.gz quickstart-pypimirror quickstart-releaseprocess quickstart-server regen.uman-all
	rm -rf TARGETDIR example-1.0.tar.gz

example-1.0.tar.gz:
	cd _src; tar czf ../example-1.0.tar.gz example-1.0

quickstart-pypimirror:
	USER=testuser sh regen.sh --update quickstart-pypimirror.rst

quickstart-releaseprocess: example-1.0.tar.gz
	USER=testuser sh regen.sh --update quickstart-releaseprocess.rst

quickstart-server:
	USER=testuser sh regen.sh --update quickstart-server.rst


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# USER MANUAL TARGETS
 

$(USER_MAN_CHAPTERS):
	@echo Regenerating $@
	USER=testuser sh regen.sh --update userman/$@.rst

regen.sample_undo:
	-git checkout userman/pysober

regen.server-fresh: regen.server-clean
	rm -rf $(TESTHOME)
	devpi-init --serverdir=$(TESTHOME)/.devpi/server
	devpi-gen-config --serverdir=$(TESTHOME)/.devpi/server --argon2-memory-cost 8 --argon2-parallelism 1 --argon2-time-cost 1
	supervisord -c gen-config/supervisord.conf
	waitforports -t 60 3141

regen.server-stop:
	@echo stopping supervisord
	-supervisorctl -c gen-config/supervisord.conf shutdown

regen.server-clean: regen.sample_undo
	@echo stopping supervisord
	-supervisorctl -c gen-config/supervisord.conf shutdown
	@echo deleting $(TESTHOME) state
	-rm -rf $(TESTHOME)
	@echo deleting pysober/dist
	-rm -rf userman/pysober/dist
	git checkout userman/pysober

regen.uman-all : regen.server-fresh $(USER_MAN_CHAPTERS)
	git checkout userman/pysober
	rm -rf userman/pysober/dist
	@echo stopping supervisord
	-supervisorctl -c gen-config/supervisord.conf shutdown

regen.uman: $(chapter)


