#!/bin/bash
set -xe -o nounset
yes yes | devpi index --delete "${DEVPI_INDEX}" || true
devpi index -c "${DEVPI_INDEX}" bases=root/pypi mirror_whitelist=Cython
devpi use "${DEVPI_INDEX}"
devpi push --index root/pypi devpi-server==4.0.0 "${DEVPI_INDEX}"
devpi push --index root/pypi devpi-server==5.2.0 "${DEVPI_INDEX}"
devpi push --index fschulze/dev Cython==3.0.0a11 "${DEVPI_INDEX}"
pushd common
yes | towncrier
devpi upload
popd
pushd server
yes | towncrier
devpi upload
popd
pushd web
yes | towncrier
devpi upload
popd
pushd client
yes | towncrier
devpi upload
popd
pushd postgresql
yes | towncrier
devpi upload
popd
