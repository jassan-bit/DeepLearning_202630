"""Sphinx hook: recreate the frozen documentary ZIP, never the 4 GB master."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from sphinx.util import logging
LOG=logging.getLogger(__name__)

def on_builder_inited(app):
 repo=Path(app.confdir).resolve().parent
 source=repo/'experiments/e3_completion_20261120'
 mirror=repo/'book_e3/_static/e3_completion_20261120'
 helper=source/'assemble_documents.py'
 if not helper.is_file():raise RuntimeError('Missing frozen E3 archive assembler: '+str(helper))
 spec=importlib.util.spec_from_file_location('_e3_completion_document_assembler',helper)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 descriptor=module.assemble(source,mirror,write_descriptor=False)
 LOG.info('E3 documentary archive verified: %s (%s bytes). Scope: %s',descriptor['archive'],descriptor['bytes'],descriptor['scope'])

def setup(app):
 app.connect('builder-inited',on_builder_inited)
 return {'version':'1.0','parallel_read_safe':True,'parallel_write_safe':True}
