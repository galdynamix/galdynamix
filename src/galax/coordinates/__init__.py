"""Copyright (c) 2023 galax maintainers. All rights reserved."""

from jaxtyping import install_import_hook
from lazy_loader import attach_stub

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.coordinates", RUNTIME_TYPECHECKER):
    __getattr__, __dir__, __all__ = attach_stub(__name__, __file__)


# Clean up the namespace
del install_import_hook, attach_stub, RUNTIME_TYPECHECKER