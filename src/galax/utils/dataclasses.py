"""galax: Galactic Dynamix in Jax."""

__all__ = ["ModuleMeta"]

import dataclasses
import functools
import inspect
from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Generic, TypeVar, cast, overload

from equinox._module import _has_dataclass_init, _ModuleMeta

from dataclassish import DataclassInstance
from dataclassish.converters import AbstractConverter

##############################################################################
# Converter


class Sentinel(Enum):
    MISSING = auto()


ArgT = TypeVar("ArgT")  # Input type
RetT = TypeVar("RetT")  # Return type
SenT = TypeVar("SenT", bound=Enum)  # Sentinel type


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class sentineled(AbstractConverter[ArgT, RetT], Generic[ArgT, RetT, SenT]):
    """Optional converter with a defined sentinel value.

    This converter allows for a field to be optional, i.e., it can be set to
    some sentinel value.  This is useful when a field is required in some
    contexts but not in others.

    See Also
    --------
    :class:`optional`

    Examples
    --------
    >>> from typing import Literal
    >>> import equinox as eqx
    >>> from galax.utils.dataclasses import sentineled, Sentinel

    >>> class Class(eqx.Module):
    ...     a: int | Literal[Sentinel.MISSING] = eqx.field(
    ...         default=Sentinel.MISSING, converter=sentineled(int, Sentinel.MISSING))

    >>> obj = Class()
    >>> obj.a
    <Sentinel.MISSING: 1>

    >>> obj = Class(a=1)
    >>> obj.a
    1

    """

    converter: Callable[[ArgT], RetT]
    sentinel: SenT

    @overload
    def __call__(self, value: SenT, /) -> SenT: ...

    @overload
    def __call__(self, value: ArgT, /) -> RetT: ...

    def __call__(self, value: ArgT | SenT, /) -> RetT | SenT:
        if value is self.sentinel:
            return cast(SenT, value)
        return self.converter(cast(ArgT, value))


##############################################################################
# ModuleMeta

T = TypeVar("T")


def _add_converter_init_to_class(cls: type[T], /) -> type[T]:
    """Make a new `__init__` method that applies the converters."""
    original_init = cls.__init__
    sig = inspect.signature(original_init)

    @functools.wraps(original_init)
    def init(self: DataclassInstance, *args: Any, **kwargs: Any) -> None:
        __tracebackhide__ = True  # pylint: disable=unused-variable

        # Apply any converter to its argument.
        ba = sig.bind(self, *args, **kwargs)
        for f in dataclasses.fields(self):
            if f.name in ba.arguments and "converter" in f.metadata:
                ba.arguments[f.name] = f.metadata["converter"](ba.arguments[f.name])
        # Call the original `__init__`.
        init.__wrapped__(*ba.args, **ba.kwargs)

    cls.__init__ = init  # type: ignore[assignment, method-assign]

    return cls


# TODO: upstream this to Equinox
# TODO: Equinox doesn't seem to respect the conversion of the default value anymore.
class ModuleMeta(_ModuleMeta):  # type: ignore[misc]
    """Equinox-compatible module metaclass.

    This metaclass extends Equinox's :class:`equinox._module._ModuleMeta` to
    support the following features:

    - Application of ``converter`` to default values on fields.
    - Application of ``converter`` to values passed to ``__init__``.

    Examples
    --------
    >>> import equinox as eqx
    >>> class Class(eqx.Module, metaclass=ModuleMeta):
    ...     a: int = eqx.field(default=1.0, converter=int)
    ...     def __post_init__(self): pass

    >>> Class.a
    1

    >>> Class(a=2.0)
    Class(a=2)
    """

    def __new__(  # noqa: D102  # pylint: disable=signature-differs
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        *,
        strict: bool = False,
        **kwargs: Any,
    ) -> type:
        # [Step 1] Create the class using `_ModuleMeta`.
        cls: type = super().__new__(
            mcs, name, bases, namespace, strict=strict, **kwargs
        )

        # [Step 2] Convert the defaults.
        for k, v in namespace.items():
            if not isinstance(v, dataclasses.Field):
                continue
            # Apply the converter to the default value.
            if "converter" in v.metadata and not isinstance(
                v.default,
                dataclasses._MISSING_TYPE,  # noqa: SLF001
            ):
                setattr(cls, k, v.metadata["converter"](v.default))

        # [Step 3] Ensure conversion happens before `__init__`.
        if _has_dataclass_init[cls]:
            cls = _add_converter_init_to_class(cls)

        return cls
