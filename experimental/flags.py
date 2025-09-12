# flaglib.py - A simple, decentralized flag library.

import argparse
import sys
from typing import Any, Dict, Generic, Optional, Type, TypeVar

# --- Global State ---
# This dictionary will store the definitions of all flags registered by any module.
# The key is the flag name (e.g., 'user'), and the value is the Flag object.
_flag_registry: Dict[str, "Flag[Any]"] = {}

# This variable will track whether parsing has already occurred.
_parsed = False

T = TypeVar("T")


class Flag(Generic[T]):
    """A container for a defined flag, its value, and its metadata."""

    def __init__(
        self,
        name: str,
        type: Type[T],
        help: str,
        required: bool,
        default: Optional[T],
    ):
        if name in _flag_registry:
            raise ValueError(f"Flag '{name}' has already been defined.")

        self.name = name
        self.type = type
        self.help = help
        self.required = required
        self.default = default
        self._value: Optional[T] = None  # Will be populated after parse_flags() is called.

        # Register this instance in the global registry.
        _flag_registry[name] = self

    def get(self) -> T:
        """
        Retrieves the value of the flag.

        Raises:
            RuntimeError: If called before flaglib.parse_flags() has been run.
        """
        if not _parsed:
            raise RuntimeError(
                "flaglib.get() was called before flaglib.parse_flags() ran. "
                "Please call parse_flags() in your main execution block."
            )
        return self._value  # type: ignore

    def __call__(self) -> T:
        return self.get()

    def __repr__(self) -> str:
        return (
            f"<Flag name='{self.name}' type={self.type.__name__} value={self._value}>"
        )


def flag(
    name: str,
    type: Type[T] = str,  # type: ignore
    required: bool = False,
    help: str = "",
    default: Optional[T] = None,
) -> Flag[T]:
    """
    Defines and registers a command-line flag.

    This should be called at the top level of a module.

    Args:
        name: The name of the flag (e.g., 'user' becomes '--user').
        type: The expected type of the flag's value (e.g., int, str, bool).
        required: Whether the flag must be provided on the command line.
        help: A help string describing what the flag does.
        default: The default value if the flag is not provided.

    Returns:
        A Flag object that can be used to retrieve the value after parsing.
    """
    return Flag(name=name, type=type, help=help, required=required, default=default)


def parse_flags():
    """

    Parses all defined flags from sys.argv.

    This function should be called ONCE at the start of the main execution
    block (if __name__ == '__main__'). It will exit the program if required
    flags are missing or if '--help' is used.
    """
    global _parsed
    if _parsed:
        # Avoid parsing more than once.
        return

    parser = argparse.ArgumentParser(description="A script using flaglib.")

    # Add all registered flags to the argparse parser.
    for flag_name, flag_instance in _flag_registry.items():
        # Special handling for boolean flags to create store_true behavior
        if flag_instance.type is bool:
            parser.add_argument(
                f"--{flag_name}", action="store_true", help=flag_instance.help
            )
        else:
            parser.add_argument(
                f"--{flag_name}",
                type=flag_instance.type,
                required=flag_instance.required,
                default=flag_instance.default,
                help=flag_instance.help,
            )

    # Use argparse to parse the command-line arguments.
    # We pass `sys.argv[1:]` to exclude the script name itself.
    parsed_args = parser.parse_args(sys.argv[1:])

    # Populate the ._value attribute of each registered Flag object.
    for flag_name, flag_instance in _flag_registry.items():
        if hasattr(parsed_args, flag_name):
            flag_instance._value = getattr(parsed_args, flag_name)

    _parsed = True
