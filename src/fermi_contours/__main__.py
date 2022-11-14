"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Fermi Contours."""


if __name__ == "__main__":
    main(prog_name="fermi-contours")  # pragma: no cover
