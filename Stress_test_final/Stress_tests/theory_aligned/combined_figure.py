# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Compose the certificate and epsilon-sweep figures into one stacked plate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Stress_tests.theory_aligned.visualization import plot_epsilon_radius_sweep


CERTIFICATE_STEMS = (
    "duffing_certificate_figure",
    "nlink_projected_radius_figure",
)


def compose_stacked_figures(
    certificate_base: str | Path,
    sweep_base: str | Path,
    output_base: str | Path,
) -> dict[str, str]:
    """Stack an equal-height-scale sweep below a width-preserved certificate."""
    certificate_base = Path(certificate_base)
    sweep_base = Path(sweep_base)
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(certificate_base.with_suffix(".png")) as certificate_source:
        certificate = certificate_source.convert("RGB")
        certificate_dpi = certificate_source.info.get("dpi", (400.0, 400.0))
    with Image.open(sweep_base.with_suffix(".png")) as sweep_source:
        sweep = sweep_source.convert("RGB")
    if sweep.width != certificate.width:
        sweep = sweep.resize((certificate.width, sweep.height), Image.Resampling.LANCZOS)
    combined = Image.new(
        "RGB", (certificate.width, certificate.height + sweep.height), "white"
    )
    combined.paste(certificate, (0, 0))
    combined.paste(sweep, (0, certificate.height))
    png_path = output_base.with_suffix(".png")
    combined.save(png_path, dpi=certificate_dpi)

    try:
        from pypdf import PageObject, PdfReader, PdfWriter, Transformation
    except ImportError as exc:
        raise RuntimeError(
            "Vector PDF composition requires pypdf; install it with "
            "`python -m pip install pypdf`."
        ) from exc

    certificate_page = PdfReader(certificate_base.with_suffix(".pdf")).pages[0]
    sweep_page = PdfReader(sweep_base.with_suffix(".pdf")).pages[0]
    certificate_width = float(certificate_page.mediabox.width)
    certificate_height = float(certificate_page.mediabox.height)
    sweep_width = float(sweep_page.mediabox.width)
    sweep_height = float(sweep_page.mediabox.height)
    page = PageObject.create_blank_page(
        width=certificate_width, height=certificate_height + sweep_height
    )
    page.merge_transformed_page(sweep_page, Transformation().scale(
        sx=certificate_width / sweep_width, sy=1.0
    ))
    page.merge_transformed_page(
        certificate_page, Transformation().translate(ty=sweep_height)
    )
    pdf_path = output_base.with_suffix(".pdf")
    writer = PdfWriter()
    writer.add_page(page)
    with pdf_path.open("wb") as stream:
        writer.write(stream)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def render_combined_theory_aligned_figure(run_dir: str | Path) -> dict[str, str]:
    """Rerender the sweep at certificate width and compose one final figure.

    If the certificate-boundary panel was eliminated (too few successful
    rays, see nlink_certificate_figure.py::MIN_CERTIFICATE_BOUNDARY_RAYS),
    there is nothing to stack it with; the sweep alone (panel label "a", the
    only remaining panel) becomes the combined figure.
    """
    theory_dir = Path(run_dir) / "stress_tests" / "theory_aligned"
    certificate_stems = [
        stem for stem in CERTIFICATE_STEMS
        if (theory_dir / f"{stem}.png").is_file()
        and (theory_dir / f"{stem}.pdf").is_file()
    ]
    if len(certificate_stems) > 1:
        raise FileNotFoundError(
            f"Expected at most one certificate figure in {theory_dir}; "
            f"found {certificate_stems}"
        )
    sweep_base = theory_dir / "eps_sweep_input"
    sweep_has_certificate_panel = len(certificate_stems) == 1
    with np.load(sweep_base.with_suffix(".npz")) as data:
        with sweep_base.with_suffix(".json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        plot_epsilon_radius_sweep(
            data["epsilon_minus_min_h"],
            data["regular_radius"],
            data["sampled_radius"],
            sweep_base,
            nominal_epsilon_minus_min_h=metadata["nominal_epsilon_minus_min_h"],
            nominal_label=rf"nominal $\epsilon={float(metadata['nominal_epsilon']):.1f}$",
            comparison_epsilon_minus_min_h=metadata.get("comparison_epsilon_minus_min_h"),
            comparison_label=(
                rf"comparison-fig. $\epsilon={float(metadata['comparison_epsilon']):.1f}$"
                if metadata.get("comparison_epsilon") is not None else r"comparison-figure $\epsilon$"
            ),
            figure_size=(12.0, 3.15),
            panel_label="c" if sweep_has_certificate_panel else "a",
        )
    if not sweep_has_certificate_panel:
        combined_base = theory_dir / "combined_theory_aligned_figure"
        outputs = {}
        for suffix in (".png", ".pdf"):
            source = sweep_base.with_suffix(suffix)
            target = combined_base.with_suffix(suffix)
            target.write_bytes(source.read_bytes())
            outputs[suffix.lstrip(".")] = str(target)
        return outputs
    return compose_stacked_figures(
        theory_dir / certificate_stems[0],
        sweep_base,
        theory_dir / "combined_theory_aligned_figure",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <run_dir>")
    print(json.dumps(render_combined_theory_aligned_figure(sys.argv[1]), indent=2))