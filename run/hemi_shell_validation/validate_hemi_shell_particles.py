#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np


def load_particles(path):
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.replace(",", " ").split()
            row = []
            for value in values:
                try:
                    row.append(float(value))
                except ValueError:
                    pass
            if len(row) >= 7:
                data.append(row[:7])
    if not data:
        raise SystemExit(f"No particle rows with at least 7 numeric columns found in {path}")
    arr = np.asarray(data, dtype=float)
    return arr[:, 0:3], arr[:, 6]


def spatial_hash_min_distance(pos, radii, spacing):
    min_allowed = 2.0 * float(np.max(radii)) + spacing
    cell_size = max(min_allowed, np.finfo(float).eps)
    cells = {}
    min_dist = math.inf
    pair = None
    for idx, point in enumerate(pos):
        cell = tuple(np.floor(point / cell_size).astype(int))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), []):
                        dist = float(np.linalg.norm(point - pos[other]))
                        if dist < min_dist:
                            min_dist = dist
                            pair = (other, idx)
        cells.setdefault(cell, []).append(idx)
    return min_dist, min_allowed, pair


def validate(pos, radii, args):
    rel = pos - np.asarray(args.center, dtype=float)
    radial = np.linalg.norm(rel, axis=1)
    max_radius = float(np.max(radii))
    inner_clearance = radial - radii - args.inner_radius
    outer_clearance = args.outer_radius - radial - radii
    plane_axis = {"x": 0, "y": 1, "z": 2}[args.axis]
    plane_clearance = rel[:, plane_axis] - radii
    min_dist, min_allowed, pair = spatial_hash_min_distance(pos, radii, args.min_spacing)

    failures = {
        "inner_boundary": int(np.count_nonzero(inner_clearance < -args.tol)),
        "outer_boundary": int(np.count_nonzero(outer_clearance < -args.tol)),
        "hemisphere_plane": int(np.count_nonzero(plane_clearance < -args.tol)),
        "overlap": int(0 if min_dist + args.tol >= min_allowed else 1),
    }
    summary = {
        "n_particles": int(len(pos)),
        "radius_min": float(np.min(radii)),
        "radius_max": max_radius,
        "radial_min": float(np.min(radial)),
        "radial_max": float(np.max(radial)),
        "inner_clearance_min": float(np.min(inner_clearance)),
        "outer_clearance_min": float(np.min(outer_clearance)),
        "plane_clearance_min": float(np.min(plane_clearance)),
        "min_pair_distance": min_dist,
        "min_allowed_distance": min_allowed,
        "min_pair": pair,
        "failures": failures,
    }
    return summary


def write_report(summary, output):
    lines = []
    for key, value in summary.items():
        lines.append(f"{key}: {value}")
    text = "\n".join(lines) + "\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
    print(text, end="")


def plot_sections(pos, radii, args, output):
    from PIL import Image, ImageDraw

    center = np.asarray(args.center, dtype=float)
    rel = pos - center
    axis = {"x": 0, "y": 1, "z": 2}[args.axis]
    other_axes = [i for i in range(3) if i != axis]
    levels = np.linspace(float(np.min(rel[:, axis])), float(np.max(rel[:, axis])), args.sections)
    thickness = max(2.5 * float(np.max(radii)), (levels[-1] - levels[0]) / max(args.sections, 1) * 0.35)

    panel = 520
    margin = 52
    image = Image.new("RGB", (panel * args.sections, panel), "white")
    draw = ImageDraw.Draw(image)
    limit = args.outer_radius * 1.08

    def project(value, offset):
        return offset + margin + (value + limit) / (2.0 * limit) * (panel - 2 * margin)

    for section_idx, level in enumerate(levels):
        xoff = section_idx * panel
        left = xoff + margin
        right = xoff + panel - margin
        top = margin
        bottom = panel - margin
        draw.rectangle((left, top, right, bottom), outline=(210, 210, 210))
        for frac in np.linspace(-1.0, 1.0, 5):
            gx = project(frac * limit, xoff)
            gy = project(frac * limit, 0)
            draw.line((gx, top, gx, bottom), fill=(235, 235, 235))
            draw.line((left, gy, right, gy), fill=(235, 235, 235))

        mask = np.abs(rel[:, axis] - level) <= thickness
        section_outer = max(args.outer_radius**2 - level**2, 0.0)
        section_inner = max(args.inner_radius**2 - level**2, 0.0)
        for boundary_radius, color in [(math.sqrt(section_outer), (0, 0, 0)), (math.sqrt(section_inner), (80, 80, 80))]:
            px_radius = boundary_radius / (2.0 * limit) * (panel - 2 * margin)
            cx = project(0.0, xoff)
            cy = project(0.0, 0)
            draw.ellipse((cx - px_radius, cy - px_radius, cx + px_radius, cy + px_radius), outline=color, width=2)

        for point, radius in zip(rel[mask], radii[mask]):
            px = project(point[other_axes[0]], xoff)
            py = project(point[other_axes[1]], 0)
            pr = max(1.4, radius / (2.0 * limit) * (panel - 2 * margin) * 10.0)
            draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=(214, 39, 40), outline=None)

        draw.text((xoff + 14, 14), f"{args.axis}={level:.3g}, n={int(np.count_nonzero(mask))}", fill=(0, 0, 0))
        draw.text((xoff + 14, panel - 30), f"axes: {'xyz'[other_axes[0]]}-{'xyz'[other_axes[1]]}", fill=(0, 0, 0))

    image.save(output)


def main():
    parser = argparse.ArgumentParser(description="Validate EL hemi-shell particle DAT files including particle radius.")
    parser.add_argument("dat_file")
    parser.add_argument("--center", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--inner-radius", type=float, required=True)
    parser.add_argument("--outer-radius", type=float, required=True)
    parser.add_argument("--min-spacing", type=float, default=0.0)
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--tol", type=float, default=1.0e-12)
    parser.add_argument("--report")
    parser.add_argument("--plot")
    parser.add_argument("--sections", type=int, default=5)
    args = parser.parse_args()

    pos, radii = load_particles(args.dat_file)
    summary = validate(pos, radii, args)
    write_report(summary, args.report)
    if args.plot:
        plot_sections(pos, radii, args, args.plot)
    failed = any(summary["failures"].values())
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
