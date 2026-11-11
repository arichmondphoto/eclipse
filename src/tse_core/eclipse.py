"""Local circumstances of a solar eclipse from Besselian elements.

Gives, for any site and time, where the Moon's centre is relative to the
Sun's centre, plus the contact times, the duration of totality and the
Moon-to-Sun radius ratio. This is the "eclipse calculator" input the
FNRGF paper's appendix needs to find the Sun's centre behind the Moon.

The method is the standard one (Explanatory Supplement to the Astronomical
Almanac; NASA eclipse bulletins): the observer is projected onto the
fundamental plane, and (u, v) is the shadow axis relative to the observer,
in Earth radii, with u towards celestial east and v towards north.

The Moon's apparent offset from the Sun is proportional to (u, v). The
scale follows from the shadow cones: at the penumbral radius L1' the
centres are S + M apart, and at the umbral radius |L2'| they are |M - S|
apart, where S and M are the apparent radii. Hence

    offset / S = (u, v) * (1 + M/S) / L1'      with  M/S = (L1' - L2') / (L1' + L2').
"""

from __future__ import annotations

import functools
import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

DATA_FILE = Path(__file__).resolve().parent / "data" / "besselian.json"
# NASA's catalogue for CANON_YEARS, shipped with the toolkit so no eclipse in that range needs a download.
BUNDLED_CANON = Path(__file__).resolve().parent / "data" / "besselian_canon.json"
CACHE_DIR = Path(os.environ.get("TSE_CACHE_DIR", Path.home() / ".cache" / "solar-eclipse-toolkit"))
CANON_CACHE = CACHE_DIR / "besselian_canon.json"
# NASA's Five Millennium Canon of Solar Eclipses, Besselian elements for every eclipse -1999..3000.
CANON_URL = "https://eclipse.gsfc.nasa.gov/eclipse_besselian_from_mysqldump2.csv"
CANON_YEARS = (1600, 2400)  # range shipped with the toolkit, and kept from a download
EARTH_EQUATORIAL_M = 6378137.0
B_OVER_A = 0.99664719  # Earth's polar/equatorial axis ratio
SUN_SEMIDIAMETER_1AU = 959.63  # arcsec


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------


@dataclass
class BesselianElements:
    date: str  # YYYY-MM-DD of t0
    type: str
    t0_tdt_hours: float
    delta_t: float
    x: list[float]
    y: list[float]
    d: list[float]
    l1: list[float]
    l2: list[float]
    mu: list[float]
    tan_f1: float
    tan_f2: float
    source: str = ""
    delta_t_observed: Optional[float] = None
    t_valid: Optional[list[float]] = None  # hours from t0 over which the polynomials are valid

    @property
    def best_delta_t(self) -> float:
        return self.delta_t_observed if self.delta_t_observed is not None else self.delta_t

    @property
    def t0(self) -> datetime:
        """Reference instant in Terrestrial Dynamical Time (as a naive datetime)."""
        return datetime.fromisoformat(self.date) + timedelta(hours=self.t0_tdt_hours)


def load_elements(path: Optional[str | os.PathLike] = None) -> dict[str, BesselianElements]:
    data = json.loads(Path(path or DATA_FILE).read_text(encoding="utf-8"))
    out = {}
    for key, e in data.get("eclipses", {}).items():
        fields = {k: v for k, v in e.items() if k in BesselianElements.__dataclass_fields__}
        out[key] = BesselianElements(date=key, **{k: v for k, v in fields.items() if k != "date"})
    return out


@functools.lru_cache(maxsize=4)
def _load_cached(path: str, _mtime_ns: int) -> dict[str, BesselianElements]:
    """A catalogue file read once (about 1900 eclipses), again only when the file changes."""
    return load_elements(path)


def all_elements(path: Optional[str | os.PathLike] = None) -> dict[str, BesselianElements]:
    """Every eclipse we know: NASA's catalogue as shipped, then a copy downloaded since
    ('totality elements-update'), then the bundled bulletins, then a user file (later wins).

    The bulletin values come from NASA's individual eclipse bulletins and take
    precedence over the catalogue for the same eclipse.
    """
    table: dict[str, BesselianElements] = {}
    for canon in (BUNDLED_CANON, CANON_CACHE):
        if canon.is_file():
            try:
                table.update(_load_cached(str(canon), canon.stat().st_mtime_ns))
            except (ValueError, KeyError, TypeError, OSError):
                pass
    table.update(load_elements(DATA_FILE))
    if path:
        table.update(load_elements(path))
    return table


def elements_for(day: date | datetime | str, path: Optional[str | os.PathLike] = None) -> BesselianElements:
    """Elements for the eclipse on this date (UT date; the day before or after also matches)."""
    if isinstance(day, str):
        day = date.fromisoformat(day[:10])
    if isinstance(day, datetime):
        day = day.date()
    table = all_elements(path)
    for offset in (0, -1, 1):
        key = (day + timedelta(days=offset)).isoformat()
        if key in table:
            return table[key]
    if BUNDLED_CANON.is_file() or CANON_CACHE.is_file():
        hint = f" The catalogue covers solar eclipses from {CANON_YEARS[0]} to {CANON_YEARS[1]}."
    else:
        hint = " Run 'totality elements-update' to download NASA's catalogue of every eclipse."
    raise KeyError(f"no Besselian elements for {day}.{hint}")


# ---------------------------------------------------------------------------
# NASA canon: download and convert
# ---------------------------------------------------------------------------


def parse_canon_csv(text: str, years: tuple[int, int] = CANON_YEARS) -> dict[str, dict]:
    """Convert NASA's canon CSV (one row per eclipse) to this module's JSON layout."""
    import csv
    import io

    out: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            y, m, d = int(row["year"]), int(row["month"]), int(row["day"])
        except (KeyError, ValueError):
            continue
        if not years[0] <= y <= years[1]:
            continue

        def f(name: str) -> float:
            return float(row[name])

        out[f"{y:04d}-{m:02d}-{d:02d}"] = {
            "type": row.get("eclipse_type", "").strip()[:1],
            "t0_tdt_hours": f("t0"),
            "delta_t": f("dt"),
            "x": [f("x0"), f("x1"), f("x2"), f("x3")],
            "y": [f("y0"), f("y1"), f("y2"), f("y3")],
            "d": [f("d0"), f("d1"), f("d2")],
            "l1": [f("l10"), f("l11"), f("l12")],
            "l2": [f("l20"), f("l21"), f("l22")],
            "mu": [f("mu0"), f("mu1"), f("mu2")],
            "tan_f1": f("tan_f1"),
            "tan_f2": f("tan_f2"),
            "t_valid": [f("tmin"), f("tmax")],
            "source": "NASA Five Millennium Canon of Solar Eclipses (Espenak & Meeus)",
        }
    return out


def update_canon(source: Optional[str | os.PathLike] = None, timeout: float = 120.0) -> tuple[Path, int]:
    """Download NASA's canon (or read a local copy) and cache it. Returns (path, eclipses)."""
    if source:
        text = Path(source).read_text(encoding="utf-8", errors="replace")
    else:
        from .net import fetch

        text = fetch(CANON_URL, timeout=timeout).decode("utf-8", errors="replace")
    eclipses = parse_canon_csv(text)
    if not eclipses:
        raise ValueError("no eclipses found in the canon file")
    CANON_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CANON_CACHE.write_text(json.dumps({"_comment": f"Converted from {CANON_URL}", "eclipses": eclipses}),
                           encoding="utf-8")
    return CANON_CACHE, len(eclipses)


def _poly(c: list[float], t: float) -> float:
    return sum(a * t**i for i, a in enumerate(c))


def _dpoly(c: list[float], t: float) -> float:
    return sum(i * a * t ** (i - 1) for i, a in enumerate(c) if i)


# ---------------------------------------------------------------------------
# Site and state
# ---------------------------------------------------------------------------


@dataclass
class Site:
    latitude: float  # degrees, north positive
    longitude: float  # degrees, east positive
    elevation_m: float = 0.0
    name: str = ""

    def geocentric(self) -> tuple[float, float]:
        """(rho sin phi', rho cos phi') in Earth equatorial radii."""
        phi = math.radians(self.latitude)
        u = math.atan(B_OVER_A * math.tan(phi))
        h = self.elevation_m / EARTH_EQUATORIAL_M
        return B_OVER_A * math.sin(u) + h * math.sin(phi), math.cos(u) + h * math.cos(phi)


@dataclass
class State:
    t: float  # hours after t0, TDT
    u: float
    v: float
    a: float  # du/dt, Earth radii per hour
    b: float  # dv/dt
    L1: float
    L2: float
    zeta: float

    @property
    def delta(self) -> float:
        return math.hypot(self.u, self.v)

    @property
    def ratio(self) -> float:
        """Apparent Moon radius / Sun radius."""
        return (self.L1 - self.L2) / (self.L1 + self.L2)

    @property
    def scale(self) -> float:
        """Sun radii of apparent separation per Earth radius on the fundamental plane."""
        return (1.0 + self.ratio) / self.L1

    def offset(self) -> tuple[float, float]:
        """Moon centre minus Sun centre, in Sun radii, (east, north)."""
        return self.u * self.scale, self.v * self.scale

    def velocity(self) -> tuple[float, float]:
        """Rate of change of the offset, Sun radii per hour, (east, north)."""
        return self.a * self.scale, self.b * self.scale

    def position_angle(self) -> float:
        """Position angle of the Moon's centre from the Sun's, degrees from north through east."""
        return math.degrees(math.atan2(self.u, self.v)) % 360.0


class LocalEclipse:
    """Circumstances of one eclipse at one site."""

    def __init__(self, elements: BesselianElements, site: Site, delta_t: Optional[float] = None):
        self.el = elements
        self.site = site
        self.delta_t = elements.best_delta_t if delta_t is None else float(delta_t)
        self._rho_s, self._rho_c = site.geocentric()

    # ------------------------------------------------------------ timescale
    def ut_from_t(self, t: float) -> datetime:
        return (self.el.t0 + timedelta(hours=t) - timedelta(seconds=self.delta_t)).replace(tzinfo=timezone.utc)

    def t_from_ut(self, when: datetime) -> float:
        if when.tzinfo is not None:
            when = when.astimezone(timezone.utc).replace(tzinfo=None)
        return ((when + timedelta(seconds=self.delta_t)) - self.el.t0).total_seconds() / 3600.0

    # ---------------------------------------------------------------- state
    def state(self, t: float) -> State:
        e = self.el
        x, y = _poly(e.x, t), _poly(e.y, t)
        dx, dy = _dpoly(e.x, t), _dpoly(e.y, t)
        d = math.radians(_poly(e.d, t))
        dd = math.radians(_dpoly(e.d, t))
        mu = _poly(e.mu, t)
        dmu = math.radians(_dpoly(e.mu, t))
        H = math.radians(mu + self.site.longitude - 0.00417807 * self.delta_t)
        rs, rc = self._rho_s, self._rho_c
        xi = rc * math.sin(H)
        eta = rs * math.cos(d) - rc * math.cos(H) * math.sin(d)
        zeta = rs * math.sin(d) + rc * math.cos(H) * math.cos(d)
        dxi = dmu * rc * math.cos(H)
        deta = dmu * xi * math.sin(d) - zeta * dd
        L1 = _poly(e.l1, t) - zeta * e.tan_f1
        L2 = _poly(e.l2, t) - zeta * e.tan_f2
        return State(t, x - xi, y - eta, dx - dxi, dy - deta, L1, L2, zeta)

    def state_at(self, when: datetime) -> State:
        return self.state(self.t_from_ut(when))

    # ------------------------------------------------------- circumstances
    def maximum(self) -> State:
        t = 0.0
        for _ in range(50):
            s = self.state(t)
            step = -(s.u * s.a + s.v * s.b) / (s.a * s.a + s.b * s.b)
            t += step
            if abs(step) < 1e-9:
                break
        return self.state(t)

    def _contact(self, t_lo: float, t_hi: float, umbral: bool) -> Optional[float]:
        def g(t: float) -> float:
            s = self.state(t)
            return s.delta - (abs(s.L2) if umbral else s.L1)

        a, b = g(t_lo), g(t_hi)
        if a * b > 0:
            return None
        for _ in range(80):
            m = 0.5 * (t_lo + t_hi)
            gm = g(m)
            if a * gm <= 0:
                t_hi, b = m, gm
            else:
                t_lo, a = m, gm
        return 0.5 * (t_lo + t_hi)

    def circumstances(self) -> "Circumstances":
        mx = self.maximum()
        tm = mx.t
        c1 = self._contact(tm - 4.0, tm, umbral=False)
        c4 = self._contact(tm, tm + 4.0, umbral=False)
        c2 = c3 = None
        kind = "partial"
        if mx.delta < abs(mx.L2):
            c2 = self._contact(tm - 0.25, tm, umbral=True)
            c3 = self._contact(tm, tm + 0.25, umbral=True)
            kind = "total" if mx.L2 < 0 else "annular"
        if mx.delta > mx.L1:
            kind = "none"
        ut = self.ut_from_t
        depth = None
        if kind in ("total", "annular"):
            depth = 100.0 * (1.0 - mx.delta / abs(mx.L2))
        return Circumstances(
            kind=kind,
            c1=ut(c1) if c1 is not None else None,
            c2=ut(c2) if c2 is not None else None,
            maximum=ut(tm),
            c3=ut(c3) if c3 is not None else None,
            c4=ut(c4) if c4 is not None else None,
            duration_s=(c3 - c2) * 3600.0 if c2 is not None and c3 is not None else None,
            magnitude=(mx.L1 - mx.delta) / (mx.L1 + mx.L2),
            ratio=mx.ratio,
            umbral_depth_pct=depth,
            sun_altitude_deg=math.degrees(math.asin(max(-1.0, min(1.0, mx.zeta)))),
            miss_sun_radii=mx.delta * mx.scale,
            position_angle_at_max=mx.position_angle() if mx.delta > 1e-9 else None,
            sun_radius_arcsec=sun_semidiameter_arcsec(ut(tm)),
            delta_t=self.delta_t,
        )


@dataclass
class Circumstances:
    kind: str
    c1: Optional[datetime]
    c2: Optional[datetime]
    maximum: datetime
    c3: Optional[datetime]
    c4: Optional[datetime]
    duration_s: Optional[float]
    magnitude: float
    ratio: float  # Moon / Sun apparent radius at maximum
    umbral_depth_pct: Optional[float]
    sun_altitude_deg: float
    miss_sun_radii: float  # Moon-Sun centre distance at maximum, in Sun radii
    position_angle_at_max: Optional[float]
    sun_radius_arcsec: float
    delta_t: float
    notes: list[str] = field(default_factory=list)

    @property
    def moon_radius_arcsec(self) -> float:
        return self.sun_radius_arcsec * self.ratio

    def describe(self, local_offset_h: Optional[float] = None) -> str:
        def fmt(t: Optional[datetime]) -> str:
            if t is None:
                return "-"
            text = t.strftime("%H:%M:%S") + " UT"
            if local_offset_h is not None:
                local = t + timedelta(hours=local_offset_h)
                text += f"  ({local:%H:%M:%S} local)"
            return text

        lines = [f"Eclipse type here: {self.kind}"]
        for name, t in (("C1 partial begins", self.c1), ("C2 totality begins", self.c2),
                        ("Maximum", self.maximum), ("C3 totality ends", self.c3), ("C4 partial ends", self.c4)):
            if t is not None:
                lines.append(f"  {name:<20} {fmt(t)}")
        if self.duration_s is not None:
            m, s = divmod(self.duration_s, 60)
            lines.append(f"  Duration of totality {int(m)}m{s:04.1f}s")
        lines.append(f"  Magnitude {self.magnitude:.4f}, Moon/Sun ratio {self.ratio:.4f}, Sun altitude {self.sun_altitude_deg:.0f} deg")
        lines.append(f"  Sun radius {self.sun_radius_arcsec:.1f}\", Moon radius {self.moon_radius_arcsec:.1f}\"")
        if self.umbral_depth_pct is not None:
            lines.append(f"  Umbral depth {self.umbral_depth_pct:.0f}% (100% = central line)")
        lines.append(f"  Closest approach of centres {self.miss_sun_radii * self.sun_radius_arcsec:.1f}\"")
        lines.append(f"  Delta T {self.delta_t:.1f} s")
        return "\n".join(lines)


def sun_semidiameter_arcsec(when: datetime) -> float:
    """Apparent solar radius from the Earth-Sun distance (accurate to ~0.1")."""
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    n = (when - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0
    g = math.radians((357.529 + 0.98560028 * n) % 360.0)
    r_au = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)
    return SUN_SEMIDIAMETER_1AU / r_au
