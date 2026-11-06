# Cylinder wake reference

The cylinder validation compares its measured `St=fD/U` with the laminar shedding relation
quoted in the [PHOENICS cylinder tutorial](https://www.cham.co.uk/phoenics/d_polis/d_applic/urb/urb.htm):

`St(Re) = -3.3265/Re + 0.1816 + 0.00016 Re`.

That tutorial attributes the relation to Thompson, Hourigan & Sheridan (1994), and notes that
confinement increases Strouhal number. The reference at Re=100 is `0.164335`. Mackerel's
2x1 channel has blockage `D/H=0.16`, a staircase rather than body-fitted boundary, and a
finite downstream domain, so the measured value is reported alongside—not replaced by—the
reference relation. See also Williamson (1996), *Vortex dynamics in the cylinder wake*,
Annual Review of Fluid Mechanics 28, 477–539, for the physical wake regime.
