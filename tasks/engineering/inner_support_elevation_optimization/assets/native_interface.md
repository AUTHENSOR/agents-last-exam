# Supported native model interface

Setup and evaluation run natively on Linux. The evaluator independently runs
all five support heights using the submitted model inputs and compares its
results with the submitted native fields and monitoring tables.

The current adapter supports Kratos10.4.3 UPw 4-node tetrahedral or 8-node
hexahedral soil, native Mohr-Coulomb layers, elastic C35 Q4 shells and native
linear beams with rigid interpolation connections. Its implemented hydraulic
choice is prescribed uniform hydrostatic drawdown. This is now an explicit
modeling restriction of the deterministic adaptation in model_definition.md.

The installed runtime is `/opt/ale-kratos/bin/python`. Preparation can create
a Python3.10 virtual environment and install the legitimate packages:

```
python3.10 -m venv /opt/ale-kratos
/opt/ale-kratos/bin/pip install KratosMultiphysics==10.4.3 KratosGeoMechanicsApplication==10.4.3 KratosStructuralMechanicsApplication==10.4.3 KratosLinearSolversApplication==10.4.3 numpy scipy
```

Setup verifies original source hashes and the installed runtime; it does not
launch a VM or modify the original sources. It stages the native construction
and result-reading tools under the task software directory. No private witness
mesh, digitization, monitor coordinates, material choices or answers are staged.

Place these native construction inputs directly under
`model_family_support_position/`:

- `model.mdpa`: native soil nodes, positive element connectivity, properties
  and fourteen `Layer01` through `Layer14` submodelparts.
- `Materials.json`: native `Support.LayerNN` material bindings using
  `GeoMohrCoulombLaw3D` and the original material values.
- `family.json`: `support-native-family-1` construction data in m-kN-s and the
  source absolute upward-z datum. Fields are `format`, `units`,
  `ground_elevation`, `domain` (xmin,xmax,ymin,ymax,zmin,zmax), `layers`
  (original id/top/bottom), `outer_polygon`, `inner_polygon`, wall-face
  connectivity in `outer_wall_faces` and `inner_wall_faces`, `braces`,
  `monitors`, `source_hashes`, and `engineering_choices`.

Brace definitions contain `system`, `segments_xy_m` and inner
`crown_segments_xy_m`. `engineering_choices.beam_section_m` declares width and
height for this adapter's common rectangular non-crown section; optional
`crowns` declares each system's axis elevation, shell attachment elevation and
section. Separate non-crown member sections are outside this adaptation.
At a declared crown elevation, members on the submitted wall perimeter use
the crown section; cross-pit members keep the common support section. Lower
perimeter support members also keep the common support section. Collinear
subdivision of the perimeter or beam graph does not change this assignment.
The native audit checks the actual installed properties against these choices.
States computed with different effective member properties cannot be reused
by changing checkpoint metadata; resume before the affected installation.
Physical `monitors` maps all DBC/CX names to your registered native xyz points.
Record your registration and engineering reasoning in `engineering_choices`.
The exact accepted fields and material API are in `native_input_adapter.py`
and `native_family_runner.py`; these inputs drive actual native construction,
not a separate claim of correctness. Choose your own justified mesh, geometry,
rock stiffness and undimensioned sections consistently across all five cases.
There is no private family hash requirement. A deeper-domain continuation of
the last listed rock can be explicitly declared; it is not recovered geology
and no60m depth is universally required.

First run the inexpensive deterministic preflight (no FE solve):

```
/opt/ale-kratos/bin/python SOFTWARE/scripts/native_model_contract.py FAMILY
```

This checks geometry and declared model admissibility only; it does not prove
native replay time or memory compliance.

Then run the staged native model for each position0..4, for example:

```
OMP_NUM_THREADS=1 /opt/ale-kratos/bin/python SOFTWARE/scripts/native_family_runner.py FAMILY FAMILY/results/pos_0m --position 0 --until full --linear-backend amgcl-gmres --max-iterations 400 --write-checkpoints
```

Here `SOFTWARE` and `FAMILY` denote the task software and delivered family
directories. Iteration budgets affect work allowance, not convergence criteria.
Retain each successful native `FileSerializer` checkpoint for continuation
during your solving window. Only the completed common Stage7 state, before any
inner member installation, may branch the five heights. Final displacements
alone are not a valid constitutive restart. Solver continuation does not extend
or reset the evaluator's independent replay deadline.

The task VM is c4-standard-8 (8 vCPU, 30 GiB RAM); the original solver window
remains 18000 seconds (5 hours). Independent native replay computes the common
Stage7 prefix once and then runs all five case suffixes in parallel. Its exact
6600-second (110-minute) wall-clock budget starts at native replay start and
includes the prefix, all five cases, native state audits and extraction. Each
single-threaded replay worker, including the common-prefix worker, receives one
CPU and a 4 GiB memory cap. Five suffix workers and a 4 GiB coordinator allowance
share a 24 GiB aggregate cap without swap, leaving VM headroom.

We recommend testing the complete family with the supplied native replay tools
within these limits with adequate time and memory headroom. A single-case run
or passing preflight does not establish full-family compliance. There is no
element-count cap and no additional mandatory mesh study. You may choose or
revise the mesh within the existing local-refinement and accuracy requirements.
The resource limit does not relax those checks or the physics, material, stage
or solver tolerance requirements.

The following optional full-family self-check runs on the actual c4-standard-8
task VM (8 vCPU, 30 GiB RAM) as `user`, using the same replay budget as evaluation.
It is not an extra assignment. These absolute paths use the default task root;
adjust them to the actual absolute paths if your task is staged elsewhere.
Choose a fresh, nonexistent `REPLAY` path for every check, changing the `-01`
suffix below, and regenerate `REVIEW` for the current family inputs.

```sh
SOFTWARE=/media/user/data/agenthle/engineering/inner_support_elevation_optimization/base/software
FAMILY=/media/user/data/agenthle/engineering/inner_support_elevation_optimization/base/output/model_family_support_position
REVIEW=/media/user/data/agenthle/engineering/inner_support_elevation_optimization/base/output/native-review-self-check-01.json
REPLAY=/media/user/data/agenthle/engineering/inner_support_elevation_optimization/base/output/native-replay-self-check-01

test ! -e "$REPLAY" &&
/opt/ale-kratos/bin/python "$SOFTWARE/scripts/native_model_contract.py" "$FAMILY" > "$REVIEW" &&
sudo -n systemd-run --wait --pipe --uid=user \
  -p Delegate=yes -p AllowedCPUs='0 1 2 3 4' -p CPUQuota=500% \
  -p MemoryMax=24G -p MemorySwapMax=0 -p OOMPolicy=continue \
  -p RuntimeMaxSec=6630 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/ale-kratos/bin/python "$SOFTWARE/scripts/native_replay_family.py" \
  "$FAMILY" "$REPLAY" "$REVIEW" --total-seconds 6600 --cpus 0 1 2 3 4
```

The supplied replay tool applies the single-CPU, 4 GiB worker caps and the
4 GiB coordinator allowance inside the delegated 24 GiB group. It computes
the common Stage7 prefix once, then the five suffixes in parallel, including
native state audits and extraction in the shared 6600-second budget.
`RuntimeMaxSec=6630` is the outer service guard with cleanup time; it does not
extend the native replay budget. Inspect `REPLAY/family-replay.json` and the
native logs for completion and resource evidence. This example is not a claim
of a newly validated VM run.

The runner saves native editable/results files, a precision nodal-field export,
calculation logs and the Stage1 baseline. Retain these generated products in
`results/pos_Nm/` (or `pos_Nm/`). They are the supported native project outputs,
not a requirement to invent extra archive formats. The evaluator independently
replays your frozen inputs, reads your saved `native_model.mdpa` and
`native_nodal_fields.tsv`, and compares physical monitors with its own results.
It does not accept your summary CSV as numerical authority.

Keep the originally required raw monitor table as `monitoring.csv` or
`monitoring.tsv` in each case directory, with columns
`monitor,x_m,y_m,z_m,ux_mm,uy_mm,uz_mm`. The native reader's equivalent
`independent_monitors.json` is also supported. Duplicate shell/crown endpoint
rows are permitted. Different profile row counts are accepted, with full
wall-top-to-toe coverage and sampling adequate to retain the physical maximum.
These are supported serialization choices for the original raw extraction,
not an additional engineering deliverable or prescribed monitor coordinates.

Evaluation first checks the published deterministic model contract, then
independently executes the common Stage7 prefix once and five native suffixes
in parallel, audits actual source facts, and reads physical DBC/CX results.
It applies the original50/50 rubric and
max(0.30mm,5%) response tolerance; percentages retain max(0.30point,8%).
No language-model service or evaluator credential is needed. Preflight violations
receive zero with the failed public check. Exceeding the published replay time
budget or a confirmed worker memory cap receives zero with a resource-exhaustion
receipt. Unexpected infrastructure or code faults and other incomplete replay
remain evaluator-unavailable, not zero. Failure to provide the published
allocation is evaluator-unavailable; reduced-resource or shortened diagnostic
replays cannot justify zero for resource exhaustion. Horizontal native hex faces
use bilinear Q4 sampling.
Shell/crown interior traces may differ when the endpoint coupling is valid;
both traces are retained, with gaps, overlaps and invalid constraints rejected.
