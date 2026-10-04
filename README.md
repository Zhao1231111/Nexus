# GPT 6 Astra as an Embodied Policy

Evaluation code and a bilingual technical report by Jiayi Su, Yixin Zheng, Mi Yan, Li Yi, Zhizheng Zhang, and He Wang. Jiayi Su and Yixin Zheng contributed equally; Li Yi, Zhizheng Zhang, and He Wang are corresponding authors.

- [Report and complete video gallery](https://anonymous-report-421.github.io/public-website/)
- [Production website repository](https://github.com/anonymous-report-421/public-website)
- [Citation](CITATION.bib)

We evaluate GPT 6 Astra Direct and a hybrid policy that reviews and optionally corrects π₀.₅ actions on ten RoboDojo tasks, with five aligned cases per task. The hybrid policy achieves a mean Score of **62.60** and **48%** success, with GPT corrections on **14.4% of executed control steps**. GPT 6 Astra Direct achieves **26%** success and a mean Score of **37.81** over the 48 episodes with available native scores. Both success rates use all 50 selected cases. Official-model comparisons are reweighted public references, not same-seed reruns.

## Repository layout

```text
README.md                     project scope, setup and this code guide
requirements-tools.txt        lightweight offline/report tooling dependencies
public_results/               published RoboDojo cases, scores and provenance
report_web/                   ready-to-serve static report and article clips
licenses/                     third-party notices and license texts
docs/                         build and release handoff notes

hybrid_rollout/
  robodojo/                   primary RoboDojo evaluation implementation
    cluster_entrypoint.sh     validate a cluster job and enter the batch runner
    batch.py                  reserve cases and isolate sequential episodes
    episode_entrypoint.sh     create one private per-episode runtime
    run_local.sh              launch and supervise policy, simulator and controller
    evaluation.py             freeze and verify paired evaluation cases
    case_ledger.py            reconstruct case status from persisted artifacts
    campaign.py               explicitly approved multi-slot cluster dispatcher
    pi05_server/              OpenPI/JAX policy RPC service and client
    robodojo_server/          simulator RPC, stepping, kinematics and decision logs
    skill/                    Codex tool contract, prompt and rollout state machine
    codex_backend/            isolated authentication profiles and config validation
    eval_panels/              frozen evaluation-panel manifests
  robolab/                    independent RoboLab adapter with the same core pattern
    pi05_server/              RoboLab policy RPC service and client
    robolab_server/           simulator RPC and single-arm control implementation
    skill/                    RoboLab Codex tool contract and controller
  report_site/
    app/                      React/Vite bilingual report application
    build.py                  private-archive evidence extraction pipeline
    package*.py               static report and article-clip packaging
    preview.py                byte-range-capable local static server
    *_smoke.cjs, test_*.py    browser and Python report checks
  annotation_ui/              browser UI for human video/result annotation
  annotation_dashboard.py     versioned annotation API and export server
  dashboard.py                read-only live rollout dashboard
  history_dashboard.py        read-only historical artifact browser
  report_server.py            static report server with optional review link
```

Generated caches, simulator assets and model checkpoints are deliberately outside
this tree. The checked-in integration code expects their locations to be supplied
through environment variables.

## Code architecture

### RoboDojo evaluation path

RoboDojo is the primary implementation and has four deliberately separated
responsibilities:

1. **Job and case isolation.** `cluster_entrypoint.sh` validates executables,
   authentication, the frozen evaluation manifest and the runtime environment.
   It then enters `batch.py`, which atomically reserves an explicit case, launches
   one `episode_entrypoint.sh` process, persists its outcome and verifies that owned
   descendants and ports have been released before another episode can begin.
2. **Process supervision.** `run_local.sh` creates a fresh archive, snapshots the
   source and launch metadata, and starts the services used by one episode. It owns
   only those child processes and propagates controller or artifact-verification
   failures to the caller.
3. **Policy and simulator services.** For the hybrid method,
   `pi05_server/server.py` loads the fixed OpenPI/JAX checkpoint and exposes π₀.₅
   inference through a loopback RPC client. `robodojo_server/server.py` embeds the
   native Isaac Sim/RoboDojo environment. `robodojo_server/session.py` owns reset,
   observation, forward-kinematics preview, bounded stepping, native termination
   checks and simulator-side records. The GPT-only method omits the π₀.₅ service.
4. **Codex controller.** `skill/run.py` starts an isolated Codex app-server session
   and publishes a small set of rollout tools. `robodojo_server/client.py` implements
   the hybrid state machine; `gpt_only_client.py` implements the direct baseline.
   Validation modules reject stale request IDs, invalid action shapes and unsafe or
   out-of-order tool calls before they can reach the simulator.

The hybrid control loop is:

```text
camera/proprioception observation
              |
              v
       pi05_infer (π₀.₅)
              |
              v
 joint-action proposal + FK preview
              |
              v
   GPT 6 Astra gate assessment
       /                    \
execute 1-15 student   execute 1-5 bounded
actions unchanged      dual-arm EEF corrections
       \                    /
              v
 native success/timeout checks -> next observation
```

The direct baseline follows the same observation and native-evaluation contracts,
but `GPTOnlyTools` emits bounded dual-arm end-effector targets without starting or
querying π₀.₅. In both methods, the simulator is the authority for executed actions,
termination and success; model text is not treated as an evaluation result.

The controller enforces a file-backed state machine:

```text
robodojo_start -> pi05_infer -> robodojo_execute -> ... -> result.json
                                      (hybrid)

robodojo_start -> robodojo_act ------------------> ... -> result.json
                                      (GPT-only)
```

Each step writes its inputs and outputs before advancing. A completed archive
therefore includes the resolved configuration, source/checkpoint identity,
observations, proposals, model decisions, executed actions, native evaluation,
token usage, logs and an `artifact_manifest.json` inventory. `case_ledger.py` and
`paired_evaluation.py` derive evaluation status from those artifacts rather than
from scheduler exit codes alone.

### Frozen cases and campaign code

`evaluation.py` defines the ten-task panel and binds every case to a task variant,
published scene layout, layout ID and evaluation seed. The JSON files in
`robodojo/eval_panels/` are immutable inputs to a run; their digest is checked at
startup. `public_results/evaluation_cases.json` is the public selection used for
the report.

The modules named `campaign.py`, `elastic_dispatch.py`, `overnight.py`,
`*_recovery.py` and `*_audit.py` are operations tooling around the same immutable
case and artifact contracts. They allocate explicit cluster slots, reconcile
interrupted work and audit persisted runs. They do not change the controller's
action semantics. `acp.py` is a SenseCore-specific adapter and is optional.

### RoboLab integration

`hybrid_rollout/robolab/` is a separate integration, not a compatibility layer over
RoboDojo. Its `run_local.sh` launches an OpenPI server, a RoboLab simulator server
and a Codex controller, while `robolab_server/client.py` keeps the same explicit
`start -> infer -> execute` scheduling model. RoboLab uses its own observation
schema, single-arm end-effector controller, task setup and Isaac Sim environment.
Code under the two integrations is intentionally duplicated where their runtime
contracts differ, so changing one simulator adapter does not silently alter the
other benchmark.

### Report and review tooling

The report has a source form and a published form:

- `report_site/app/` contains the React 19/Vite application. `src/main.jsx` mounts
  the app, `src/content/report/` implements the bilingual paper-style report and
  `src/content/dashboard/` implements the data view. The reviewed data snapshot is
  imported from `src/data.json`; report copy and publication metadata are kept in
  adjacent JSON files.
- `report_site/build.py` extracts and validates evidence from the original rollout
  archives. It is an offline release-production tool and cannot be reproduced from
  this history-free public snapshot alone.
- `report_site/package.py` and `package_article.py` assemble static deliverables.
  `report_site/preview.py` serves them locally with HTTP byte-range support, which
  is required for reliable video seeking.
- `report_web/` is the already-built site. `public_results/` is the smaller,
  machine-readable public result set; provenance files record how it was selected.
- `dashboard.py` and `history_dashboard.py` are read-only artifact viewers.
  `annotation_dashboard.py` stores human review in a separate versioned annotation
  root and never imports or advances the policy or simulator.

### Where to start reading

For the shortest path through the implementation, read:

1. `hybrid_rollout/robodojo/run_local.sh` for process boundaries and artifacts.
2. `hybrid_rollout/robodojo/skill/run.py` for the Codex session and tool dispatch.
3. `hybrid_rollout/robodojo/robodojo_server/client.py` for the hybrid control loop,
   or `gpt_only_client.py` for the direct baseline.
4. `hybrid_rollout/robodojo/robodojo_server/session.py` for authoritative simulator
   stepping and native evaluation.
5. `hybrid_rollout/report_site/app/src/content/report/ReportContent.jsx` for how
   the published results and qualitative evidence are presented.

This is a history-free source release. It contains no credentials, private annotations, raw model sessions, original operations logs, simulator assets or model checkpoints. The website repository hosts 100 RoboDojo evaluation videos and 43 gallery clips. The report contains one extra historical demonstration clip used only for qualitative analysis. The [RoboLab gallery](https://anonymous-report-421.github.io/public-website/robolab-gallery.html) contains 150 recordings (three methods, ten tasks, five trials); its unchanged original MP4s are hosted in the website repository's [video release](https://github.com/anonymous-report-421/public-website/releases/tag/robolab-gallery-20260914) to keep the static Pages site below its size limit. The gallery manifest records each video's URL, size and SHA-256.

## Preview the report

From the repository root:

```sh
python3 -m hybrid_rollout.report_site.preview --directory report_web --port 8768
```

Open `http://127.0.0.1:8768/`. The preview server supports video byte ranges. English is the default; the language button or `?lang=zh` switches to Chinese. The static `report_web/` folder can also be hosted on a standard HTTP server.

## Rebuild the report

Requires Node.js compatible with the locked Vite version (Node 24 was used for this release).

```sh
cd hybrid_rollout/report_site/app
npm ci --ignore-scripts --no-audit --no-fund
npm run build
```

The build verifies the preserved shared runtime and produces a self-contained `dist/index.html`. Copy that file to `report_web/index.html`, keeping `data.json`, `scores.csv`, `citation.bib`, licenses and media alongside it. The public authoring snapshot is included; no private experiment storage is needed for this rebuild. Full gallery media are in the separate website repository. Do not run `build.py` merely to rebuild the frontend: that offline evidence-extraction utility requires the original, non-public trajectory archives.

## Run a new evaluation

The integrations depend on externally installed simulator and policy environments. They are **not** a standalone simulator distribution. The published RoboDojo results used Isaac Sim 5.1; the current local compatibility entry points require the external RoboDojo Isaac Sim 6.0.1 / IsaacLab 3.0 branch. See the [local launch guide](hybrid_rollout/robodojo/README.md) for the verified camera configuration and smoke command. RoboDojo also needs matching source/assets and, for the hybrid policy, its π₀.₅ OpenPI/JAX checkpoint. RoboLab has its own environment. Obtain these from their upstream projects under their own terms. See `hybrid_rollout/robodojo/SOURCE.json` for the recorded historical revisions and checkpoint identity.

1. Prepare your simulator environment, OpenPI/JAX environment, checkpoint and Codex CLI. Keep these separate from the lightweight tools environment.
2. Review `hybrid_rollout/robodojo/cluster.env.example` and set your paths, image, user/group IDs, credentials file paths and explicit case identity. Public export paths are generic examples, not runnable cluster credentials.
3. Use `hybrid_rollout.robodojo.codex_backend.profiles` to manage **isolated** rollout authentication. Never commit keys or `auth.json`, and do not modify your interactive Codex home. Inspect the CLI with `--help` before performing login or lease operations.
4. The container entry is `hybrid_rollout/robodojo/cluster_entrypoint.sh`; `acp.py` is an optional SenseCore-specific submission adapter. A container runs one explicit case, saves artifacts, and cleans up its own components. Cluster-specific campaign migration helpers require deployment-local plans and are not a one-command public launch recipe.
5. Use `public_results/evaluation_cases.json` to align task, scene and seed across policies. Each case has one hybrid and one Direct result. Set `ROLLOUT_EVALUATION_METHOD=pi05_plus_gpt` or `gpt_only`; the latter is an internal CLI identifier for GPT 6 Astra Direct and does not launch π₀.₅.

The model is fixed to `gpt-6-astra` with `xhigh` reasoning. No provider fallback is implied. You need your own authorized account or gateway access. Policy/model execution can incur costs. Read the skills and action contract before launching; no simulation or model calls are made by the preview or offline tests.

### Deployment and security boundaries

Gateway URLs in `codex_backend/profiles.py` and `codex_backend/config.toml` use reserved `.invalid` placeholders. Configure your own authorized endpoint explicitly in the selected profile and matching configuration, and set `ROLLOUT_GATEWAY_NO_PROXY` for your network if needed. Historical profile identifiers are retained only for compatibility; no company or relay service is configured by this release. The optional SenseCore adapter uses that platform's public control-plane endpoints, not credentials.

The annotation/review tools are intended for a trusted local network, not unauthenticated Internet hosting. Keep the default loopback binding, or put an authenticated reverse proxy in front of any LAN-accessible service. Public GitHub Pages serves only the compiled static report and selected media, never the annotation API or rollout services. Codex tool execution and simulator RPC belong in an isolated, trusted environment; do not supply untrusted programs, serialized RPC data or trajectory archives. Store credentials outside this repository, with restricted permissions. The checked-in token/proxy examples in tests are synthetic fixtures.

## Offline tests

Install `requirements-tools.txt` in a dedicated virtual environment. Simulator and OpenPI dependencies must be installed separately in their respective environments.

```sh
python3 -m unittest hybrid_rollout.report_site.test_i18n hybrid_rollout.report_site.test_editorial_revision -v
python3 -m pytest hybrid_rollout/robodojo/test_proposal_diagnostics.py hybrid_rollout/robodojo/test_prompt_context.py hybrid_rollout/robodojo/test_network_continue.py
```

Tests that require original shared storage, a proprietary simulator or historical operations fixtures cannot be reproduced solely from this public snapshot. The release has not rerun paid model or GPU experiments.

## Licensing and citation

Project-owned code is MIT licensed. Third-party components, fonts, benchmark assets and logos retain their own terms; see `THIRD_PARTY_NOTICES.md`. Model weights and simulator assets are not redistributed.

If this report or code is useful to your research, please cite:

```bibtex
@misc{su2026astra,
  title        = {{GPT 6 Astra} as an Embodied Policy},
  author       = {Su, Jiayi and Zheng, Yixin and Yan, Mi and Yi, Li and Zhang, Zhizheng and Wang, He},
  year         = {2026},
  howpublished = {Technical report and code},
  url          = {https://github.com/anonymous-report-421/eval-of-gpt-6-astra-as-policy}
}
```
