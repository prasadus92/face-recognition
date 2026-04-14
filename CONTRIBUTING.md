# Contributing

Thanks for your interest — contributions of all shapes (bug fixes, extractors, distance metrics, docs, benchmark datasets, deep-learning adapters) are very welcome.

By participating in this project you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Table of contents

- [Ground rules](#ground-rules)
- [Development setup](#development-setup)
- [Branching and commits](#branching-and-commits)
- [Coding standards](#coding-standards)
- [Testing](#testing)
- [Quality gates](#quality-gates)
- [Submitting a pull request](#submitting-a-pull-request)
- [Where to start](#where-to-start)
- [Security issues](#security-issues)

## Ground rules

1. Open an issue before large changes. A 10-line comment on an issue is cheaper than a 500-line rejected PR.
2. **One logical change per PR.** Keep diffs reviewable.
3. **Don't break the build.** CI runs the same gates locally and in GitHub Actions; if they go red, please fix or mark the PR draft.
4. **Tests or it didn't happen.** New behaviour needs unit tests; new REST endpoints need integration tests.
5. **No bundled model weights, no bundled cascade XML** unless they have an explicit compatible licence and you document its source.

## Development setup

### Prerequisites

- **JDK 17 or newer** (Temurin recommended).
- **Maven 3.9+**.
- Optional: Docker if you want to test the image, `pre-commit` if you want the local hooks.

### Clone and build

```bash
git clone https://github.com/prasadus92/face-recognition.git
cd face-recognition
mvn verify                       # runs the full quality gate (tests + checkstyle + spotbugs + jacoco)
```

### Run the server locally

```bash
mvn spring-boot:run
# or
java -jar target/face-recognition-*-exec.jar
```

### Run the CLI

```bash
java -jar target/face-recognition-*-exec.jar --help
```

### IDE setup

- **IntelliJ IDEA** — `File → Open` the project root; Maven is auto-detected. Enable annotation processing (Picocli + Spring Boot configuration metadata rely on it).
- **VS Code** — install the `Extension Pack for Java` and open the project folder.
- **Eclipse** — `File → Import → Maven → Existing Maven Projects`.

## Branching and commits

- Base all work on `master`.
- Use short, descriptive branch names: `feat/fisherfaces-lda`, `fix/knn-confidence`, `docs/onnx-setup`.
- Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) in commit subjects:
  ```
  feat(extraction): add calibrated KNN confidence normalizer
  fix(persistence): guard against truncated model files
  docs(readme): correct quick-start command
  ```
- Keep commits atomic — one reviewable idea each. Squash WIP commits before opening the PR.

## Coding standards

- **Java 17** source and target. Prefer records, pattern matching, `switch` expressions, `var` where it aids readability.
- **4-space indent**, LF line endings, UTF-8, 120 char soft limit — enforced by Checkstyle and `.editorconfig`.
- **K&R braces** on classes, methods and control structures (same-line opening brace).
- Public types and methods in `domain` / `application` / `api` layers have Javadoc with `@param`, `@return`, `@throws` as appropriate.
- **Clean-architecture dependency rule** — `domain` depends on nothing; `application` depends only on `domain`; `infrastructure` depends on `domain` (+ external libs); `api` composes the three. Don't introduce cycles.
- Keep methods small and cohesive; prefer immutable data where practical.
- **No `System.out.println`** — use SLF4J (`LoggerFactory.getLogger(MyClass.class)`).
- **No swallowed exceptions.** Wrap, rethrow, or handle meaningfully.

## Testing

- Unit tests live next to the code they cover under `src/test/java/...`.
- Use **JUnit 5** + **AssertJ** + **Mockito** (all provided by `spring-boot-starter-test`).
- REST changes must include a **MockMvc integration test** (`@SpringBootTest` / `@AutoConfigureMockMvc`) for happy and error paths.
- Algorithm changes should include a regression test on a small, deterministic dataset; put fixtures under `src/test/resources/datasets/`.
- Tag slow tests with `@Tag("slow")` so they can be filtered in CI.

Run targeted subsets:

```bash
mvn test                          # unit + integration
mvn -Dtest=EigenfacesExtractorTest test
mvn -P benchmarks exec:java       # run the benchmark harness
```

## Quality gates

`mvn verify` runs, and CI enforces:

- **JUnit** — all tests pass
- **JaCoCo** — line coverage must not drop below the configured floor (see `pom.xml`)
- **Checkstyle** — style rules in `config/checkstyle/checkstyle.xml`
- **SpotBugs** — static analysis, SpotBugs `HIGH` findings fail the build
- **CycloneDX** — SBOM generation as a side-effect of `verify`
- **CodeQL** — runs on every PR via GitHub Actions

Please run `mvn verify` locally before pushing.

## Submitting a pull request

1. Fork the repo and create your branch.
2. Make your change, add tests, run `mvn verify`.
3. Fill out the PR template — what changed, why, how it was tested.
4. Link the related issue (`Fixes #123`).
5. Keep the PR scope focused; split unrelated changes into separate PRs.
6. A maintainer will review. Please respond to feedback promptly and keep the thread civil — see the [Code of Conduct](CODE_OF_CONDUCT.md).

## Where to start

- Issues tagged `good first issue` — small, focused tickets.
- Issues tagged `help wanted` — larger items where guidance is available.
- Docs improvements: everything under `docs/`, `README.md`, `ROADMAP.md`.
- Add a new **distance metric** — implement `DistanceMetric`, add a test, wire it through `application.yml`.
- Add a new **benchmark dataset** adapter under `benchmark/`.

## Security issues

Do **not** file public issues for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the private disclosure channel.

---

Thanks again — small contributions compound. If anything about the setup or process is unclear, open a Discussion and I'll fix the docs.
