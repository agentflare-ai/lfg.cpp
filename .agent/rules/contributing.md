---
trigger: always_on
---

## Testing
- ALWAYS run existing tests before starting any work to establish a baseline
- ALWAYS write tests achieving >90% coverage for new code
- ALWAYS execute tests when completing work to verify new tests pass and no regressions occur
- NEVER declare work complete without a passing test suite

## Linting & Formatting
- ALWAYS check for linting errors before marking work complete
- ALWAYS fix all linting warnings—never leave them for later
- ALWAYS run formatters to ensure consistent code style

## Code Hygiene
- ALWAYS remove unused code, imports, and files when completing work
- ALWAYS prefer breaking changes over deprecation and dead code accumulation
- NEVER leave deprecated code paths when removal is feasible
- ALWAYS delete rather than comment out unused code