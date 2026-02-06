---
trigger: glob
globs: src/**/*.c*, src/**/*.h*
---

## Memory & RAII
- ALWAYS use RAII—acquire resources in constructors, release in destructors
- ALWAYS use smart pointers over raw owning pointers
- NEVER call `new`/`delete` directly—use `std::make_unique`/`std::make_shared`
- ALWAYS use `std::span` or `std::string_view` for non-owning views

## Constructors & Dependency Injection
- ALWAYS accept dependencies by `const&` in constructors, not raw pointers
- ALWAYS inject dependencies via constructor parameters, not global/singleton access
- NEVER access global state from within classes—pass it in explicitly
- ALWAYS mark single-argument constructors `explicit`

## Performance
- ALWAYS pass large objects by `const&`, return by value (RVO applies)
- ALWAYS use `std::move` when transferring ownership
- ALWAYS reserve `std::vector` capacity when size is known
- ALWAYS prefer stack allocation over heap when lifetime permits
- NEVER use `std::endl`—use `'\n'`

## Return Types
- ALWAYS return `std::optional<T>` instead of `nullptr` or sentinel values for "may not exist"
- ALWAYS return `std::expected<T, E>` instead of `nullptr` or error codes for "may fail"
- ALWAYS use `.value_or()` when a default is acceptable

## Error Handling
- ALWAYS use `std::expected<T, E>` over exceptions for recoverable errors in performance-critical code
- ALWAYS define meaningful error types—never use `std::expected<T, std::string>` for complex error cases
- NEVER use `.value()` in hot paths—use `operator*` after explicit check
- ALWAYS use monadic operations (`.and_then()`, `.transform()`, `.or_else()`) to chain operations
- NEVER mix exceptions and `std::expected` for the same error domain

## Type Safety
- NEVER use C-style casts—use `static_cast` etc.
- ALWAYS initialize all variables at declaration
- ALWAYS use `enum class` over unscoped enums

## Concurrency
- ALWAYS use `std::scoped_lock` over manual lock/unlock
- NEVER access shared mutable state without synchronization

## Code Quality
- ALWAYS use `[[nodiscard]]` where ignoring return values is a bug
- ALWAYS treat warnings as errors (`-Werror`)

