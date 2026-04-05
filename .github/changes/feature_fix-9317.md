# fix: resolve memory leak in data loader

## Summary
- Fix buffer allocation not being released between batches
- Add explicit cleanup in finally blocks
- Add memory usage assertions in tests

## Test plan
- [x] Reproduce memory leak scenario
